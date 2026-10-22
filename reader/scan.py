"""Scanned pages: a page with no text layer is read from its image by NVIDIA Nemotron 3.5 Super VL.

The transcription becomes the page's text, for the reader and for the quote check, and every page
and requirement it touches says so: on a scanned page the quote check proves the quote is in the
vision model's transcription of the page image, not in a text layer. Each page is tried twice at
most, with a short pause first when the API answered 429 or 5xx (with a thinking effort other than
"none", a try whose answer comes back empty is asked once more with thinking off, so a page costs
four calls at most). A page that still fails is reported as not read, never filled in. At most
`cap` scanned pages are read per run (the first ones in the document); the rest are reported as
over the cap. All the scans of a run share one time limit (`deadline_s`): a page not finished by
then is stopped and reported as not read in time, so a hung endpoint cannot hold the run.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .llm import LLMClient
from .models import Page
from .pages import render_page

VISION_SYSTEM = """You transcribe one scanned page of a public tender document, for a reader that checks quotes word for word.
Copy every word exactly as printed, in reading order, line by line: headings, paragraphs, numbered and bulleted items, table rows (one row per line, cells separated by " | "), stamps, headers and footers.
Leave out line numbers printed in the margin (legal texts number every line); keep the numbering of items and clauses.
Do not translate, summarise, correct spelling or add anything that is not on the page. Write [illegible] for a word you cannot read.
If the page has no readable text, reply with exactly: NO TEXT
Reply with the transcription only."""

SCAN_DEADLINE_S = 120.0  # all the scans of one run
RETRY_PAUSE_S = 2.0  # before the second try, when the API said 429 or 5xx

_NO_TEXT = re.compile(r"^\W*no (readable )?text\W*$", re.IGNORECASE)
_WHOLE_FENCE = re.compile(r"^```[a-z]*\n(.*)\n```$", re.DOTALL)


def build_vision_prompt(page: Page) -> str:
    return f"Page {page.number} of the tender. Transcribe it."


def clean_transcription(text: str) -> str:
    """The model's reply as page text: fences removed, "NO TEXT" becomes an empty page."""
    text = text.strip()
    fence = _WHOLE_FENCE.match(text)
    if fence:
        text = fence.group(1).strip()
    return "" if _NO_TEXT.match(text) else text


@dataclass
class ScanOutcome:
    read: list[int] = field(default_factory=list)  # transcribed: page.text is now the transcription
    failed: list[int] = field(default_factory=list)  # two tries failed: not read
    timed_out: list[int] = field(default_factory=list)  # not finished within the time limit: not read
    errors: list[str] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)  # over the cap: not read


# on_page(page, why): why is None when the page was read, else "failed" or "time limit"
OnPage = Callable[[Page, str | None], Awaitable[None]]


def worth_a_pause(exc: Exception | None) -> bool:
    """The API was busy or broken (429, 5xx): wait a moment before trying again."""
    status = getattr(exc, "status", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def plan_scans(pages: list[Page], cap: int) -> tuple[list[Page], list[int]]:
    """(scanned pages to read, numbers of scanned pages over the cap), in document order."""
    scans = [p for p in pages if p.scan]
    return scans[:cap], [p.number for p in scans[cap:]]


async def read_scans(
    pdf: bytes,
    pages: list[Page],
    llm: LLMClient,
    model: str,
    *,
    cap: int = 40,
    dpi: int = 150,
    concurrency: int = 4,
    effort: str | None = None,
    deadline_s: float = SCAN_DEADLINE_S,
    pause_s: float = RETRY_PAUSE_S,
    on_page: OnPage | None = None,
) -> ScanOutcome:
    """Transcribe the scanned pages in place, `concurrency` at a time, within `deadline_s` seconds
    for all of them (0: no limit)."""
    todo, skipped = plan_scans(pages, cap)
    outcome = ScanOutcome(skipped=skipped)
    sem = asyncio.Semaphore(concurrency)
    settled: set[int] = set()  # pages whose outcome is recorded

    async def one(page: Page) -> None:
        last_error: Exception | None = None
        text: str | None = None
        async with sem:
            try:
                image, mime = await asyncio.to_thread(render_page, pdf, page.number, dpi)
            except Exception as exc:  # a page pymupdf cannot draw is a page not read
                last_error = exc
            else:
                for attempt in range(2):
                    if attempt and worth_a_pause(last_error):
                        await asyncio.sleep(pause_s)
                    try:
                        reply = await llm.chat_image(
                            model=model,
                            system=VISION_SYSTEM,
                            user=build_vision_prompt(page),
                            image=image,
                            mime=mime,
                            effort=effort,
                        )
                        text, last_error = clean_transcription(reply), None
                        break
                    except Exception as exc:  # network, API or empty reply: retry once, then record
                        last_error = exc
        # no await from here to the record: a page stopped by the time limit is either read or not
        if text is not None:
            page.text, page.transcribed = text, True
            outcome.read.append(page.number)
        else:
            outcome.failed.append(page.number)
            outcome.errors.append(f"page {page.number}: {last_error}")
        settled.add(page.number)
        if on_page:
            await on_page(page, None if text is not None else "failed")

    tasks = [asyncio.create_task(one(p)) for p in todo]
    try:
        if tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=deadline_s if deadline_s > 0 else None, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]  # an outright failure, not a page error
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for page in todo:
                if page.number not in settled:
                    outcome.timed_out.append(page.number)
                    outcome.errors.append(f"page {page.number}: not read within the {deadline_s:g} s limit for scans")
                    if on_page:
                        await on_page(page, "time limit")
    finally:  # whatever stops this function, no page task is left running
        for task in tasks:
            task.cancel()
    outcome.read.sort()
    outcome.failed.sort()
    outcome.errors.sort(key=lambda e: int(e.split()[1].rstrip(":")))
    return outcome
