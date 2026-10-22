"""Scanned pages: a page with no text layer is read from its image by NVIDIA Nemotron 3.5 Super VL.

The transcription becomes the page's text, for the reader and for the quote check, and every page
and requirement it touches says so: on a scanned page the quote check proves the quote is in the
vision model's transcription of the page image, not in a text layer. Each page is asked twice at
most; a page that still fails is reported as not read, never filled in. At most `cap` scanned pages
are read per run (the first ones in the document); the rest are reported as over the cap.
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
    errors: list[str] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)  # over the cap: not read


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
    on_page: Callable[[Page, bool], Awaitable[None]] | None = None,
) -> ScanOutcome:
    """Transcribe the scanned pages in place, `concurrency` at a time."""
    todo, skipped = plan_scans(pages, cap)
    outcome = ScanOutcome(skipped=skipped)
    sem = asyncio.Semaphore(concurrency)

    async def one(page: Page) -> None:
        last_error: Exception | None = None
        async with sem:
            try:
                image, mime = await asyncio.to_thread(render_page, pdf, page.number, dpi)
            except Exception as exc:  # a page pymupdf cannot draw is a page not read
                last_error = exc
            else:
                for _attempt in range(2):
                    try:
                        reply = await llm.chat_image(
                            model=model,
                            system=VISION_SYSTEM,
                            user=build_vision_prompt(page),
                            image=image,
                            mime=mime,
                            effort=effort,
                        )
                        page.text = clean_transcription(reply)
                        page.transcribed = True
                        last_error = None
                        break
                    except Exception as exc:  # network, API or empty reply: retry once, then record
                        last_error = exc
        if last_error is None:
            outcome.read.append(page.number)
        else:
            outcome.failed.append(page.number)
            outcome.errors.append(f"page {page.number}: {last_error}")
        if on_page:
            await on_page(page, last_error is None)

    tasks = [asyncio.create_task(one(p)) for p in todo]
    try:
        await asyncio.gather(*tasks)
    finally:  # if one page's task fails outright (not a page error), the rest are not left running
        for task in tasks:
            task.cancel()
    outcome.read.sort()
    outcome.failed.sort()
    outcome.errors.sort(key=lambda e: int(e.split()[1].rstrip(":")))
    return outcome
