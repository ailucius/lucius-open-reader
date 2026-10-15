"""Fetch -> pages -> extract -> verify -> decide, emitting progress events along the way."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .config import Settings
from .decide import decide
from .extract import extract_requirements
from .fetch import check_pdf_bytes, fetch_pdf
from .llm import LLMClient
from .models import Page, ReadStats, RunResult
from .pages import chunk_pages, extract_pages
from .verify import verify_requirements

Event = dict[str, Any]
Emit = Callable[[Event], Awaitable[None]]


async def _noop(_: Event) -> None:
    return None


async def run_reader(
    *,
    settings: Settings,
    llm: LLMClient,
    profile: str = "",
    url: str | None = None,
    pdf: bytes | None = None,
    source_name: str | None = None,
    emit: Emit = _noop,
) -> tuple[RunResult, bytes]:
    """Read one tender PDF end to end. Returns the result and the PDF bytes that were read."""
    if pdf is None:
        if not url:
            raise ValueError("give a PDF URL or a PDF file")
        await emit({"type": "stage", "stage": "fetch", "message": "Downloading the PDF"})
        pdf = await fetch_pdf(url, settings)
    else:
        if len(pdf) > settings.max_pdf_bytes:
            raise ValueError(f"PDF is larger than {settings.max_pdf_mb} MB")
        check_pdf_bytes(pdf)

    await emit({"type": "stage", "stage": "pages", "message": "Extracting page text"})
    pages: list[Page] = await asyncio.to_thread(extract_pages, pdf, settings.max_pages)
    empty = sum(1 for p in pages if len(p.text.strip()) < 20)
    chunks = chunk_pages(pages, settings.chunk_pages)
    stats = ReadStats(pages_total=len(pages), chunks_total=len(chunks), empty_pages=empty)
    await emit({"type": "pages", "total": len(pages), "empty": empty, "chunks": len(chunks)})

    found = 0
    done_pages = 0
    lock = asyncio.Lock()

    async def on_chunk(chunk: list[Page], n: int) -> None:
        nonlocal found, done_pages
        async with lock:
            found += n
            done_pages += len(chunk)
            await emit(
                {
                    "type": "progress",
                    "pages_read": done_pages,
                    "pages_total": len(pages),
                    "requirements": found,
                    "chunk": f"{chunk[0].number}-{chunk[-1].number}",
                }
            )

    await emit({"type": "stage", "stage": "read", "message": f"Reading with {settings.reader_model}"})
    outcome = await extract_requirements(chunks, llm, settings.reader_model, settings.concurrency, on_chunk)

    await emit({"type": "stage", "stage": "verify", "message": "Checking every quote against its page"})
    reqs = verify_requirements(outcome.requirements, pages, settings.verify_threshold)

    # merge exact duplicates (same page, same normalised quote); counted, not silent
    seen: set[tuple[int, str]] = set()
    unique = []
    for r in sorted(reqs, key=lambda r: r.page):
        key = (r.page, " ".join(r.quote.lower().split()))
        if key in seen and r.quote:
            stats.duplicates_merged += 1
            continue
        seen.add(key)
        unique.append(r)
    for i, r in enumerate(unique, start=1):
        r.id = f"R{i}"

    stats.pages_read = len(pages) - outcome.failed_pages
    stats.chunks_failed = outcome.failed_chunks
    stats.malformed_items = outcome.malformed
    stats.requirements = len(unique)
    stats.verified = sum(1 for r in unique if r.verified)
    stats.unverified = stats.requirements - stats.verified
    stats.pages_corrected = sum(1 for r in unique if r.cited_page is not None)
    await emit(
        {
            "type": "verified",
            "verified": stats.verified,
            "unverified": stats.unverified,
            "pages_corrected": stats.pages_corrected,
        }
    )

    result = RunResult(
        source=url or source_name or "upload.pdf",
        pdf_url=url,
        profile=profile,
        reader_model=settings.reader_model,
        decider_model=settings.decider_model,
        fake_llm=settings.fake_llm,
        stats=stats,
        requirements=unique,
    )
    if not profile.strip():
        result.decision_note = "Add a company profile to get a bid / no-bid call."
    elif not unique:
        result.decision_note = "No requirements were found, so there is nothing to decide on."
    else:
        await emit({"type": "stage", "stage": "decide", "message": f"Deciding with {settings.decider_model}"})
        decision, left_out = await decide(profile, unique, llm, settings.decider_model)
        result.decision = decision
        if left_out:
            result.decision_note = f"{left_out} optional requirements were left out of the decision prompt."
    return result, pdf
