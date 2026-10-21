"""Fetch -> pages -> scans -> extract -> verify -> decide, emitting progress events along the way."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .decide import decide
from .extract import extract_requirements
from .fetch import check_pdf_bytes, fetch_pdf
from .llm import LLMClient
from .models import Page, ReadStats, RunResult
from .pages import chunk_pages, extract_pages
from .scan import plan_scans, read_scans
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
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
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
    # blank: no text and no page image to read it from (scans are read below, or reported as not read)
    empty_numbers = [p.number for p in pages if not p.scan and len(p.text.strip()) < 20]
    empty = len(empty_numbers)
    to_scan, over_cap = plan_scans(pages, settings.max_scan_pages)
    chunks = chunk_pages(pages, settings.chunk_pages)
    stats = ReadStats(
        pages_total=len(pages),
        chunks_total=len(chunks),
        empty_pages=empty,
        empty_page_numbers=empty_numbers,
        scan_cap=settings.max_scan_pages,
    )
    await emit(
        {
            "type": "pages",
            "total": len(pages),
            "empty": empty,
            "empty_pages": empty_numbers,
            "chunks": len(chunks),
            "scans": [p.number for p in to_scan],
            "scans_over_cap": over_cap,
            "scan_cap": settings.max_scan_pages,
        }
    )

    unread_scans: set[int] = set(over_cap)
    if to_scan:
        await emit(
            {
                "type": "stage",
                "stage": "scan",
                "message": f"Reading {len(to_scan)} scanned page{'s' if len(to_scan) > 1 else ''} "
                f"from the page image with {settings.vision_model}",
            }
        )
        scans_done = 0
        scan_lock = asyncio.Lock()

        async def on_scan(page: Page, ok: bool) -> None:
            nonlocal scans_done
            async with scan_lock:
                scans_done += 1
                await emit(
                    {
                        "type": "scan",
                        "page": page.number,
                        "ok": ok,
                        "chars": len(page.text) if ok else 0,
                        "done": scans_done,
                        "total": len(to_scan),
                    }
                )

        scanned = await read_scans(
            pdf,
            pages,
            llm,
            settings.vision_model,
            cap=settings.max_scan_pages,
            dpi=settings.scan_dpi,
            concurrency=settings.concurrency,
            effort=settings.vision_effort,
            on_page=on_scan,
        )
        stats.scanned_pages = scanned.read
        stats.scan_failed_pages = scanned.failed
        unread_scans.update(scanned.failed)
    stats.scan_skipped_pages = over_cap

    found = 0
    done_pages = 0
    read_pages = 0
    failed_numbers: set[int] = set()
    lock = asyncio.Lock()

    async def on_chunk(chunk: list[Page], n: int, ok: bool) -> None:
        nonlocal found, done_pages, read_pages
        async with lock:
            found += n
            done_pages += len(chunk)
            if ok:
                read_pages += sum(1 for p in chunk if p.number not in unread_scans)
            else:
                failed_numbers.update(p.number for p in chunk)
            await emit(
                {
                    "type": "progress",
                    "pages_done": done_pages,
                    "pages_read": read_pages,
                    "pages_failed": len(failed_numbers),
                    "pages_total": len(pages),
                    "requirements": found,
                    "chunk": f"{chunk[0].number}-{chunk[-1].number}",
                    "ok": ok,
                }
            )

    await emit({"type": "stage", "stage": "read", "message": f"Reading with {settings.reader_model}"})
    outcome = await extract_requirements(
        chunks, llm, settings.reader_model, settings.concurrency, on_chunk, effort=settings.reader_effort
    )

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
    transcribed = {p.number for p in pages if p.transcribed}
    for i, r in enumerate(unique, start=1):
        r.id = f"R{i}"
        r.scanned = r.page in transcribed

    stats.pages_read = len(pages) - len(failed_numbers | unread_scans)
    stats.chunks_failed = outcome.failed_chunks
    stats.malformed_items = outcome.malformed
    stats.requirements = len(unique)
    stats.verified = sum(1 for r in unique if r.verified)
    stats.unverified = stats.requirements - stats.verified
    stats.pages_corrected = sum(1 for r in unique if r.cited_page is not None)
    stats.joined_quotes = sum(1 for r in unique if r.joined)
    stats.thinking_retries = getattr(llm, "thinking_retries", 0)
    await emit(
        {
            "type": "verified",
            "requirements": stats.requirements,  # after exact duplicates are merged
            "duplicates_merged": stats.duplicates_merged,
            "verified": stats.verified,
            "unverified": stats.unverified,
            "pages_corrected": stats.pages_corrected,
            "joined": stats.joined_quotes,
            "scanned": sum(1 for r in unique if r.scanned),
        }
    )

    result = RunResult(
        source=url or source_name or "upload.pdf",
        pdf_url=url,
        profile=profile,
        reader_model=settings.reader_model,
        decider_model=settings.decider_model,
        vision_model=settings.vision_model if to_scan else None,
        fake_llm=settings.fake_llm,
        read_at=started_at,
        stats=stats,
        requirements=unique,
    )
    if not profile.strip():
        result.decision_note = "Add a company profile to get a bid / no-bid call."
    elif not unique:
        result.decision_note = "No requirements were found, so there is nothing to decide on."
    else:
        await emit({"type": "stage", "stage": "decide", "message": f"Deciding with {settings.decider_model}"})
        decision, unassessed = await decide(
            profile,
            unique,
            llm,
            settings.decider_model,
            settings.decider_effort,
            pages=pages,
            concurrency=settings.concurrency,
        )
        result.decision = decision
        if unassessed:
            result.decision_note = f"The decider gave no judgment for {unassessed} mandatory requirements."
    return result, pdf
