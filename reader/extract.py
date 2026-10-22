"""Chunk-wise requirement extraction with the reader model (Nemotron 3 Super by default)."""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .models import CATEGORIES, Page, Requirement

MAX_CHARS_PER_PAGE = 12_000

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["requirement", "quote", "page", "category", "mandatory"],
                "properties": {
                    "requirement": {"type": "string"},
                    "quote": {"type": "string"},
                    "page": {"type": "integer"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "mandatory": {"type": "boolean"},
                },
            },
        }
    },
}

EXTRACT_SYSTEM = """You read public tender documents for a company deciding whether to bid.
From the pages given, list every requirement a bidder must meet, every document or certificate
they must provide, and every rule of the submission (deadline, format, place, validity, guarantees).

For each item return:
- requirement: one plain-English sentence saying what the bidder must do or have.
- quote: text copied VERBATIM from the page that states it. One contiguous span, at most 40 words.
  Do not paraphrase, fix spelling, skip words in the middle or join text from different places.
  When a requirement sits in a list, quote the list item itself rather than the lead-in plus the item.
- page: the number from the "=== PAGE n ===" header of the page the quote is on.
- category: eligibility, technical, commercial, legal, submission or other.
- mandatory: true if the text makes it compulsory (must, shall, required, will be disqualified);
  false if it is optional, preferred or scored only.

Only use the pages given. Never invent requirements. If there are none, return an empty list.
Return JSON only."""


def build_extract_prompt(chunk: list[Page]) -> str:
    first, last = chunk[0].number, chunk[-1].number
    parts = [f"Tender pages {first} to {last}.", ""]
    for page in chunk:
        parts.append(f"=== PAGE {page.number} ===")
        parts.append(page.text[:MAX_CHARS_PER_PAGE].strip())
        parts.append("")
    parts.append("=== END OF PAGES ===")
    return "\n".join(parts)


class OrderedSlots:
    """A semaphore that hands a free slot to the waiting chunk earliest in the document, not to the
    one that asked first. A chunk that waited for its scanned pages to be transcribed is read next,
    instead of after every chunk that queued while it waited (which made it the slow tail of a run)."""

    def __init__(self, size: int) -> None:
        self._free = size
        self._waiting: list[tuple[int, int, asyncio.Future[None]]] = []
        self._order = itertools.count()

    async def acquire(self, key: int) -> None:
        if self._free > 0 and not self._waiting:
            self._free -= 1
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (key, next(self._order), future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():  # the slot came with the cancellation: pass it on
                self.release()
            raise

    def release(self) -> None:
        while self._waiting:
            _, _, future = heapq.heappop(self._waiting)
            if not future.done():
                future.set_result(None)
                return
        self._free += 1


@dataclass
class ExtractOutcome:
    requirements: list[Requirement] = field(default_factory=list)
    malformed: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    failed_pages: int = 0


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "0"}:
        return False
    return None


def parse_items(raw: dict[str, Any], chunk: list[Page], label: str) -> tuple[list[Requirement], int]:
    """Validate model items. Returns (requirements, malformed_count). Nothing is dropped unless unusable."""
    items = raw.get("requirements")
    if not isinstance(items, list):
        return [], 1
    lo, hi = chunk[0].number, chunk[-1].number
    out: list[Requirement] = []
    malformed = 0
    for item in items:
        if not isinstance(item, dict):
            malformed += 1
            continue
        text = str(item.get("requirement") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not text and not quote:
            malformed += 1
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            page = lo
        page = min(max(page, lo), hi)  # the model only saw this chunk
        category = str(item.get("category") or "other").strip().lower()
        if category not in CATEGORIES:
            category = "other"
        mandatory = _as_bool(item.get("mandatory"))
        out.append(
            Requirement(
                requirement=text or quote,
                quote=quote,
                page=page,
                category=category,  # type: ignore[arg-type]
                mandatory=True if mandatory is None else mandatory,
                source_chunk=label,
            )
        )
    return out, malformed


async def extract_requirements(
    chunks: list[list[Page]],
    llm: LLMClient,
    model: str,
    concurrency: int = 4,
    on_chunk_done: Callable[[list[Page], int, bool], Awaitable[None]] | None = None,
    effort: str | None = None,
    before_chunk: Callable[[list[Page]], Awaitable[None]] | None = None,
) -> ExtractOutcome:
    """Run every chunk through the reader model, `concurrency` at a time, in document order.

    `before_chunk` is awaited before a chunk asks for a reader slot (the pipeline uses it to wait for
    the chunk's scanned pages to be transcribed), so a waiting chunk never holds a slot; free slots go
    to the waiting chunk earliest in the document."""
    slots = OrderedSlots(concurrency)
    outcome = ExtractOutcome()
    results: list[list[Requirement]] = [[] for _ in chunks]

    async def one(index: int, chunk: list[Page]) -> None:
        if before_chunk:
            await before_chunk(chunk)
        label = f"pages {chunk[0].number}-{chunk[-1].number}"
        if not any(p.text.strip() for p in chunk):
            if on_chunk_done:
                await on_chunk_done(chunk, 0, True)
            return
        await slots.acquire(index)
        try:
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    raw = await llm.chat_json(
                        model=model,
                        system=EXTRACT_SYSTEM,
                        user=build_extract_prompt(chunk),
                        schema_name="requirements",
                        schema=EXTRACT_SCHEMA,
                        effort=effort,
                    )
                    reqs, bad = parse_items(raw, chunk, label)
                    results[index] = reqs
                    outcome.malformed += bad
                    last_error = None
                    break
                except Exception as exc:  # network, API or JSON errors: retry once, then record
                    last_error = exc
            if last_error is not None:
                outcome.failed_chunks.append(f"{label}: {last_error}")
                outcome.failed_pages += len(chunk)
        finally:
            slots.release()
        if on_chunk_done:
            await on_chunk_done(chunk, len(results[index]), last_error is None)

    await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
    outcome.requirements = [r for chunk_reqs in results for r in chunk_reqs]
    return outcome
