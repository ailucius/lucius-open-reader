"""Chunk-wise requirement extraction with the reader model (Nemotron 3 Super by default)."""

from __future__ import annotations

import asyncio
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
  Do not paraphrase, fix spelling or join text from different places.
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
    on_chunk_done: Callable[[list[Page], int], Awaitable[None]] | None = None,
) -> ExtractOutcome:
    """Run every chunk through the reader model, `concurrency` at a time, in document order."""
    sem = asyncio.Semaphore(concurrency)
    outcome = ExtractOutcome()
    results: list[list[Requirement]] = [[] for _ in chunks]

    async def one(index: int, chunk: list[Page]) -> None:
        label = f"pages {chunk[0].number}-{chunk[-1].number}"
        if not any(p.text.strip() for p in chunk):
            if on_chunk_done:
                await on_chunk_done(chunk, 0)
            return
        async with sem:
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    raw = await llm.chat_json(
                        model=model,
                        system=EXTRACT_SYSTEM,
                        user=build_extract_prompt(chunk),
                        schema_name="requirements",
                        schema=EXTRACT_SCHEMA,
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
        if on_chunk_done:
            await on_chunk_done(chunk, len(results[index]))

    await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
    outcome.requirements = [r for chunk_reqs in results for r in chunk_reqs]
    return outcome
