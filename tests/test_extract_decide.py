from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest

from reader.decide import decide, deciding_from, parse_assessments, parse_scope, verdict_from
from reader.extract import extract_requirements, parse_items
from reader.llm import FakeLLM, LLMError
from reader.models import Page, Requirement, Scope
from reader.pages import chunk_pages, extract_pages


class ScriptedLLM:
    """Returns canned replies and records the highest number of concurrent calls."""

    def __init__(self, reply: dict[str, Any] | Exception, delay: float = 0.01) -> None:
        self.reply = reply
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_parse_items_clamps_pages_and_counts_malformed() -> None:
    chunk = [Page(number=7, text="a"), Page(number=8, text="b")]
    raw = {
        "requirements": [
            {"requirement": "R", "quote": "Q", "page": 99, "category": "Technical", "mandatory": "yes"},
            {"requirement": "R2", "quote": "Q2", "page": "7", "category": "weird", "mandatory": False},
            "not a dict",
            {"requirement": "", "quote": ""},
        ]
    }
    reqs, bad = parse_items(raw, chunk, "pages 7-8")
    assert bad == 2
    assert [r.page for r in reqs] == [8, 7]
    assert [r.category for r in reqs] == ["technical", "other"]
    assert [r.mandatory for r in reqs] == [True, False]


async def test_extract_respects_concurrency_limit() -> None:
    pages = [Page(number=i, text=f"page {i}") for i in range(1, 37)]
    llm = ScriptedLLM({"requirements": []})
    await extract_requirements(chunk_pages(pages, 6), llm, "m", concurrency=2)
    assert llm.calls == 6
    assert llm.peak <= 2


async def test_extract_records_failed_chunks_after_one_retry() -> None:
    pages = [Page(number=i, text="x") for i in range(1, 7)]
    llm = ScriptedLLM(LLMError("boom"))
    outcome = await extract_requirements(chunk_pages(pages, 6), llm, "m")
    assert llm.calls == 2
    assert outcome.failed_chunks and "pages 1-6" in outcome.failed_chunks[0]
    assert outcome.failed_pages == 6


async def test_fake_llm_extracts_from_generated_pdf(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    outcome = await extract_requirements(chunk_pages(pages, 3), FakeLLM(), "fake")
    texts = " ".join(r.quote for r in outcome.requirements)
    assert "grade 7" in texts and "ISO 4427" in texts and "30 November 2026" in texts


def _reqs() -> list[Requirement]:
    return [
        Requirement(id="R1", requirement="Grade 7 registration", quote="q", page=2, verified=True, category="eligibility"),
        Requirement(id="R2", requirement="Bid guarantee 2%", quote="q", page=4, verified=False, category="commercial"),
        Requirement(id="R3", requirement="Sign form MBD 4", quote="q", page=9, verified=True, category="legal"),
        Requirement(id="R4", requirement="Optional site visit", quote="q", page=9, mandatory=False),
    ]


def test_assessments_keep_batch_ids_and_the_verdict_follows_them() -> None:
    reqs = _reqs()
    raw = {
        "assessments": [
            {"id": "R1", "status": "not stated", "hard": True, "why": "grade not in profile"},
            {"id": "R2", "status": "Fails", "why": "no bank facility"},
            {"id": "R3", "status": "not stated", "hard": False, "why": "form"},
            {"id": "R99", "status": "fails", "why": "invented"},
            {"id": "[R1]", "status": "met", "why": "dup"},
            {"id": "R3", "status": "maybe", "why": "dup with a bad status"},
        ]
    }
    got, rejected = parse_assessments(raw, reqs[:3])
    assert [(a.id, a.status, a.hard) for a in got] == [("R1", "not stated", True), ("R2", "fails", False), ("R3", "not stated", False)]
    assert rejected == ["R99"]
    assert verdict_from(None, got) == "no bid"
    assert [(x.id, x.status, x.page, x.verified) for x in deciding_from(got, reqs)] == [
        ("R2", "fails", 4, False),
        ("R1", "not stated", 2, True),  # hard to obtain comes before the easy one
        ("R3", "not stated", 9, True),
    ]
    assert verdict_from(None, got[:1]) == "bid with conditions"
    assert verdict_from(None, []) == "bid"
    assert verdict_from(Scope(fit="outside"), []) == "no bid"


def test_scope_keeps_names_only_when_they_are_on_the_pages() -> None:
    pages = [Page(number=1, text="ABAQULUSI LOCAL MUNICIPALITY. Tender 8/2/1-656: Supply of water pumps.")]
    scope = parse_scope({"title": "Supply of water pumps", "buyer": "City of Gotham", "buys": "Pumps.", "fit": "Partial", "why": "w"}, pages)
    assert scope.title == "Supply of water pumps" and scope.buyer is None and scope.fit == "partly"


async def test_decide_runs_scope_batches_and_summary_with_effort(tender_pdf: bytes) -> None:
    fake = FakeLLM()
    pages = extract_pages(tender_pdf)
    decision, unassessed = await decide("Civil works contractor, grade 6.", _reqs(), fake, "decider", "low", pages=pages)
    assert unassessed == 0
    assert [c["schema_name"] for c in fake.calls] == ["scope", "assessments", "summary"]
    assert all(c["effort"] == "low" for c in fake.calls)
    prompt = fake.calls[1]["user"]
    assert "[R1] page 2 | verified | eligibility" in prompt and "[R2] page 4 | UNVERIFIED | commercial" in prompt
    assert "R4" not in prompt  # optional requirements are not assessed
    assert decision.verdict == "bid with conditions"
    assert [x.id for x in decision.deciding] == ["R1"] and decision.unmet_total == 1
    assert decision.counts == {"fails": 0, "not stated": 1, "met": 0, "routine": 2}


class SkippingLLM:
    """Assesses only the first id it is asked about, so every batch needs the second attempt."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["schema_name"] != "assessments":
            raise LLMError("no scope or summary here")
        ids = re.findall(r"^\[(R\d+)\]", kwargs["user"], re.M)
        self.asked.append(ids)
        return {"assessments": [{"id": ids[0], "status": "met", "why": ""}]}


async def test_skipped_ids_are_asked_again_once_then_reported() -> None:
    llm = SkippingLLM()
    decision, unassessed = await decide("p", _reqs(), llm, "m")
    assert llm.asked == [["R1", "R2", "R3"], ["R2", "R3"]]
    assert unassessed == 1 and decision.unassessed == ["R3"]
    assert decision.verdict == "bid" and decision.scope is None and decision.reasons == []


async def test_failed_chunk_is_reported_as_not_read() -> None:
    pages = [Page(number=i, text=f"page {i} text: bidders must submit form {i}") for i in range(1, 5)]
    seen: list[tuple[str, int, bool]] = []

    async def on_chunk(chunk: list[Page], n: int, ok: bool) -> None:
        seen.append((f"{chunk[0].number}-{chunk[-1].number}", n, ok))

    outcome = await extract_requirements(chunk_pages(pages, 2), ScriptedLLM(LLMError("model reply is empty")), "m", 2, on_chunk)
    assert sorted(seen) == [("1-2", 0, False), ("3-4", 0, False)]
    assert outcome.failed_pages == 4 and len(outcome.failed_chunks) == 2
