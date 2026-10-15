from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reader.decide import decide, parse_decision
from reader.extract import extract_requirements, parse_items
from reader.llm import FakeLLM, LLMError
from reader.models import Page, Requirement
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
        Requirement(id="R1", requirement="Grade 7 registration", quote="q", page=2, verified=True),
        Requirement(id="R2", requirement="Bid guarantee 2%", quote="q", page=4, verified=False),
    ]


def test_parse_decision_rejects_ids_not_in_list() -> None:
    raw = {
        "verdict": "No-Bid",
        "confidence": 80,
        "deciding": [{"id": "R1", "why": "not registered"}, {"id": "R99", "why": "invented"}, {"id": "[R1]", "why": "dup"}],
        "reasons": ["Not registered at grade 7."],
        "conditions": [],
    }
    d = parse_decision(raw, _reqs(), "m")
    assert d.verdict == "no bid"
    assert d.confidence == pytest.approx(0.8)
    assert [x.id for x in d.deciding] == ["R1"]
    assert d.deciding[0].page == 2
    assert d.rejected_ids == ["R99"]


def test_parse_decision_unknown_verdict_is_an_error() -> None:
    with pytest.raises(LLMError):
        parse_decision({"verdict": "maybe", "confidence": 0.5}, _reqs(), "m")


async def test_decide_prompt_carries_ids_pages_and_flags() -> None:
    fake = FakeLLM()
    reqs = _reqs()
    decision, left_out = await decide("Civil works contractor, grade 6.", reqs, fake, "decider")
    assert left_out == 0
    prompt = fake.calls[0]["user"]
    assert "[R1] page 2 | mandatory | verified" in prompt
    assert "[R2] page 4 | mandatory | UNVERIFIED" in prompt
    assert decision.verdict == "bid with conditions"
    assert {x.id for x in decision.deciding} == {"R1", "R2"}
