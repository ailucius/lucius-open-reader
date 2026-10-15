"""Data shapes shared by every stage of the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["eligibility", "technical", "commercial", "legal", "submission", "other"]
CATEGORIES: tuple[str, ...] = ("eligibility", "technical", "commercial", "legal", "submission", "other")
Verdict = Literal["bid", "bid with conditions", "no bid"]
VERDICTS: tuple[str, ...] = ("bid", "bid with conditions", "no bid")


class Page(BaseModel):
    number: int  # 1-based, as a human reads it
    text: str


class Requirement(BaseModel):
    id: str = ""
    requirement: str
    quote: str
    page: int
    category: Category = "other"
    mandatory: bool = True
    verified: bool = False
    match_score: float = 0.0
    cited_page: int | None = None  # set when verification moved the page
    source_chunk: str = ""


class DecidingRequirement(BaseModel):
    id: str
    requirement: str
    page: int
    verified: bool
    why: str


class Decision(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    deciding: list[DecidingRequirement] = []
    reasons: list[str] = []
    conditions: list[str] = []
    rejected_ids: list[str] = []  # ids the model cited that are not in the list
    model: str = ""


class ReadStats(BaseModel):
    pages_total: int = 0
    pages_read: int = 0
    chunks_total: int = 0
    chunks_failed: list[str] = []
    malformed_items: int = 0
    duplicates_merged: int = 0
    requirements: int = 0
    verified: int = 0
    unverified: int = 0
    pages_corrected: int = 0
    empty_pages: int = 0


class RunResult(BaseModel):
    source: str
    pdf_url: str | None = None
    profile: str = ""
    reader_model: str
    decider_model: str
    fake_llm: bool
    stats: ReadStats
    requirements: list[Requirement]
    decision: Decision | None = None
    decision_note: str | None = None


class SearchHit(BaseModel):
    title: str
    url: str
    buyer: str | None = None
    snippet: str = ""
    is_pdf: bool = False
    score: float | None = None
