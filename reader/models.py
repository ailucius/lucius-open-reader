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
    scan: bool = False  # no usable text layer, but an image covers the page: read it from the image
    transcribed: bool = False  # `text` is the vision model's transcription of the page image


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
    joined: bool = False  # the quote joins 2-3 verbatim passages of one page (the model skipped words)
    scanned: bool = False  # the quote was checked against the vision model's transcription of a scanned page
    source_chunk: str = ""


Status = Literal["fails", "not stated", "met", "routine"]


class Assessment(BaseModel):
    id: str
    status: Status
    hard: bool = False  # takes weeks, money or a third party to put right
    why: str = ""


class DecidingRequirement(BaseModel):
    id: str
    requirement: str
    page: int
    verified: bool
    status: Status = "not stated"
    hard: bool = False
    why: str


class Scope(BaseModel):
    title: str | None = None  # kept only when found on the first pages
    buyer: str | None = None  # kept only when found on the first pages
    buys: str = ""
    fit: Literal["fits", "partly", "outside"] = "partly"
    why: str = ""


class Decision(BaseModel):
    verdict: Verdict  # computed from the scope and the assessments, not chosen by the model
    confidence: float = Field(ge=0.0, le=1.0)
    scope: Scope | None = None
    deciding: list[DecidingRequirement] = []
    unmet_total: int = 0  # requirements failed or still to confirm (deciding shows the first few)
    reasons: list[str] = []
    conditions: list[str] = []
    assessments: list[Assessment] = []
    counts: dict[str, int] = {}
    unassessed: list[str] = []  # mandatory ids the model gave no judgment for
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
    joined_quotes: int = 0
    empty_pages: int = 0  # no text and nothing to read from an image (blank pages)
    empty_page_numbers: list[int] = []
    scanned_pages: list[int] = []  # scans read from the page image by the vision model
    scan_failed_pages: list[int] = []  # scans the vision model could not read (two tries), so not read
    scan_skipped_pages: list[int] = []  # scans over the per-run cap, so not read
    scan_timeout_pages: list[int] = []  # scans not finished within the time limit for scans, so not read
    scan_cap: int = 0
    thinking_retries: int = 0


class RunResult(BaseModel):
    source: str
    pdf_url: str | None = None
    profile: str = ""
    reader_model: str
    decider_model: str
    vision_model: str | None = None  # set when scanned pages were sent to it
    fake_llm: bool
    read_at: str | None = None  # UTC, ISO 8601
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
    official: bool = False  # a government or multilateral domain
    score: float | None = None
