from __future__ import annotations

import pymupdf
import pytest

from reader.config import Settings

TENDER_PAGES: list[str] = [
    "Invitation to Tender No. W-2026-14\nCity of Example Water Department\nConstruction of a bulk water supply pipeline.",
    "Eligibility\nBidders must be registered with the national contractor board at grade 7 or higher.\n"
    "The bidder shall have completed at least three pipeline projects of similar size in the past five years.",
    "Technical\nAll pipes shall comply with ISO 4427 for polyethylene pipes.\nPump stations should preferably use variable speed drives.",
    "Commercial\nA bid guarantee of 2% of the bid price must be submitted with the bid.\n"
    "Prices shall be quoted in local currency including VAT.",
    "Submission\nBids must be submitted no later than 30 November 2026 at 12:00 local time.\n"
    "Late bids will be rejected without opening.",
    "Legal\nThe contract shall be governed by the laws of the Republic.\n"
    "Bidders must sign the declaration of interest form attached as Annex C.",
    "General conditions\nThis page lists general information for bidders.",
]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def tender_pdf() -> bytes:
    return make_pdf(TENDER_PAGES)


@pytest.fixture
def settings() -> Settings:
    return Settings(fake_llm=True, concurrency=2, chunk_pages=3)
