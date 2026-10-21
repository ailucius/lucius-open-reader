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


def make_scanned_pdf(kinds: list[tuple[str, str]], dpi: int = 100) -> bytes:
    """A PDF built page by page as a scanner and a portal would leave it. Each item is (kind, text):
    "text" a normal text layer; "scan" the text drawn, rendered to an image, and only the image kept;
    "stamp" a scan with a short text layer on top (a portal's page number); "logo" a small image and
    no text; "blank" nothing."""
    out = pymupdf.open()
    for kind, text in kinds:
        page = out.new_page(width=595, height=842)
        if kind == "text":
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=11)
        elif kind in ("scan", "stamp", "logo"):
            src = pymupdf.open()
            drawn = src.new_page(width=595, height=842)
            drawn.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=11)
            image = drawn.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).tobytes("png")
            src.close()
            where = pymupdf.Rect(40, 40, 140, 90) if kind == "logo" else page.rect
            page.insert_image(where, stream=image)
            if kind == "stamp":
                page.insert_text((500, 830), "Page 4 of 9", fontsize=8)
    data = out.tobytes()
    out.close()
    return data


@pytest.fixture
def tender_pdf() -> bytes:
    return make_pdf(TENDER_PAGES)


@pytest.fixture
def settings() -> Settings:
    return Settings(fake_llm=True, concurrency=2, chunk_pages=3)
