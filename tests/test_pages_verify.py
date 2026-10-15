from __future__ import annotations

import pytest

from reader.models import Page, Requirement
from reader.pages import PdfError, chunk_pages, extract_pages
from reader.verify import normalise, verify_requirements
from tests.conftest import TENDER_PAGES, make_pdf


def test_extract_pages_reads_every_page(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    assert [p.number for p in pages] == list(range(1, len(TENDER_PAGES) + 1))
    assert "grade 7" in pages[1].text


def test_extract_pages_rejects_too_many_pages() -> None:
    with pytest.raises(PdfError):
        extract_pages(make_pdf(["a"] * 4), max_pages=3)


def test_extract_pages_rejects_garbage() -> None:
    with pytest.raises(PdfError):
        extract_pages(b"%PDF-1.7 not really a pdf")


def test_chunk_pages_groups_six() -> None:
    pages = [Page(number=i, text="x") for i in range(1, 14)]
    sizes = [len(c) for c in chunk_pages(pages, 6)]
    assert sizes == [6, 6, 1]


def test_normalise_ignores_case_punctuation_space_and_hyphen_breaks() -> None:
    assert normalise("Bidders  MUST be\nregis-\ntered, at grade 7!") == normalise("bidders must be registered at grade 7")


def _req(quote: str, page: int) -> Requirement:
    return Requirement(requirement="r", quote=quote, page=page)


def test_verify_exact_and_loose_quotes(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    reqs = [
        _req("Bidders must be registered with the national contractor board at grade 7 or higher.", 2),
        _req("BIDDERS MUST BE REGISTERED WITH THE NATIONAL CONTRACTOR BOARD, at grade 7", 2),
    ]
    out = verify_requirements(reqs, pages)
    assert all(r.verified for r in out)
    assert all(r.cited_page is None for r in out)


def test_verify_corrects_a_wrong_page(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    out = verify_requirements([_req("All pipes shall comply with ISO 4427 for polyethylene pipes.", 2)], pages)
    assert out[0].verified is True
    assert out[0].page == 3
    assert out[0].cited_page == 2


def test_verify_keeps_invented_quotes_flagged(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    reqs = [_req("Bidders must hold a valid space launch licence issued by the moon authority.", 2)]
    out = verify_requirements(reqs, pages)
    assert len(out) == 1
    assert out[0].verified is False
    assert out[0].page == 2
    assert out[0].match_score < 0.8


def test_verify_short_quote_needs_exact_phrase(tender_pdf: bytes) -> None:
    pages = extract_pages(tender_pdf)
    out = verify_requirements([_req("Late bids", 5), _req("Early bids", 5)], pages)
    assert [r.verified for r in out] == [True, False]


def test_verify_quote_across_page_break() -> None:
    pages = [
        Page(number=1, text="The supplier shall deliver the pumps to the site"),
        Page(number=2, text="within ninety days of the contract award date."),
    ]
    out = verify_requirements([_req("deliver the pumps to the site within ninety days of the contract", 2)], pages)
    assert out[0].verified is True
    assert out[0].page == 1
