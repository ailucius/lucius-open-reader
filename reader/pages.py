"""PDF to per-page text (pymupdf), and page chunking."""

from __future__ import annotations

import pymupdf

from .models import Page


class PdfError(ValueError):
    pass


def extract_pages(pdf: bytes, max_pages: int = 500) -> list[Page]:
    try:
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception as exc:  # pymupdf raises several internal types
        raise PdfError(f"cannot open PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise PdfError("PDF is password protected")
        if doc.page_count > max_pages:
            raise PdfError(f"PDF has {doc.page_count} pages; the limit is {max_pages}")
        return [Page(number=i + 1, text=page.get_text("text")) for i, page in enumerate(doc)]


def chunk_pages(pages: list[Page], size: int = 6) -> list[list[Page]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [pages[i : i + size] for i in range(0, len(pages), size)]
