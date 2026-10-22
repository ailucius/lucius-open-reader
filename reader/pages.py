"""PDF to per-page text (pymupdf), scan detection, page images for the vision model, and page chunking."""

from __future__ import annotations

import threading

import pymupdf

from .models import Page

# A page is a scan when its text layer has fewer than SCAN_MAX_CHARS characters (a scanner or a
# portal may stamp "Page 3 of 40" on top) and images cover at least SCAN_MIN_IMAGE_SHARE of it.
# A page with no text and no such image is blank: there is nothing to read on it.
SCAN_MAX_CHARS = 50
SCAN_MIN_IMAGE_SHARE = 0.3
# Page images for the vision model: grey, at most MAX_SIDE_PX on the long side (so an A1 drawing
# does not cost twenty A4 pages), PNG unless JPEG is much smaller (photographed or noisy scans).
MAX_SIDE_PX = 2000
PNG_LIMIT_BYTES = 900_000
JPEG_QUALITY = 80


# MuPDF is not thread-safe, even across separate documents: every use runs under this lock (pages
# are extracted and rendered in worker threads, and two runs can overlap on the server).
_MUPDF = threading.Lock()


class PdfError(ValueError):
    pass


def text_chars(text: str) -> int:
    return len("".join(text.split()))


def image_share(page: pymupdf.Page) -> float:
    """Share of the page covered by images (0 to 1); overlapping images can only round it up to 1."""
    area = page.rect.get_area()
    if area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        covered += (pymupdf.Rect(info["bbox"]) & page.rect).get_area()
    return min(1.0, covered / area)


def looks_scanned(page: pymupdf.Page, text: str) -> bool:
    return text_chars(text) < SCAN_MAX_CHARS and image_share(page) >= SCAN_MIN_IMAGE_SHARE


def _open(pdf: bytes) -> pymupdf.Document:
    try:
        return pymupdf.open(stream=pdf, filetype="pdf")
    except Exception as exc:  # pymupdf raises several internal types
        raise PdfError(f"cannot open PDF: {exc}") from exc


def extract_pages(pdf: bytes, max_pages: int = 500) -> list[Page]:
    with _MUPDF, _open(pdf) as doc:
        if doc.needs_pass:
            raise PdfError("PDF is password protected")
        if doc.page_count > max_pages:
            raise PdfError(f"PDF has {doc.page_count} pages; the limit is {max_pages}")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(Page(number=i + 1, text=text, scan=looks_scanned(page, text)))
        return pages


def render_page(pdf: bytes, number: int, dpi: int = 150) -> tuple[bytes, str]:
    """One page as an image for the vision model: (bytes, MIME type)."""
    with _MUPDF, _open(pdf) as doc:
        page = doc[number - 1]
        long_side_in = max(page.rect.width, page.rect.height) / 72
        dpi = max(36, min(dpi, int(MAX_SIDE_PX / long_side_in))) if long_side_in > 0 else dpi
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        png = pix.tobytes("png")
        if len(png) <= PNG_LIMIT_BYTES:
            return png, "image/png"
        jpeg = pix.tobytes("jpg", jpg_quality=JPEG_QUALITY)
        return (jpeg, "image/jpeg") if len(jpeg) < len(png) else (png, "image/png")


def chunk_pages(pages: list[Page], size: int = 6) -> list[list[Page]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [pages[i : i + size] for i in range(0, len(pages), size)]
