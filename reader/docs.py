"""Find a tender's documents from a notice or listing page: a small agent over Tavily and Nemotron.

1. Tavily Extract (advanced depth) reads the page as markdown; its links are collected.
2. Nemotron picks which links are this tender's documents and, if none are there, which pages
   to open next (at most OPEN_NEXT per step, at most MAX_STEPS pages deep).
3. Every pick is probed: only links whose first bytes are a PDF are offered to read.

Tavily Map and Crawl returned nothing for municipal sites we tried (15 Oct 2026) while Extract read
them, and listing pages keep the documents one or two clicks deep, hence the step loop.
The picker runs Nemotron 3 Super with high thinking at temperature 0.6 (NVIDIA's setting for
reasoning): at temperature 0 Ultra looped on a 135-link page ("Notices (L121)" until the token
limit), and at 0.6 Ultra picked nothing where Super found the right pages.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

import httpx
from pydantic import BaseModel

from .config import TAVILY_EXTRACT_URL, Settings
from .fetch import FetchError, probe_document
from .find import FindError
from .llm import LLMClient, LLMError

MAX_LINKS = 220
MAX_STEPS = 2
OPEN_NEXT = 3
MAX_DOCS = 8
PAGE_CHARS = 2500

_LINK = re.compile(r"\[([^\]]{0,200})\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_SKIP = re.compile(r"^(?:mailto:|tel:|javascript:|#)|\.(?:png|jpe?g|gif|svg|webp|ico|css|js|mp4|mp3)(?:\?|$)", re.I)
_SOCIAL = ("facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com", "youtube.com", "wa.me", "whatsapp.com")
_DOCISH = re.compile(r"\.(?:pdf|docx?|zip|rar)(?:\?|$)|download|blob|attachment|document|getfile|viewfile", re.I)


class Link(BaseModel):
    id: str
    text: str
    url: str
    docish: bool  # the address looks like a file


class FoundDocument(BaseModel):
    title: str
    url: str
    why: str = ""
    found_on: str
    is_pdf: bool
    size: int | None = None


class DocumentSearch(BaseModel):
    start_url: str
    pages_read: list[str]
    pages_failed: list[str]
    documents: list[FoundDocument]
    note: str | None = None


PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["documents", "open_next"],
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "why"],
                "properties": {"id": {"type": "string"}, "why": {"type": "string"}},
            },
        },
        "open_next": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "why"],
                "properties": {"id": {"type": "string"}, "why": {"type": "string"}},
            },
        },
    },
}

PICK_SYSTEM = f"""You help a company find the documents of a public tender on a buyer's website.
You get what the company is looking for, an excerpt of one web page, and the page's links with ids
like [L12]. Each link shows its text and address; "file" marks addresses that look like files.

Return:
- documents: the links that are the tender documents a bidder needs: the tender document or bid
  pack, terms of reference, specifications, bills of quantities, forms, addenda. Only for open
  tenders that match what the company is looking for. Not cancellation letters, award notices,
  lists of received bids, annual plans or general policies. Best first, at most {MAX_DOCS}.
- open_next: if the documents are not on this page, the links most likely to lead to them (the
  tender's own notice page, a "tenders" or "bids" page, a downloads page). At most {OPEN_NEXT}.
  Empty when documents were found.
why: one short plain sentence.
Use only the ids listed. Never invent an id or an address. Return JSON only."""


def collect_links(markdown: str, base_url: str) -> list[Link]:
    seen: set[str] = set()
    out: list[Link] = []
    base_host = urlsplit(base_url).hostname or ""
    for text, raw in _LINK.findall(markdown):
        if _SKIP.search(raw):
            continue
        url = urldefrag(urljoin(base_url, raw)).url
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if any(parts.hostname == s or parts.hostname.endswith("." + s) for s in _SOCIAL):
            continue
        if url in seen or url.rstrip("/") == base_url.rstrip("/"):
            continue
        seen.add(url)
        label = " ".join(text.replace("!", " ").split())[:140]
        if not label:
            label = parts.path.rsplit("/", 1)[-1][:80] or parts.hostname
        out.append(Link(id="", text=label, url=url, docish=bool(_DOCISH.search(url))))
    # files first, then links on the same site, so the cap keeps what matters
    out.sort(key=lambda l: (not l.docish, (urlsplit(l.url).hostname or "") != base_host))
    out = out[:MAX_LINKS]
    for i, link in enumerate(out, start=1):
        link.id = f"L{i}"
    return out


def build_pick_prompt(query: str, page_url: str, markdown: str, links: list[Link]) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown)  # the links are listed below
    text = " ".join(text.split())[:PAGE_CHARS]
    lines = [f"Looking for: {query.strip() or 'the documents of the tender on this page'}", "", f"Page: {page_url}",
             f"Excerpt: {text}", "", f"Links ({len(links)}):"]
    for link in links:
        lines.append(f"[{link.id}] {'file | ' if link.docish else ''}{link.text} | {link.url}")
    return "\n".join(lines)


async def tavily_extract(urls: list[str], api_key: str, transport: httpx.AsyncBaseTransport | None = None) -> tuple[dict[str, str], list[str]]:
    """Returns ({url: markdown}, failed urls)."""
    async with httpx.AsyncClient(timeout=90.0, transport=transport) as client:
        resp = await client.post(
            TAVILY_EXTRACT_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"urls": urls, "extract_depth": "advanced", "format": "markdown"},
        )
    if resp.status_code >= 400:
        raise FindError(f"Tavily extract {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    pages = {str(r.get("url")): str(r.get("raw_content") or "") for r in data.get("results") or []}
    failed = [str(f.get("url")) for f in data.get("failed_results") or []]
    return pages, failed


async def find_documents(
    url: str,
    query: str,
    settings: Settings,
    llm: LLMClient,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DocumentSearch:
    if not settings.tavily_api_key:
        raise FindError("TAVILY_API_KEY is not set")
    result = DocumentSearch(start_url=url, pages_read=[], pages_failed=[], documents=[])

    # the address may already be a document ("/download/3001" was a PDF)
    try:
        is_pdf, final, size = await probe_document(url, settings, transport)
    except FetchError:
        is_pdf, final, size = False, url, None
    if is_pdf:
        result.documents.append(FoundDocument(title="This link is the document", url=final, found_on=url, is_pdf=True, size=size))
        return result

    picked: list[tuple[Link, str, str]] = []  # (link, why, page it was on)
    frontier = [url]
    visited: set[str] = set()
    for _step in range(MAX_STEPS):
        frontier = [u for u in frontier if u not in visited][:OPEN_NEXT]
        if not frontier:
            break
        visited.update(frontier)
        pages, failed = await tavily_extract(frontier, settings.tavily_api_key, transport)
        result.pages_failed += failed
        next_pages: list[str] = []
        for page_url, markdown in pages.items():
            result.pages_read.append(page_url)
            links = collect_links(markdown, page_url)
            if not links:
                continue
            try:
                raw = await llm.chat_json(
                    model=settings.finder_model,
                    system=PICK_SYSTEM,
                    user=build_pick_prompt(query, page_url, markdown, links),
                    schema_name="links",
                    schema=PICK_SCHEMA,
                    effort=settings.finder_effort,
                    temperature=settings.finder_temperature,
                )
            except LLMError:
                result.pages_failed.append(page_url)
                continue
            by_id = {l.id: l for l in links}
            for item in raw.get("documents") or []:
                link = by_id.get(str(item.get("id") or "").strip("[] "))
                if link and all(link.url != p[0].url for p in picked):
                    picked.append((link, str(item.get("why") or "").strip(), page_url))
            for item in raw.get("open_next") or []:
                link = by_id.get(str(item.get("id") or "").strip("[] "))
                if link and link.url not in visited:
                    next_pages.append(link.url)
        if picked:
            break
        frontier = next_pages

    async def check(link: Link, why: str, page: str) -> FoundDocument | None:
        try:
            ok, final, size = await probe_document(link.url, settings, transport)
        except FetchError:
            return None
        return FoundDocument(title=link.text, url=final, why=why, found_on=page, is_pdf=ok, size=size)

    checked = await asyncio.gather(*(check(*p) for p in picked[:MAX_DOCS]))
    docs = [d for d in checked if d is not None]
    docs.sort(key=lambda d: not d.is_pdf)  # PDFs first; other files (Word, ZIP) are listed but cannot be read here
    result.documents = docs
    if not docs:
        result.note = (
            "No tender documents were found within two pages of this one."
            if result.pages_read
            else "This page could not be read. Open it in your browser and paste the document's link."
        )
    return result
