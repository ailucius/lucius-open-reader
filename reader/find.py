"""Find public tender notices and PDF links with the Tavily Search API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from .config import TAVILY_SEARCH_URL
from .models import SearchHit


class FindError(RuntimeError):
    pass


_BUYER = re.compile(
    r"\b("
    r"(?:Department|Ministry|Municipality|City|County|Council|Authority|Agency|Office|Board)"
    r" of(?: the)? [A-Z][\w&'.-]*(?:(?: and| of| the| &)? [A-Z][\w&'.-]*){0,6}"
    r"|(?:[A-Z][\w&'.-]* ){1,5}(?:Metropolitan Municipality|District Municipality|Local Municipality"
    r"|Municipality|City Council|County Council|Council|Authority|Agency|Water Board|Utility)"
    r")"
)


def guess_buyer(*texts: str) -> str | None:
    """Best-effort buyer name from visible text; None when nothing looks like a buyer."""
    for text in texts:
        match = _BUYER.search(text or "")
        if match:
            return match.group(1).strip(" .,-")
    return None


def is_pdf_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")


def build_query(query: str) -> str:
    q = " ".join(query.split())
    if not re.search(r"\b(tender|bid|rfp|rfq|procurement)\b", q, re.I):
        q += " tender"
    return q


async def search_tenders(
    query: str,
    api_key: str | None,
    max_results: int = 10,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[SearchHit]:
    if not api_key:
        raise FindError("TAVILY_API_KEY is not set")
    if not query.strip():
        raise FindError("empty query")
    body: dict[str, Any] = {
        "query": build_query(query),
        "search_depth": "basic",
        "topic": "general",
        "max_results": max(1, min(max_results, 20)),
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
        resp = await client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
    if resp.status_code >= 400:
        raise FindError(f"Tavily {resp.status_code}: {resp.text[:200]}")
    hits: list[SearchHit] = []
    for item in resp.json().get("results") or []:
        url = str(item.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        title = str(item.get("title") or url)
        snippet = str(item.get("content") or "")[:400]
        hits.append(
            SearchHit(
                title=title,
                url=url,
                buyer=guess_buyer(title, snippet),
                snippet=snippet,
                is_pdf=is_pdf_url(url),
                score=item.get("score"),
            )
        )
    hits.sort(key=lambda h: (not h.is_pdf, -(h.score or 0.0)))
    return hits
