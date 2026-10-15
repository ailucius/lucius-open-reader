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


_TOK = r"[A-Z][a-z][\w'&-]*"  # a capitalised word; all-caps words ("YEARS", "TENDER") are headings, not names
_BUYER = re.compile(
    r"\b("
    rf"(?:Department|Ministry|Municipality|City|County|Council|Authority|Agency|Office|Board) of(?: the)? {_TOK}"
    rf"(?:(?: and| of| the| &)? {_TOK}){{0,6}}"
    rf"|(?:{_TOK} ){{1,5}}(?:Metropolitan Municipality|District Municipality|Local Municipality"
    r"|Municipality|City Council|County Council|Council|Authority|Agency|Water Board|Utility)"
    r")\b"
)
_GENERIC = {
    "local", "district", "metropolitan", "city", "county", "municipality", "council", "authority", "agency",
    "water", "board", "utility", "sewer", "sewage", "pump", "pumps", "station", "stations", "tender", "tenders",
    "bid", "bids", "supply", "the", "of", "and", "public", "works", "latest", "new",
}

# Paywalled tender aggregators: they copy notices from the buyers' own sites and gate the documents.
AGGREGATORS: tuple[str, ...] = (
    "tendersontime.com", "globaltenders.com", "tendersinfo.com", "bidhits.co.za", "tenderalerts.co.za",
    "tenderbulletins.co.za", "biddingsource.com", "tenderimpulse.com", "tendersglobal.net", "tender247.com",
    "easytenders.co.za", "onlinetenders.co.za", "etenderlive.co.za", "bidprime.com", "instantmarkets.com",
    "govtribe.com", "tenderdetail.com", "dgmarket.com", "tendertiger.com", "aitenders.co.za", "protenders.co.za",
    "tenders-sa.org", "tendernotices.co.za", "rfqguy.co.za", "irelandtenders.com", "tenderscout.ie",
    "tenderskenya.co.ke", "developmentaid.org", "bidsinfo.com", "tendersniper.com",
    # not tender sources at all
    "scribd.com", "facebook.com", "linkedin.com", "youtube.com",
)
_OFFICIAL_SUFFIXES = (
    ".gov", ".mil", ".int", ".europa.eu", "worldbank.org", "ungm.org", "iadb.org", "afdb.org", "adb.org",
    "undp.org", "unops.org", "unicef.org",
)
_OFFICIAL_PARTS = (".gov.", ".gob.", ".gouv.", ".govt.", ".gc.ca", ".gv.at", ".admin.ch")


def guess_buyer(*texts: str) -> str | None:
    """Best-effort buyer name from visible text; None when nothing looks like a named buyer."""
    for text in texts:
        for match in _BUYER.finditer(text or ""):
            name = match.group(1).strip(" .,-")
            if any(w.lower() not in _GENERIC for w in re.findall(r"[A-Za-z']+", name)):
                return name
    return None


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def is_official(url: str) -> bool:
    """A government or multilateral domain (the buyer's own site), as far as the address shows."""
    host = host_of(url)
    return host.endswith(_OFFICIAL_SUFFIXES) or any(part in host + "." for part in _OFFICIAL_PARTS)


def is_aggregator(url: str) -> bool:
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in AGGREGATORS)


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
        "search_depth": "advanced",
        "topic": "general",
        "max_results": max(1, min(max_results, 20)),
        "include_answer": False,
        "exclude_domains": list(AGGREGATORS),
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
        if not url.startswith(("http://", "https://")) or is_aggregator(url):
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
                official=is_official(url),
                score=item.get("score"),
            )
        )
    # the buyer's own documents first: official PDFs, other PDFs, official pages, the rest
    hits.sort(key=lambda h: (not h.is_pdf, not h.official, -(h.score or 0.0)))
    return hits
