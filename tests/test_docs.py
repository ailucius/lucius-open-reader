"""The document finder, offline: Tavily Extract and the buyer's site are mocked, the model is the fake."""

from __future__ import annotations

import json

import httpx

from reader.config import Settings
from reader.docs import collect_links, find_documents
from reader.llm import FakeLLM

PAGE = """# Tenders
* [T 21/2026/27 Tender Document](/wp-content/uploads/T-21-Tender-Document.pdf) pdf | 842 KB
* [T 21/2026/27 Advert](https://buyer.example/wp-content/uploads/T-21-Advert.pdf)
* [Not a pdf](https://buyer.example/files/fake.pdf)
* [Logo](https://buyer.example/logo.png) [Mail us](mailto:scm@buyer.example) [Share](https://www.facebook.com/buyer)
* [Request for quotes](https://buyer.example/quotes/) [Tenders again](/tenders/#top)
"""


def test_collect_links_resolves_filters_and_puts_files_first() -> None:
    links = collect_links(PAGE, "https://buyer.example/tenders/")
    urls = [l.url for l in links]
    assert urls[:3] == [
        "https://buyer.example/wp-content/uploads/T-21-Tender-Document.pdf",
        "https://buyer.example/wp-content/uploads/T-21-Advert.pdf",
        "https://buyer.example/files/fake.pdf",
    ]
    assert "https://buyer.example/quotes/" in urls
    assert not any(u.endswith(".png") or "facebook" in u or u.startswith("mailto") for u in urls)
    assert "https://buyer.example/tenders/" not in urls  # the page itself, even with an anchor
    assert [l.id for l in links] == [f"L{i}" for i in range(1, len(links) + 1)]
    assert links[0].docish and not links[-1].docish


async def test_find_documents_reads_the_page_and_offers_only_real_pdfs() -> None:
    extract_calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://api.tavily.com/extract":
            body = json.loads(request.content)
            extract_calls.append(body)
            return httpx.Response(200, json={"results": [{"url": body["urls"][0], "raw_content": PAGE}], "failed_results": []})
        if url.endswith(".pdf") and "fake" not in url:
            return httpx.Response(206, content=b"%PDF-1.7\n" + b"x" * 100, headers={"content-range": "bytes 0-2047/842000"})
        return httpx.Response(200, content=b"<html>not a pdf</html>", headers={"content-type": "text/html"})

    settings = Settings(tavily_api_key="tvly-test", allow_private_urls=True)
    llm = FakeLLM()
    found = await find_documents("https://buyer.example/tenders/", "pumps", settings, llm, httpx.MockTransport(handler))
    assert extract_calls[0]["extract_depth"] == "advanced"
    assert found.pages_read == ["https://buyer.example/tenders/"]
    pdfs = [d for d in found.documents if d.is_pdf]
    assert [d.title for d in pdfs] == ["T 21/2026/27 Tender Document", "T 21/2026/27 Advert"]
    assert pdfs[0].size == 842000 and pdfs[0].found_on == "https://buyer.example/tenders/"
    assert [d.is_pdf for d in found.documents] == [True, True, False]  # the fake .pdf is listed last, not readable
    assert llm.calls[0]["schema_name"] == "links" and llm.calls[0]["effort"] == "high"


async def test_a_link_that_is_itself_a_pdf_needs_no_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tavily" not in str(request.url)
        return httpx.Response(200, content=b"%PDF-1.4 tender", headers={"content-length": "1464465"})

    found = await find_documents(
        "https://buyer.example/download/3001", "", Settings(tavily_api_key="tvly-test", allow_private_urls=True), FakeLLM(),
        httpx.MockTransport(handler),
    )
    assert [(d.url, d.is_pdf, d.size) for d in found.documents] == [("https://buyer.example/download/3001", True, 1464465)]
    assert found.pages_read == []
