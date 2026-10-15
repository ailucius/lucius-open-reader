"""Network-facing modules, tested offline with httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from reader.config import Settings
from reader.fetch import FetchError, fetch_pdf
from reader.find import FindError, guess_buyer, search_tenders
from reader.llm import LLMError, NebiusClient, parse_json_content

PUBLIC = "http://8.8.8.8"  # an IP literal, so no DNS lookup is needed in tests


async def test_fetch_follows_redirect_and_returns_pdf(tender_pdf: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/t.pdf"})
        return httpx.Response(200, content=tender_pdf)

    data = await fetch_pdf(f"{PUBLIC}/start", Settings(), transport=httpx.MockTransport(handler))
    assert data == tender_pdf


async def test_fetch_enforces_size_cap() -> None:
    big = b"%PDF-1.7\n" + b"0" * (1024 * 1024 + 10)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=big))
    with pytest.raises(FetchError, match="larger than 1 MB"):
        await fetch_pdf(f"{PUBLIC}/big.pdf", Settings(max_pdf_mb=1), transport=transport)


async def test_fetch_rejects_non_pdf() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>hello</html>"))
    with pytest.raises(FetchError, match="not a PDF"):
        await fetch_pdf(f"{PUBLIC}/x.pdf", Settings(), transport=transport)


@pytest.mark.parametrize(
    "url",
    ["ftp://8.8.8.8/x.pdf", "file:///etc/passwd", "http://127.0.0.1/x.pdf", "http://10.0.0.5/x.pdf", "http://169.254.169.254/latest"],
)
async def test_fetch_blocks_bad_schemes_and_private_hosts(url: str) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"%PDF-1.7"))
    with pytest.raises(FetchError):
        await fetch_pdf(url, Settings(), transport=transport)


async def test_fetch_blocks_redirect_to_private_host() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/x.pdf"}))
    with pytest.raises(FetchError, match="non-public"):
        await fetch_pdf(f"{PUBLIC}/start", Settings(), transport=transport)


async def test_tavily_search_shapes_hits() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Tender news", "url": "https://news.example/a", "content": "story", "score": 0.9},
                    {
                        "title": "Bid W-14: water pipeline",
                        "url": "https://www.city.example/docs/W-14.PDF",
                        "content": "Issued by the Department of Water and Sanitation for bulk supply.",
                        "score": 0.5,
                    },
                    {"title": "bad", "url": "javascript:alert(1)", "content": ""},
                ]
            },
        )

    hits = await search_tenders("water supply works South Africa", "tvly-test", transport=httpx.MockTransport(handler))
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["query"] == "water supply works South Africa tender"  # type: ignore[index]
    assert [h.is_pdf for h in hits] == [True, False]
    assert hits[0].buyer == "Department of Water and Sanitation"


async def test_tavily_needs_a_key() -> None:
    with pytest.raises(FindError):
        await search_tenders("x", None)


def test_guess_buyer_municipality() -> None:
    assert guess_buyer("Tender from Ekurhuleni Metropolitan Municipality for pipes") == "Ekurhuleni Metropolitan Municipality"
    assert guess_buyer("T-22/2022 THREE YEARS Knysna Municipality repair of pumps") == "Knysna Municipality"
    assert guess_buyer("Sewer Pump Stations In. Local Municipality") is None  # only generic words
    assert guess_buyer("nothing here") is None


def test_official_and_aggregator_domains() -> None:
    from reader.find import is_aggregator, is_official

    assert is_official("https://www.etenders.gov.za/home/Download?x=1")
    assert is_official("https://cms.tacoma.gov/purchasing/a.pdf")
    assert is_official("https://ted.europa.eu/en/notice/1")
    assert is_official("https://www.find-tender.service.gov.uk/Notice/1")
    assert not is_official("https://www.knysna.co.za/a.pdf")
    assert is_aggregator("https://www.tendersontime.com/south-africa-tenders/")
    assert not is_aggregator("https://www.knysna.gov.za/tenders/")


async def test_search_excludes_aggregators_and_puts_official_pdfs_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tendersontime.com" in body["exclude_domains"] and body["search_depth"] == "advanced"
        return httpx.Response(200, json={"results": [
            {"title": "aggregator", "url": "https://www.globaltenders.com/za/pumps", "content": "", "score": 0.99},
            {"title": "page", "url": "https://www.capetown.gov.za/tenders", "content": "", "score": 0.95},
            {"title": "other pdf", "url": "https://consultant.example/t.pdf", "content": "", "score": 0.9},
            {"title": "official pdf", "url": "https://www.knysna.gov.za/t.pdf", "content": "", "score": 0.4},
        ]})

    hits = await search_tenders("pumps", "tvly-test", transport=httpx.MockTransport(handler))
    assert [h.title for h in hits] == ["official pdf", "other pdf", "page"]
    assert [h.official for h in hits] == [True, False, True]


def test_parse_json_content_handles_think_and_fences() -> None:
    assert parse_json_content('<think>hmm {"x": 0}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_content('Sure: {"a": [1, 2]} done') == {"a": [1, 2]}


async def test_nebius_client_request_shape_and_json_object_fallback() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tokenfactory.nebius.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer nb-test"
        body = json.loads(request.content)
        bodies.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": "json_schema not supported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"requirements": []}'}}]})

    client = NebiusClient("nb-test", "https://api.tokenfactory.nebius.com/v1/", transport=httpx.MockTransport(handler))
    out = await client.chat_json(model="nvidia/nemotron-3-super-120b-a12b", system="s", user="u", schema_name="requirements", schema={"type": "object"})
    assert out == {"requirements": []}
    assert [b["response_format"]["type"] for b in bodies] == ["json_schema", "json_object"]  # type: ignore[index]
    assert bodies[0]["model"] == "nvidia/nemotron-3-super-120b-a12b"


async def test_nebius_client_sends_effort_and_retries_without_thinking_on_empty_answer() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            # thinking used the whole budget: no answer at all
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"requirements": []}'}, "finish_reason": "stop"}]})

    client = NebiusClient("nb-test", "https://api.tokenfactory.nebius.com/v1/", transport=httpx.MockTransport(handler), max_tokens=4096)
    out = await client.chat_json(model="m", system="s", user="u", schema_name="requirements", schema={"type": "object"}, effort="low")
    assert out == {"requirements": []}
    assert bodies[0]["reasoning_effort"] == "low" and bodies[0]["max_tokens"] == 4096
    assert "reasoning_effort" not in bodies[1]
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.thinking_retries == 1


async def test_nebius_client_empty_answer_without_thinking_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]})

    client = NebiusClient("nb-test", "https://api.tokenfactory.nebius.com/v1/", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="empty"):
        await client.chat_json(model="m", system="s", user="u", schema_name="x", schema={"type": "object"}, effort="none")
