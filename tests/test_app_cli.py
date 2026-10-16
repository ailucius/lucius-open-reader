"""End-to-end through the web API (SSE) and the CLI, with the fake LLM and a generated PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reader.__main__ import main
from reader.app import create_app
from reader.config import Settings


def _events(text: str) -> list[dict[str, object]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_web_run_with_upload_streams_progress_and_result(tender_pdf: bytes) -> None:
    app = create_app(Settings(fake_llm=True, chunk_pages=3, concurrency=2))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Lucius Open Reader" in page.text
        assert client.get("/api/health").json()["fake_llm"] is True

        started = client.post(
            "/api/runs",
            data={"profile": "Pipeline contractor, grade 6 registration, ISO 9001."},
            files={"file": ("tender.pdf", tender_pdf, "application/pdf")},
        )
        assert started.status_code == 200, started.text
        run_id = started.json()["run_id"]

        with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
            body = "".join(resp.iter_text())
        events = _events(body)
        kinds = [e["type"] for e in events]
        assert kinds[0] == "stage" and "pages" in kinds and "progress" in kinds and kinds[-1] == "result"
        progress = [e for e in events if e["type"] == "progress"]
        assert progress[-1]["pages_read"] == 7

        result = events[-1]["result"]
        assert isinstance(result, dict)
        reqs = result["requirements"]
        assert len(reqs) >= 8
        assert all(r["verified"] for r in reqs)
        assert result["decision"]["verdict"] in ("bid", "bid with conditions", "no bid")

        pdf = client.get(f"/api/runs/{run_id}/pdf")
        assert pdf.headers["content-type"] == "application/pdf" and pdf.content == tender_pdf


def test_web_run_rejects_bad_input() -> None:
    app = create_app(Settings(fake_llm=True))
    with TestClient(app) as client:
        assert client.post("/api/runs", data={"url": ""}).status_code == 400
        assert client.post("/api/runs", data={"url": "ftp://x/y.pdf"}).status_code == 400
        assert client.post("/api/find", json={"query": "x"}).status_code == 503  # no Tavily key


def test_cli_prints_json(tmp_path: Path, tender_pdf: bytes, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("READER_CHUNK_PAGES", "6")
    path = tmp_path / "tender.pdf"
    path.write_bytes(tender_pdf)
    assert main([str(path), "--profile", "Water pipeline contractor", "--fake"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stats"]["pages_total"] == 7
    assert out["stats"]["chunks_total"] == 2
    assert out["decision"]["verdict"] == "bid with conditions"


def test_web_runs_are_limited_per_address_and_per_day(tender_pdf: bytes) -> None:
    files = {"file": ("tender.pdf", tender_pdf, "application/pdf")}
    app = create_app(Settings(fake_llm=True, runs_per_hour_per_ip=2, runs_per_day=3))
    with TestClient(app) as client:
        a = {"x-forwarded-for": "198.51.100.7"}
        b = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
        assert client.post("/api/runs", files=files, headers=a).status_code == 200
        assert client.post("/api/runs", files=files, headers=a).status_code == 200
        third = client.post("/api/runs", files=files, headers=a)
        assert third.status_code == 429 and "hour" in third.json()["detail"]
        assert client.post("/api/runs", files=files, headers=b).status_code == 200
        day = client.post("/api/runs", files=files, headers={"x-forwarded-for": "192.0.2.1"})
        assert day.status_code == 429 and "today" in day.json()["detail"]


def test_example_route_serves_the_saved_run_when_present() -> None:
    from reader.app import EXAMPLE

    app = create_app(Settings(fake_llm=True))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["example_ready"] is EXAMPLE.is_file()
        resp = client.get("/api/example")
        if EXAMPLE.is_file():
            body = resp.json()
            assert body["requirements"] and body["decision"]["verdict"] and body["pdf_url"].startswith("https://")
        else:
            assert resp.status_code == 404


def test_documents_route_needs_a_search_key_and_a_web_address() -> None:
    app = create_app(Settings(fake_llm=True))
    with TestClient(app) as client:
        assert client.post("/api/documents", json={"url": "ftp://x/y"}).status_code == 400
        assert client.post("/api/documents", json={"url": "https://buyer.example/tenders/"}).status_code == 503


def test_favicon_and_robots_are_served() -> None:
    app = create_app(Settings(fake_llm=True))
    with TestClient(app) as client:
        for path in ("/favicon.svg", "/favicon.ico"):
            r = client.get(path)
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("image/svg+xml")
            assert r.text.lstrip().startswith("<svg")
        r = client.get("/robots.txt")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "Disallow: /api/" in r.text
        assert 'rel="icon"' in client.get("/").text
