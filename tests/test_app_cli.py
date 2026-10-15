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
