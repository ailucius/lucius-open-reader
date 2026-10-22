"""Record the Nebius replies for one real run on a small PDF with scanned pages (needs NEBIUS_API_KEY).

    NEBIUS_API_KEY=... .venv/bin/python -m tests.fixtures.record_scan_run

Writes tests/fixtures/scan_run.json: for every model call, the replay key, the model, the request's
text parts and the reply exactly as Token Factory sent it (the page images are not stored; their size
and SHA-256 are). tests/test_recorded.py replays it offline. Re-record when a prompt changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from reader.config import Settings
from reader.llm import NebiusClient
from reader.pipeline import run_reader
from tests.conftest import TENDER_PAGES, make_scanned_pdf

OUT = Path(__file__).parent / "scan_run.json"
PROFILE = "Pipeline contractor registered at grade 6. We lay HDPE pipes; no bank guarantee facility yet."
# page 1 text, page 2 a scan, page 3 a scan with a portal's page stamp on top, page 4 text
KINDS = [("text", TENDER_PAGES[0]), ("scan", TENDER_PAGES[1]), ("stamp", TENDER_PAGES[3]), ("text", TENDER_PAGES[4])]


def recorded_pdf() -> bytes:
    return make_scanned_pdf(KINDS, dpi=150)


def replay_key(body: dict[str, Any]) -> str:
    """Which call this is, independent of the order concurrent calls go out in."""
    user = body["messages"][-1]["content"]
    if isinstance(user, list):  # a page image
        return "image | " + next(part["text"] for part in user if part["type"] == "text")
    name = ((body.get("response_format") or {}).get("json_schema") or {}).get("name", "json")
    if name == "requirements":
        return f"{name} | {user.splitlines()[0]}"
    if name == "assessments":
        return f"{name} | " + next(line for line in user.splitlines() if line.startswith("Requirements to assess"))
    return name


class Recorder(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        content = await response.aread()
        body = json.loads(request.content)
        text_parts, images = [], []
        for message in body["messages"]:
            parts = message["content"] if isinstance(message["content"], list) else [{"type": "text", "text": message["content"]}]
            for part in parts:
                if part["type"] == "text":
                    text_parts.append({"role": message["role"], "text": part["text"]})
                else:
                    raw = part["image_url"]["url"].split(",", 1)[1].encode()
                    images.append({"base64_chars": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.calls.append(
            {
                "key": replay_key(body),
                "model": body["model"],
                "request": {k: v for k, v in body.items() if k != "messages"} | {"text": text_parts, "images": images},
                "status": response.status_code,
                "reply": json.loads(content),
            }
        )
        # the body is already decoded: drop the headers that describe the encoded one
        headers = [(k, v) for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=content)


async def main() -> int:
    settings = Settings.from_env()
    if not settings.nebius_api_key:
        print("NEBIUS_API_KEY is not set", file=sys.stderr)
        return 1
    recorder = Recorder()
    client = NebiusClient(settings.nebius_api_key, settings.nebius_base_url, settings.llm_timeout_s, transport=recorder)
    result, _ = await run_reader(settings=settings, llm=client, profile=PROFILE, pdf=recorded_pdf(), source_name="W-2026-14.pdf")
    # a call asked again (an empty answer, or ids the decider skipped) shares its key: number them in
    # the order they went out, which is fixed for calls with the same key
    seen: dict[str, int] = {}
    for call in recorder.calls:
        seen[call["key"]] = seen.get(call["key"], 0) + 1
        call["attempt"] = seen[call["key"]]
    OUT.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "base_url": settings.nebius_base_url,
                "profile": PROFILE,
                "calls": sorted(recorder.calls, key=lambda c: (c["key"], c["attempt"])),
                "result": json.loads(result.model_dump_json()),
                "usage": client.usage,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"{len(recorder.calls)} calls recorded to {OUT}")
    print(json.dumps(result.stats.model_dump(), indent=1))
    print("verdict:", result.decision.verdict if result.decision else None, "| usage:", client.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
