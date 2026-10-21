"""FastAPI web app: one HTML page, a run API with Server-Sent Events progress."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import __version__
from .config import Settings
from .docs import find_documents
from .export import checklist_csv, checklist_xlsx, export_filename
from .fetch import FetchError
from .find import FindError, search_tenders
from .llm import LLMClient, get_client
from .models import RunResult
from .pipeline import Event, run_reader

STATIC = Path(__file__).parent / "static"
ROBOTS = "User-agent: *\nAllow: /\nDisallow: /api/\n"
EXAMPLE = STATIC / "example.json"  # one saved run from the live models, served without calling them
MAX_RUNS = 20
RUN_TTL_S = 3600
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def checklist_response(result: RunResult, fmt: str) -> Response:
    """The bid checklist as a download: `xlsx` (Checklist and Summary sheets) or `csv` (the checklist)."""
    if fmt == "xlsx":
        body, media = checklist_xlsx(result), XLSX
    elif fmt == "csv":
        body, media = checklist_csv(result), "text/csv; charset=utf-8"
    else:
        raise HTTPException(404, "the checklist comes as .xlsx or .csv")
    return Response(
        body,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{export_filename(result, fmt)}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@dataclass
class Run:
    id: str
    created: float = field(default_factory=time.time)
    events: list[Event] = field(default_factory=list)
    done: bool = False
    result: dict[str, Any] | None = None
    pdf: bytes | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None

    async def emit(self, event: Event) -> None:
        self.events.append(event)
        self.changed.set()
        self.changed = asyncio.Event()


class FindRequest(BaseModel):
    query: str
    max_results: int = 10


class DocumentsRequest(BaseModel):
    url: str
    query: str = ""


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Lucius Open Reader", version=__version__)
    runs: dict[str, Run] = {}
    started: list[tuple[float, str]] = []  # (time, client address) of runs started, for the limits
    searched: list[tuple[float, str]] = []  # searches and document finds (they spend Tavily and model credit)

    def client_address(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def check_search_limits(address: str) -> None:
        """Searches are cheaper than runs: four times the run limits, when those are set."""
        now = time.time()
        searched[:] = [(t, a) for t, a in searched if now - t < 86400]
        if settings.runs_per_day and len(searched) >= 4 * settings.runs_per_day:
            raise HTTPException(429, "This demo has used its searches for today. Paste a PDF link instead.")
        if settings.runs_per_hour_per_ip:
            recent = sum(1 for t, a in searched if a == address and now - t < 3600)
            if recent >= 4 * settings.runs_per_hour_per_ip:
                raise HTTPException(429, "You have searched the most allowed in an hour on this demo. Try again later.")
        searched.append((now, address))

    def check_limits(address: str) -> None:
        now = time.time()
        started[:] = [(t, a) for t, a in started if now - t < 86400]
        if settings.runs_per_day and len(started) >= settings.runs_per_day:
            raise HTTPException(429, "This demo has read its tenders for today. Try again tomorrow, or open the finished example.")
        if settings.runs_per_hour_per_ip:
            recent = sum(1 for t, a in started if a == address and now - t < 3600)
            if recent >= settings.runs_per_hour_per_ip:
                raise HTTPException(429, "You have read the most tenders allowed in an hour on this demo. Try again later.")
        started.append((now, address))

    def client() -> LLMClient:
        try:
            return llm or get_client(settings)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    def prune() -> None:
        now = time.time()
        for rid in [r.id for r in runs.values() if now - r.created > RUN_TTL_S]:
            runs.pop(rid, None)
        while len(runs) >= MAX_RUNS:
            oldest = min(runs.values(), key=lambda r: r.created)
            runs.pop(oldest.id, None)

    # GET and HEAD: link checkers and preview bots ask with HEAD first.
    @app.api_route("/", methods=["GET", "HEAD"])
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.api_route("/og.png", methods=["GET", "HEAD"], include_in_schema=False)
    async def preview_image() -> FileResponse:
        return FileResponse(STATIC / "og.png", media_type="image/png")

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")

    @app.get("/robots.txt", include_in_schema=False)
    async def robots() -> Response:
        return Response(ROBOTS, media_type="text/plain")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "fake_llm": settings.fake_llm,
            "reader_model": settings.reader_model,
            "decider_model": settings.decider_model,
            "vision_model": settings.vision_model,
            "max_scan_pages": settings.max_scan_pages,
            "llm_ready": settings.fake_llm or bool(settings.nebius_api_key),
            "find_ready": bool(settings.tavily_api_key),
            "example_ready": EXAMPLE.is_file(),
            "max_pdf_mb": settings.max_pdf_mb,
        }

    @app.get("/api/example")
    async def example() -> FileResponse:
        if not EXAMPLE.is_file():
            raise HTTPException(404, "no example on this server")
        return FileResponse(EXAMPLE, media_type="application/json")

    @app.get("/api/example/checklist.{fmt}")
    async def example_checklist(fmt: str) -> Response:
        if not EXAMPLE.is_file():
            raise HTTPException(404, "no example on this server")
        return checklist_response(RunResult.model_validate_json(EXAMPLE.read_bytes()), fmt)

    @app.post("/api/find")
    async def find(req: FindRequest, request: Request) -> dict[str, Any]:
        if settings.tavily_api_key:
            check_search_limits(client_address(request))
        try:
            hits = await search_tenders(req.query, settings.tavily_api_key, req.max_results)
        except FindError as exc:
            raise HTTPException(503 if "not set" in str(exc) else 502, str(exc)) from exc
        return {"query": req.query, "results": [h.model_dump() for h in hits]}

    @app.post("/api/documents")
    async def documents(req: DocumentsRequest, request: Request) -> dict[str, Any]:
        if not req.url.strip().startswith(("http://", "https://")):
            raise HTTPException(400, "only http and https URLs are allowed")
        if not settings.tavily_api_key:
            raise HTTPException(503, "TAVILY_API_KEY is not set")
        llm_client = client()
        check_search_limits(client_address(request))
        try:
            found = await find_documents(req.url.strip(), req.query[:300], settings, llm_client)
        except FindError as exc:
            raise HTTPException(502, str(exc)) from exc
        except FetchError as exc:
            raise HTTPException(400, str(exc)) from exc
        return found.model_dump()

    @app.post("/api/runs")
    async def start_run(
        request: Request,
        url: str = Form(""),
        profile: str = Form(""),
        file: UploadFile | None = File(None),
    ) -> dict[str, str]:
        llm_client = client()
        pdf: bytes | None = None
        name: str | None = None
        if file is not None and file.filename:
            pdf = await file.read(settings.max_pdf_bytes + 1)
            if len(pdf) > settings.max_pdf_bytes:
                raise HTTPException(413, f"PDF is larger than {settings.max_pdf_mb} MB")
            name = file.filename
        elif url.strip():
            if not url.strip().startswith(("http://", "https://")):
                raise HTTPException(400, "only http and https URLs are allowed")
        else:
            raise HTTPException(400, "give a PDF URL or upload a PDF")
        if len(profile) > 4000:
            raise HTTPException(400, "profile is longer than 4000 characters")
        check_limits(client_address(request))

        prune()
        run = Run(id=uuid.uuid4().hex)
        runs[run.id] = run

        async def work() -> None:
            try:
                result, pdf_bytes = await run_reader(
                    settings=settings,
                    llm=llm_client,
                    profile=profile,
                    url=None if pdf is not None else url.strip(),
                    pdf=pdf,
                    source_name=name,
                    emit=run.emit,
                )
                run.pdf = pdf_bytes
                run.result = result.model_dump()
                await run.emit({"type": "result", "run_id": run.id, "result": run.result})
            except Exception as exc:
                await run.emit({"type": "error", "message": str(exc) or exc.__class__.__name__})
            finally:
                run.done = True
                run.changed.set()

        run.task = asyncio.create_task(work())
        return {"run_id": run.id, "events": f"/api/runs/{run.id}/events"}

    def get_run(run_id: str) -> Run:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found (runs are kept in memory for one hour)")
        return run

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        run = get_run(run_id)

        async def stream() -> AsyncIterator[str]:
            sent = 0
            while True:
                waiter = run.changed
                while sent < len(run.events):
                    event = run.events[sent]
                    sent += 1
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                if run.done and sent >= len(run.events):
                    return
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> JSONResponse:
        run = get_run(run_id)
        return JSONResponse({"run_id": run.id, "done": run.done, "events": len(run.events), "result": run.result})

    @app.get("/api/runs/{run_id}/checklist.{fmt}")
    async def run_checklist(run_id: str, fmt: str) -> Response:
        run = get_run(run_id)
        if run.result is None:
            raise HTTPException(404, "the checklist is ready when the run has finished")
        return checklist_response(RunResult.model_validate(run.result), fmt)

    @app.get("/api/runs/{run_id}/pdf")
    async def run_pdf(run_id: str) -> Response:
        run = get_run(run_id)
        if run.pdf is None:
            raise HTTPException(404, "PDF not ready")
        return Response(
            run.pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="tender.pdf"', "X-Content-Type-Options": "nosniff"},
        )

    return app


app = create_app()
