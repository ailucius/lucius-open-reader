"""Scanned pages, offline: image-only PDFs generated in the test, a fake vision model, mocked HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pymupdf
import pytest

from reader.config import Settings
from reader.llm import FakeLLM, LLMError, NebiusClient
from reader.models import Page, Requirement
from reader.extract import OrderedSlots, extract_requirements
from reader.pages import MAX_SIDE_PX, chunk_pages, extract_pages, render_page
from reader.pipeline import run_reader
from reader.scan import VISION_SYSTEM, clean_transcription, read_scans
from reader.verify import verify_requirements
from tests.conftest import TENDER_PAGES, make_scanned_pdf

SCAN_TEXT = {
    2: TENDER_PAGES[1],  # eligibility: two requirements
    3: TENDER_PAGES[3],  # commercial: two requirements
}


def test_detects_scans_but_not_blank_pages_or_logos() -> None:
    pdf = make_scanned_pdf(
        [
            ("text", TENDER_PAGES[0]),
            ("scan", TENDER_PAGES[1]),
            ("blank", ""),
            ("stamp", TENDER_PAGES[3]),
            ("logo", "LOGO"),
        ]
    )
    pages = extract_pages(pdf)
    assert [p.scan for p in pages] == [False, True, False, True, False]
    assert pages[1].text.strip() == ""  # an image only: no text layer at all
    assert "Page 4 of 9" in pages[3].text  # the stamp is a text layer, but far too short to be the page
    assert not any(p.transcribed for p in pages)


def test_render_page_is_grey_png_at_the_dpi_and_caps_large_sheets() -> None:
    pdf = make_scanned_pdf([("scan", TENDER_PAGES[1])])
    image, mime = render_page(pdf, 1, dpi=150)
    assert mime == "image/png" and image[:8] == b"\x89PNG\r\n\x1a\n"
    pix = pymupdf.Pixmap(image)
    assert (pix.width, pix.height) == (1240, 1755) and pix.n == 1  # A4 at 150 dpi, one grey channel

    doc = pymupdf.open()
    doc.new_page(width=2384, height=3370)  # A1, as tender drawings often are
    image, _ = render_page(doc.tobytes(), 1, dpi=150)
    pix = pymupdf.Pixmap(image)
    assert max(pix.width, pix.height) <= MAX_SIDE_PX


def test_clean_transcription() -> None:
    assert clean_transcription("NO TEXT") == ""
    assert clean_transcription("  no readable text. ") == ""
    assert clean_transcription("```text\nBidders must register.\n```") == "Bidders must register."
    assert clean_transcription("No text may be added to the form.") == "No text may be added to the form."


class ScriptedVision(FakeLLM):
    """Page 2 fails once then reads; page 3 always fails; page 4 has no text; others read as given."""

    def __init__(self, transcripts: dict[int, str]) -> None:
        super().__init__(transcripts)
        self.attempts: dict[int, int] = {}

    async def chat_image(self, **kwargs: Any) -> str:
        page = int(kwargs["user"].split()[1])
        self.attempts[page] = self.attempts.get(page, 0) + 1
        self.image_calls.append(kwargs)
        if page == 2 and self.attempts[page] == 1:
            raise LLMError("Nebius API 503: busy")
        if page == 3:
            raise LLMError("image reply is empty (finish_reason 'stop')")
        if page == 4:
            return "NO TEXT"
        return self.transcripts[page]


async def test_read_scans_transcribes_in_place_retries_once_and_never_invents() -> None:
    pdf = make_scanned_pdf([("text", TENDER_PAGES[0])] + [("scan", TENDER_PAGES[i]) for i in (1, 2, 3)])
    pages = extract_pages(pdf)
    llm = ScriptedVision({2: "Bidders must be registered at grade 7."})
    done: list[tuple[int, bool]] = []

    async def on_page(page: Page, ok: bool) -> None:
        done.append((page.number, ok))

    outcome = await read_scans(pdf, pages, llm, "vl", cap=40, dpi=72, concurrency=2, effort="none", on_page=on_page)
    assert outcome.read == [2, 4] and outcome.failed == [3] and outcome.skipped == []
    assert outcome.errors == ["page 3: image reply is empty (finish_reason 'stop')"]
    assert llm.attempts == {2: 2, 3: 2, 4: 1}  # one retry, then the page is reported, not filled in
    assert sorted(done) == [(2, True), (3, False), (4, True)]
    assert pages[1].text == "Bidders must be registered at grade 7." and pages[1].transcribed
    assert pages[2].text.strip() == "" and not pages[2].transcribed
    assert pages[3].text == "" and pages[3].transcribed  # read: the page has no text on it
    assert pages[0].text.startswith("Invitation to Tender") and not pages[0].transcribed
    call = llm.image_calls[0]
    assert call["system"] == VISION_SYSTEM and call["mime"] == "image/png" and call["effort"] == "none"
    assert call["image"][:8] == b"\x89PNG\r\n\x1a\n"


async def test_scan_cap_reads_the_first_pages_and_reports_the_rest() -> None:
    pdf = make_scanned_pdf([("scan", t) for t in TENDER_PAGES[:5]], dpi=50)
    pages = extract_pages(pdf)
    llm = FakeLLM({n: f"Page {n} text." for n in range(1, 6)})
    outcome = await read_scans(pdf, pages, llm, "vl", cap=2, dpi=50)
    assert outcome.read == [1, 2] and outcome.skipped == [3, 4, 5]
    assert len(llm.image_calls) == 2

    pages = extract_pages(pdf)
    llm = FakeLLM({})
    outcome = await read_scans(pdf, pages, llm, "vl", cap=0)
    assert outcome.read == [] and outcome.skipped == [1, 2, 3, 4, 5] and llm.image_calls == []


def test_quote_check_runs_against_the_transcription() -> None:
    # what the model transcribed is the page; a quote the transcription does not hold is not found,
    # even if it might be on the image
    page = Page(number=2, text="Bidders must be registered with the national contractor board at grade 7 or higher.", scan=True, transcribed=True)
    reqs = [
        Requirement(requirement="r", quote="registered with the national contractor board at grade 7", page=2),
        Requirement(requirement="r", quote="three pipeline projects of similar size in the past five years", page=2),
    ]
    out = verify_requirements(reqs, [page])
    assert [r.verified for r in out] == [True, False]


async def test_pipeline_reads_a_mixed_pdf_and_marks_what_came_from_scans() -> None:
    pdf = make_scanned_pdf(
        [("text", TENDER_PAGES[0]), ("scan", TENDER_PAGES[1]), ("scan", TENDER_PAGES[3]), ("text", TENDER_PAGES[4]), ("blank", "")]
    )
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    settings = Settings(fake_llm=True, chunk_pages=3, concurrency=2, scan_dpi=72)
    llm = FakeLLM(SCAN_TEXT)
    result, _ = await run_reader(settings=settings, llm=llm, profile="Pipeline contractor, grade 6.", pdf=pdf, emit=emit)

    pages_event = next(e for e in events if e["type"] == "pages")
    assert pages_event["scans"] == [2, 3] and pages_event["scans_over_cap"] == [] and pages_event["empty_pages"] == [5]
    scans = [e for e in events if e["type"] == "scan"]
    assert sorted(e["page"] for e in scans) == [2, 3] and all(e["ok"] for e in scans)
    # the chunk holding the scans is read only after both are transcribed (the text-only chunk 4-5 may
    # finish earlier)
    order = [(e["type"], e.get("page") or e.get("chunk")) for e in events if e["type"] in ("scan", "progress")]
    assert order.index(("progress", "1-3")) > max(order.index(("scan", 2)), order.index(("scan", 3)))

    s = result.stats
    assert s.scanned_pages == [2, 3] and s.scan_failed_pages == [] and s.scan_skipped_pages == [] and s.scan_cap == 40
    assert s.pages_read == 5 and s.empty_page_numbers == [5]
    assert result.vision_model == settings.vision_model and result.read_at
    by_page: dict[int, list[Requirement]] = {}
    for r in result.requirements:
        by_page.setdefault(r.page, []).append(r)
    assert {2, 3, 4} <= set(by_page)
    assert all(r.scanned for p in (2, 3) for r in by_page[p])
    assert not any(r.scanned for r in by_page[4])
    assert all(r.verified for r in result.requirements)
    assert any("grade 7" in r.quote for r in by_page[2]) and any("bid guarantee" in r.quote for r in by_page[3])
    assert sorted(c["user"] for c in llm.image_calls) == ["Page 2 of the tender. Transcribe it.", "Page 3 of the tender. Transcribe it."]
    assert result.decision is not None and result.decision.verdict == "bid with conditions"


async def test_pipeline_reports_scans_it_could_not_read_and_scans_over_the_cap() -> None:
    pdf = make_scanned_pdf([("scan", TENDER_PAGES[1]), ("scan", TENDER_PAGES[3]), ("scan", TENDER_PAGES[4])], dpi=50)
    settings = Settings(fake_llm=True, chunk_pages=3, max_scan_pages=2, scan_dpi=50)
    llm = FakeLLM({1: TENDER_PAGES[1]})  # page 2 is not in the fake's transcripts: it fails, twice
    result, _ = await run_reader(settings=settings, llm=llm, pdf=pdf)
    s = result.stats
    assert s.scanned_pages == [1] and s.scan_failed_pages == [2] and s.scan_skipped_pages == [3]
    assert s.pages_read == 1
    assert len(llm.image_calls) == 3  # page 1 once, page 2 twice, page 3 never
    assert {r.page for r in result.requirements} == {1}
    assert all(r.scanned for r in result.requirements)


async def test_nebius_chat_image_request_shape() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>Bidders must register."}, "finish_reason": "stop"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "line\n" * 50}, "finish_reason": "length"}]})

    client = NebiusClient("nb-test", "https://api.tokenfactory.nebius.com/v1/", transport=httpx.MockTransport(handler))
    text = await client.chat_image(model="nvidia/Nemotron-3.5-Super-VL", system="s", user="Page 2 of the tender. Transcribe it.", image=b"\x89PNG-bytes", mime="image/png", effort="none")
    assert text == "Bidders must register."
    body = bodies[0]
    assert body["model"] == "nvidia/Nemotron-3.5-Super-VL" and body["temperature"] == 0.0 and body["max_tokens"] <= 8192
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "response_format" not in body
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Page 2 of the tender. Transcribe it."}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode()
    with pytest.raises(LLMError, match="ran out of tokens"):  # a reply that hits the budget is a loop, not a page
        await client.chat_image(model="m", system="s", user="u", image=b"x", mime="image/png", effort="none")


class GatedVision(FakeLLM):
    """Holds every page image until the test opens the gate, and records when the reader starts."""

    def __init__(self, transcripts: dict[int, str]) -> None:
        super().__init__(transcripts)
        self.gate = asyncio.Event()
        self.log: list[str] = []

    async def chat_image(self, **kwargs: Any) -> str:
        await self.gate.wait()
        self.log.append("scan " + kwargs["user"].split()[1])
        return await super().chat_image(**kwargs)

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["schema_name"] == "requirements":
            self.log.append(kwargs["user"].splitlines()[0])
        return await super().chat_json(**kwargs)


async def test_text_chunks_are_read_while_scans_are_transcribed() -> None:
    # pages 1-2 text, 3-4 scans: chunk 1-2 must not wait for the vision model; chunk 3-4 must
    pdf = make_scanned_pdf([("text", TENDER_PAGES[0]), ("text", TENDER_PAGES[4]), ("scan", TENDER_PAGES[1]), ("scan", TENDER_PAGES[3])], dpi=50)
    llm = GatedVision({3: TENDER_PAGES[1], 4: TENDER_PAGES[3]})
    settings = Settings(fake_llm=True, chunk_pages=2, scan_dpi=50)
    run = asyncio.create_task(run_reader(settings=settings, llm=llm, pdf=pdf))
    for _ in range(200):  # let the pipeline run until it can go no further without the scans
        await asyncio.sleep(0)
        if llm.log:
            break
    await asyncio.sleep(0.05)
    assert llm.log == ["Tender pages 1 to 2."]  # the text chunk was read; the scanned chunk waits
    llm.gate.set()
    result, _ = await run
    assert llm.log[-1] == "Tender pages 3 to 4." and sorted(llm.log[1:3]) == ["scan 3", "scan 4"]
    assert result.stats.scanned_pages == [3, 4] and result.stats.pages_read == 4
    assert {r.page for r in result.requirements if r.scanned} == {3, 4}


async def test_a_failing_scan_task_never_leaves_a_chunk_waiting() -> None:
    # the scan task dies outright (here: the progress stream breaks on the first scan event) with two
    # scans still queued: the run must stop with the error, not wait for ever on the chunk's scans
    async def emit(event: dict[str, Any]) -> None:
        if event["type"] == "scan":
            raise RuntimeError("the browser went away")

    pdf = make_scanned_pdf([("text", TENDER_PAGES[0])] + [("scan", TENDER_PAGES[i]) for i in (1, 3, 4)], dpi=50)
    llm = FakeLLM({2: TENDER_PAGES[1], 3: TENDER_PAGES[3], 4: TENDER_PAGES[4]})
    settings = Settings(fake_llm=True, chunk_pages=2, concurrency=1, scan_dpi=50)
    with pytest.raises(RuntimeError, match="browser went away"):
        await asyncio.wait_for(run_reader(settings=settings, llm=llm, pdf=pdf, emit=emit), timeout=5)


async def test_ordered_slots_go_to_the_earliest_chunk_and_survive_cancellation() -> None:
    slots = OrderedSlots(1)
    await slots.acquire(0)
    got: list[int] = []

    async def take(key: int) -> None:
        await slots.acquire(key)
        got.append(key)

    late, early, gone = asyncio.create_task(take(9)), asyncio.create_task(take(3)), asyncio.create_task(take(1))
    await asyncio.sleep(0)
    gone.cancel()  # a waiter that goes away while queued must not take or lose a slot
    await asyncio.sleep(0)
    slots.release()
    await asyncio.sleep(0)
    assert got == [3]  # asked second, but earlier in the document
    slots.release()
    await asyncio.gather(late, early)
    assert got == [3, 9]
    slots.release()
    await slots.acquire(5)  # the one slot is free again: no slot was lost to the cancelled waiter
    slots.release()


async def test_a_chunk_that_waited_for_its_scans_is_read_before_later_chunks() -> None:
    order: list[str] = []
    gate = asyncio.Event()

    class Logged(FakeLLM):
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            order.append(kwargs["user"].splitlines()[0])
            if len(order) == 1:
                gate.set()  # chunk 1's scans are done while chunk 2 holds the only slot
                await asyncio.sleep(0.01)
            return await super().chat_json(**kwargs)

    async def scans_of_chunk_1(chunk: list[Page]) -> None:
        if chunk[0].number == 1:
            await gate.wait()

    from tests.conftest import make_pdf

    chunks = chunk_pages(extract_pages(make_pdf(TENDER_PAGES[:5])), 1)
    await extract_requirements(chunks, Logged(), "m", concurrency=1, before_chunk=scans_of_chunk_1)
    assert order == ["Tender pages 2 to 2.", "Tender pages 1 to 1.", "Tender pages 3 to 3.", "Tender pages 4 to 4.", "Tender pages 5 to 5."]
