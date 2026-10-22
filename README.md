# Lucius Open Reader

Read a public tender before you bid.

Give it a public tender PDF (or search for one). NVIDIA Nemotron 3 Super on Nebius Token Factory reads
the document six pages at a time and lists every requirement with a verbatim quote and its page.
Scanned pages, which have no text layer, are first read from the page image by NVIDIA Nemotron 3.5
Super VL. Every quote is then checked against the page text, so you can see which items are grounded
and which are not. Add a few lines about your company and Nemotron 3 Ultra judges each mandatory
requirement against it; the bid / no-bid call follows from those judgments, so every call can be
traced to the pages behind it. The result downloads as a bid checklist (Excel or CSV).

On a real 44-page municipal tender (South Africa, sewage pump spares) a run takes 30 to 45 seconds:
all 44 pages read, 78 to 96 requirements, 96 to 100% of quotes found on their page, the buyer and
title read from the tender, and a stable "bid with conditions" for a pump supplier and "no bid" for
an IT consultancy.

Built for the Nebius x NVIDIA Global AI Hackathon, track "Best Apps and Agents". Apache-2.0.

**Demo video (2:22):** <https://youtu.be/BwU8NtvhWL8>

**Live demo:** <https://lucius-open-reader-384599766402.europe-west1.run.app> (reads are capped per
visitor and per day; "Open a finished example" shows a saved live run without calling the models).

## What it does

1. **Find (optional).** A search such as `water supply works South Africa` goes to the Tavily Search
   API (advanced depth). Paywalled tender aggregators are excluded; government and multilateral PDFs
   come first, then other PDFs, then government pages. Each result shows a buyer name when one is
   visible on it.
   **Find its documents.** Most results are notice or listing pages, not documents. For those,
   `docs.py` runs a small agent: Tavily Extract (advanced) reads the page as markdown, Nemotron 3
   Super (high reasoning, temperature 0.6) picks the links that are this tender's documents (main
   tender document first) or, if there are none, the pages to open next (at most two levels), and
   every pick is probed: only links whose first bytes are a PDF are offered to read. A link that is
   itself a PDF (such as `/download/3001`) is recognised at once. Tavily Map and Crawl returned
   nothing on the council sites we tried, while Extract read them, hence the step loop.
2. **Read.** The PDF is downloaded (http/https only, 40 MB cap, timeout, private addresses blocked),
   split into pages with PyMuPDF, grouped into chunks of about 6 pages, and each chunk is sent to the
   reader model (Nemotron 3 Super) with a JSON schema. Chunks run concurrently with a small limit.
   Each requirement has: `requirement` (plain English), `quote` (verbatim), `page`, `category`
   (eligibility, technical, commercial, legal, submission, other) and `mandatory`.
   **Scanned pages.** A page with no usable text layer but an image over it is drawn as an image and
   transcribed by the vision model (Nemotron 3.5 Super VL); the transcription becomes that page's text
   for the reader and for the quote check. See [Scanned pages](#scanned-pages).
3. **Verify.** Each quote is normalised (case, whitespace, punctuation, hyphenated line breaks) and
   matched against the cited page by word 5-grams. If 80% or more of the quote's 5-grams are on the
   page it is `verified`. If not, the neighbouring pages (and the break between two pages) are tried
   and the page number is corrected when the quote is found there. A quote the model stitched from
   two or three passages of one page (it skipped a clause) passes only if every piece, at least four
   words long and in page order, is on the page word for word; it is labelled "joined". Nothing is
   dropped: unverified items stay in the list with a flag and their match score.
4. **Decide.** Three steps with the decider model (Nemotron 3 Ultra), so no single answer is long:
   - *scope*: from the first three pages, the tender's title and buyer (kept only if they appear on
     those pages), what it buys, and whether that is the company's line of work (`fits`, `partly`,
     `outside`);
   - *assess*: every mandatory requirement, in batches of 20 run side by side, is judged against the
     profile: `fails`, `not stated` (must be confirmed or obtained), `met` or `routine` (a form any
     bidder fills in), with `hard` when it takes weeks, money or a third party. Ids the model skipped
     are asked again once; any still missing are reported, never guessed;
   - *summary*: up to four reasons and up to eight conditions, from the judgments.

   The model never picks the verdict. Work outside the company's line, or any `fails`, gives
   `no bid`; any `not stated` gives `bid with conditions`; otherwise `bid`. The requirements that
   decide it are listed failures first, then hard to obtain, then by category. Why: on the same list
   a single free-form call answered "bid 0.92", "no bid 0.95" and "bid with conditions 0.75" in three
   runs, and one call judging all 78 items at once skipped two thirds of them.
5. **Export.** "Download the checklist" gives an Excel workbook built on the server from the
   finished run: a *Checklist* sheet (No., requirement, the tender's words, page, category,
   mandatory, judgment, why, quote check, and an empty *Done* column with a yes / no / not needed
   list) and a *Summary* sheet (tender, buyer, source, pages read, scanned pages, the call with its
   confidence, reasons, conditions, the requirements that decide it, models used). "or as CSV" gives
   the Checklist sheet as UTF-8 CSV with the call in a last column on every row. The quote check
   column says `found`, `found, page fixed`, `found, joined` or `not found`, plus `scanned page`
   when the quote was checked against a transcription. Tender text, model output, the profile and
   the file name never run as formulas: in the workbook a value starting with `=`, `+`, `-`, `@`, a
   tab or a carriage return is a text cell with Excel's quote prefix (shown exactly as written); in
   the CSV it gets a leading apostrophe.
6. **Web UI.** One HTML page (vanilla JS, no build step; fonts from Google Fonts): search, PDF URL
   or upload, profile, live progress over Server-Sent Events with a ledger of the PDF's pages that
   fill in as they are read (scanned pages carry a folded corner), then the call (tender, buyer,
   scope fit, tally, reasons, checklist, downloads), the requirements that decide it, and a
   filterable table. Each page number links to the PDF at `#page=N`; a page tile filters the table
   to that page. Requirements from scanned pages carry a `scanned page` tag. "Open a finished
   example" shows a saved live run without calling the models.
7. **CLI.** `python -m reader <pdf-url-or-path> --profile "..."` prints the full result as JSON; with
   `--progress` the progress events and, at the end, the tokens used per model go to stderr.

## Architecture

```
                 +--------------------- reader/app.py (FastAPI) ----------------------+
 browser  ---->  |  GET /            static/index.html                                |
 (SSE)    <----  |  POST /api/find   -> find.py ---- HTTPS ----> Tavily /search        |
                 |  POST /api/runs   -> pipeline.py (background task, events)         |
                 |  GET  /api/runs/{id}/events   (text/event-stream)                  |
                 |  GET  /api/runs/{id}/pdf      (the PDF that was read)              |
                 |  GET  /api/runs/{id}/checklist.xlsx | .csv   -> export.py          |
                 +-------------------------------------------------------------------+
                                          |
   CLI: python -m reader ---------------->|
                                          v
 pipeline.py:  fetch.py -> pages.py -+-> extract.py ----------> verify.py -> decide.py
               download    PyMuPDF   |   chunks of 6 pages      5-gram       bid / no-bid
               40 MB cap   per-page  |   N concurrent calls     quote check  ids checked
               no private  text,     |   (a chunk waits only         |            |
               hosts       scans     +-> scan.py: its own scans)     |            |
                                         scanned page -> image       |            |
                                         -> transcription            |            |
                                          |                          v            v
                                    llm.py  -- OpenAI-compatible HTTPS -->  Nebius Token Factory
                                    (NebiusClient | FakeLLM)                READER_MODEL  (Nemotron 3 Super)
                                                                            VISION_MODEL  (Nemotron 3.5 Super VL)
                                                                            DECIDER_MODEL (Nemotron 3 Ultra)
```

Modules (all in `reader/`): `config.py` (env settings), `models.py` (pydantic shapes), `find.py`,
`fetch.py`, `pages.py`, `scan.py`, `extract.py`, `verify.py`, `decide.py`, `export.py`, `llm.py`,
`pipeline.py`, `app.py`, `__main__.py` (CLI), `static/index.html`.

## How it uses Nemotron on Nebius Token Factory

Nebius Token Factory serves an OpenAI-compatible API. `llm.py` calls it directly with httpx:

- `POST {NEBIUS_BASE_URL}chat/completions`, header `Authorization: Bearer $NEBIUS_API_KEY`
- default base URL `https://api.tokenfactory.nebius.com/v1/`
  (from <https://docs.tokenfactory.nebius.com/api-reference/introduction.md>)
- `response_format = {"type": "json_schema", "json_schema": {"name", "schema", "strict": true}}`;
  if the deployment answers 400/422 it retries once with `{"type": "json_object"}` and the schema in
  the system prompt (both forms are documented at
  <https://docs.tokenfactory.nebius.com/ai-models-inference/json.md>)
- `temperature = 0`; reasoning blocks (`<think>...</think>`) and code fences are stripped before parsing
- `reasoning_effort = "low"` for both models. Measured on the 44-page tender: at the default effort
  Nemotron 3 Super spent the whole 8,192-token budget thinking on dense pages and returned an empty
  answer (`finish_reason: "length"`) for 2 of 8 chunks; at `"medium"` it thought for 12,486 tokens and
  returned no requirements; at `"low"` every chunk answered in about 8 seconds. For the decider,
  `"medium"` called boilerplate collusion declarations decisive. If an answer still comes back empty
  the client asks once more with thinking off (`chat_template_kwargs = {"enable_thinking": false}`).
- Scanned pages go to the vision model as one user message with a text part and an `image_url` part
  holding a base64 data URL (`data:image/png;base64,...`), thinking off, temperature 0, at most 8,192
  answer tokens; it answers in plain text, not JSON (a page of text inside a JSON string would need
  every quote mark escaped, and one broken escape would lose the page). A reply that runs out of
  tokens (`finish_reason: "length"`) is treated as a failure, since a page is far shorter than that.
- Each reply's `usage` is added up per model (`NebiusClient.usage`), which the CLI prints at the end.

Default models, taken from the Nebius model catalog <https://tokenfactory.nebius.com/model-catalog.md>
(checked 15 Oct 2026; the vision model 22 Oct 2026, in `GET {NEBIUS_BASE_URL}models` and with a live
call that transcribed a test page word for word in 1 second):

| Role | Env var | Default id | USD per 1M tokens (in / out) |
|---|---|---|---|
| Reader (per chunk) and document finder | `READER_MODEL`, `FINDER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | 0.30 / 0.90 |
| Scanned pages (one call per page image) | `VISION_MODEL` | `nvidia/Nemotron-3.5-Super-VL` | 0.30 / 0.90 |
| Decider (scope, assessments, summary) | `DECIDER_MODEL` | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1.00 / 3.00 |

Prices are the catalog's on 22 Oct 2026.

The live list for your account is at `GET {NEBIUS_BASE_URL}models`; override the env vars if an id differs.

## Scanned pages

Many tenders mix typed pages with scans: signed forms, annexes, a council's ordinance. A scan has no
text layer, so before this step those pages were reported as empty and nothing on them reached the
list.

1. **Detect.** `pages.py` marks a page as a scan when its text layer has fewer than 50 characters
   (a portal may stamp "Page 3 of 40" on a scan) and images cover at least 30% of it. A page with
   no text and no such image is blank, and is reported as blank.
2. **Render.** The page is drawn in grey at 150 dpi (an A4 page is 1240 x 1755 pixels, about 2,200
   input tokens), capped at 2,000 pixels on the long side so an A1 drawing does not cost twenty A4
   pages; PNG, or JPEG when the PNG is over 900 KB and the JPEG is smaller.
3. **Transcribe.** `scan.py` sends the image to Nemotron 3.5 Super VL with an instruction to copy
   every word exactly as printed, in reading order, tables one row per line, and to leave out line
   numbers printed in the margin. A page is asked twice at most; a page that still fails is reported
   as not read, never filled in. The transcription becomes the page's text.
4. **Read and check.** The reader reads the transcription with the rest of its chunk. The quote
   check runs against the transcription: on a scanned page it proves the quote is in the vision
   model's transcription of the image, not in a text layer. Every such requirement is tagged
   `scanned page` on the page, in the checklist and in the export, and the result says which pages
   were read from an image.
5. **Bounded.** At most `READER_MAX_SCAN_PAGES` (default 40) scanned pages are read per run, the
   first ones in the document; the rest are marked on the ledger and named in a warning. Page images
   go out `READER_CONCURRENCY` at a time while the text chunks are read: a chunk waits only for its
   own scanned pages, and a free reader slot goes to the waiting chunk earliest in the document.
   MuPDF is not thread-safe, so page extraction and rendering run one at a time.

Measured on 22 Oct 2026 (live models, 4 calls at a time, the main branch before this step against
this version; one run per line unless two are given):

| Tender | Code | Pages | Read from an image | Requirements | Quotes found on their page | Time | Cost (USD) |
|---|---|---|---|---|---|---|---|
| 120-page municipal pump-station tender (South Africa), no scans | before | 120 | none | 150 | 150 of 150 | 88 s | 0.12 |
| same | after (2 runs) | 120 | none | 189; 133 | all | 88 s; 48 s | 0.17; 0.11 |
| 375-page pump-station rehabilitation tender (US city), pages 18 to 26 are a scanned city ordinance | before (2 runs) | 375 | none: 9 pages reported as having no text | 259; 351 (none from pages 18 to 26) | 257; 346 | 105 s; 128 s | 0.28; 0.37 |
| same | after | 375 | 9 of 9, done 14 s into the run | 214 (11 from the scans) | 214 of 214 (11 of 11 on scans) | 105 s | 0.31 (0.0097 of it for the 9 pages) |

On the 375-page tender the scans add the apprentice rules of the ordinance: at least 12% of labour
hours by registered apprentices in the trades that have them, monthly certified reports, a fee for
each hour short, and debarment for non-compliance; "certify participation in an apprenticeship
programme" was among the requirements that decide the call. A scanned page costs about 2,250 input
and 450 output tokens, so about USD 0.001; the cap of 40 keeps a run's scans under USD 0.05. The
requirement count and the call vary from run to run on the same tender (here 133 to 189, and 214 to
404 in three runs with scans read on the 375-page one); the time follows the number of requirements
the decider judges more than the scans.

Two things the measurements changed. On the first run one quote of ten from the scans failed the
check: the ordinance numbers every line in the margin, the transcription kept those numbers, and
the reader's quotes did not; the vision prompt now leaves margin line numbers out, and 21 of 21
quotes from the scans in the next two runs were found. And the chunks that waited for their scans
went to the back of the reader queue and became the slow tail of the read (93 s for about 74 s of
work); since free slots go to the earliest waiting chunk, the read ends with the rest (42 s for 41 s
of work).

## How it uses Tavily

`find.py` sends `POST https://api.tavily.com/search` with `Authorization: Bearer $TAVILY_API_KEY` and
`{"query", "search_depth": "basic", "topic": "general", "max_results"}` (reference:
<https://docs.tavily.com/documentation/api-reference/endpoint/search>). The word "tender" is added to
the query when no procurement word is present. Results whose path ends in `.pdf` come first.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env   # fill in your keys; .env is git-ignored
set -a; . ./.env; set +a
.venv/bin/uvicorn reader.app:app --port 8000
# open http://127.0.0.1:8000
```

Offline demo, no keys and no network calls to any model:

```bash
READER_FAKE_LLM=1 .venv/bin/uvicorn reader.app:app --port 8000
```

The fake client is rule-based (it picks sentences with "must", "shall", "required" and so on). It
exists for tests and demos; the real reading is done by Nemotron.

CLI:

```bash
.venv/bin/python -m reader https://example.gov/tender.pdf --profile "Civil works contractor, grade 7, ISO 9001" --progress
.venv/bin/python -m reader ./tender.pdf --fake
```

Tests (offline: a fake LLM, PDFs generated in the test with PyMuPDF, including image-only scans,
mocked HTTP, and a replay of real Token Factory replies recorded in `tests/fixtures/scan_run.json`
for a 4-page PDF with two scanned pages; re-record it with `tests/fixtures/record_scan_run.py` and a
key when a prompt changes):

```bash
.venv/bin/python -m pytest -q
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NEBIUS_API_KEY` | none | Nebius Token Factory API key (required unless fake mode) |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1/` | OpenAI-compatible base URL |
| `READER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | model that reads each chunk |
| `DECIDER_MODEL` | `nvidia/Nemotron-3-Ultra-550b-a55b` | model that makes the bid call |
| `VISION_MODEL` | `nvidia/Nemotron-3.5-Super-VL` | model that transcribes scanned pages |
| `VISION_EFFORT` | `none` | its thinking (copying a page needs none) |
| `READER_MAX_SCAN_PAGES` | `40` | scanned pages read from the image per run (`0`: none) |
| `READER_SCAN_DPI` | `150` | resolution of the page image (72 to 300) |
| `TAVILY_API_KEY` | none | enables Find |
| `READER_FAKE_LLM` | `0` | `1` uses the offline fake client |
| `READER_CONCURRENCY` | `4` | chunks read at the same time; also page images transcribed at the same time |
| `READER_CHUNK_PAGES` | `6` | pages per chunk |
| `READER_MAX_PDF_MB` | `40` | download and upload size cap |
| `READER_FETCH_TIMEOUT` | `60` | seconds for the PDF download |
| `READER_LLM_TIMEOUT` | `180` | seconds per model call |
| `READER_MAX_PAGES` | `500` | longer PDFs are refused |
| `READER_VERIFY_THRESHOLD` | `0.8` | share of quote 5-grams that must be on the page |
| `READER_EFFORT` | `low` | reader thinking: `none`, `low`, `medium`, `high` or `default` |
| `DECIDER_EFFORT` | `low` | decider thinking, same values |
| `READER_MAX_TOKENS` | `16384` | answer budget per model call (thinking included) |
| `READER_RUNS_PER_HOUR_PER_IP` | `0` | runs one address may start per hour (`0`: no limit) |
| `READER_RUNS_PER_DAY` | `0` | runs the server starts per day (`0`: no limit); searches get four times both limits |
| `FINDER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | model that picks a page's tender documents |
| `FINDER_EFFORT` | `high` | its thinking effort |
| `FINDER_TEMPERATURE` | `0.6` | its temperature (NVIDIA's sampling for Nemotron when it reasons) |
| `READER_ALLOW_PRIVATE_URLS` | `0` | `1` allows localhost / private hosts (local testing only) |

## Deploy on Nebius Serverless Endpoints

The app is a single container listening on port 8000. Build and push the image to a registry the
endpoint can pull from (Nebius Container Registry in the same project needs no credentials):

```bash
docker build -t <registry>/<path>/lucius-open-reader:0.1.0 .
docker push <registry>/<path>/lucius-open-reader:0.1.0
```

Create the endpoint with the Nebius CLI (flags from
<https://docs.nebius.com/serverless/quickstart/endpoints.md> and
<https://docs.nebius.com/serverless/endpoints/manage.md>). Keep the API keys in a Nebius secret and
pass them with `--env-secret`, not `--env`:

```bash
export AUTH_TOKEN=$(openssl rand -hex 32)
nebius ai endpoint create \
  --name lucius-open-reader \
  --image <registry>/<path>/lucius-open-reader:0.1.0 \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --public \
  --container-port 8000 \
  --auth token \
  --token "$AUTH_TOKEN" \
  --env-secret NEBIUS_API_KEY=<secret_selector> \
  --env-secret TAVILY_API_KEY=<secret_selector> \
  --subnet-id <subnet_ID>

export ENDPOINT_ID=$(nebius ai endpoint get-by-name --name lucius-open-reader --format jsonpath='{.metadata.id}')
nebius ai endpoint get "$ENDPOINT_ID" --format json | jq -r '.status.public_endpoints[] | select(startswith("https://"))'
curl -H "Authorization: Bearer $AUTH_TOKEN" https://<endpoint-url>/api/health
```

With `--auth token` every request needs the bearer token, which a plain browser does not send. For a
public demo page, use `--auth none` and rely on the app's own limits (`READER_RUNS_PER_HOUR_PER_IP`,
`READER_RUNS_PER_DAY`), or put the endpoint behind a proxy that adds the header. The same limits
bound the vision model's cost: at most `READER_MAX_SCAN_PAGES` page images per run. The per-address
limit reads `X-Forwarded-For`, which a client can set, so the daily cap is the real guard. Runs are held in memory (last 20, one hour), so run a single replica.

## Limits and honest notes

- Scanned pages are read from their image, up to `READER_MAX_SCAN_PAGES` a run; scans over the cap,
  and scans the vision model fails on twice, are named on the page and in the export, not read. On a
  scanned page the quote check proves the quote is in the model's transcription, not that the
  transcription is right: compare the quote with the page image. Handwriting and stamps may be
  transcribed badly or as `[illegible]`.
- A page whose text was turned into vector outlines (some drawings) has neither text nor an image, so
  it is reported as blank rather than read.
- The SSRF guard checks every redirect hop and every resolved address, but DNS can change between the
  check and the connection; run the service without access to internal networks.
- The verify step proves a quote is on the page. It does not prove the plain-English requirement is a
  fair summary of the quote; read the quote.
- Verdicts are advice from a model, not a legal or commercial opinion.

## License

Apache License 2.0. See `LICENSE`.
