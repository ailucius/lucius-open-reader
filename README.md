# Lucius Open Reader

Read a public tender before you bid.

Give it a public tender PDF (or search for one). NVIDIA Nemotron on Nebius Token Factory reads the
document six pages at a time and lists every requirement with a verbatim quote and its page. Every
quote is then checked against the page text, so you can see which items are grounded and which are
not. Add a few lines about your company and a second Nemotron model makes a bid / no-bid call that
cites the requirements behind it.

Built for the Nebius x NVIDIA Global AI Hackathon, track "Best Apps and Agents". Apache-2.0.

## What it does

1. **Find (optional).** A search such as `water supply works South Africa` goes to the Tavily Search
   API. Results come back with title, URL, a guessed buyer name when one is visible, and PDF links first.
2. **Read.** The PDF is downloaded (http/https only, 40 MB cap, timeout, private addresses blocked),
   split into pages with PyMuPDF, grouped into chunks of about 6 pages, and each chunk is sent to the
   reader model (Nemotron 3 Super) with a JSON schema. Chunks run concurrently with a small limit.
   Each requirement has: `requirement` (plain English), `quote` (verbatim), `page`, `category`
   (eligibility, technical, commercial, legal, submission, other) and `mandatory`.
3. **Verify.** Each quote is normalised (case, whitespace, punctuation, hyphenated line breaks) and
   matched against the cited page by word 5-grams. If 80% or more of the quote's 5-grams are on the
   page it is `verified`. If not, the neighbouring pages (and the break between two pages) are tried
   and the page number is corrected when the quote is found there. Nothing is dropped: unverified
   items stay in the list with a flag and their match score.
4. **Decide.** The company profile and the numbered requirement list go to the decider model
   (Nemotron 3 Ultra): verdict (`bid`, `bid with conditions`, `no bid`), confidence, the requirements
   that decide it (by id, with pages) and plain reasons. Ids the model cites that are not in the
   list are rejected and reported, so the call can never rest on an invented requirement.
5. **Web UI.** One HTML page (vanilla JS, no build step, no CDN): search, PDF URL or upload, profile,
   Run, live progress over Server-Sent Events, then the verdict card, the deciding requirements, and
   a filterable table. Each page number links to the PDF at `#page=N`.
6. **CLI.** `python -m reader <pdf-url-or-path> --profile "..."` prints the full result as JSON.

## Architecture

```
                 +--------------------- reader/app.py (FastAPI) ----------------------+
 browser  ---->  |  GET /            static/index.html                                |
 (SSE)    <----  |  POST /api/find   -> find.py ---- HTTPS ----> Tavily /search        |
                 |  POST /api/runs   -> pipeline.py (background task, events)         |
                 |  GET  /api/runs/{id}/events   (text/event-stream)                  |
                 |  GET  /api/runs/{id}/pdf      (the PDF that was read)              |
                 +-------------------------------------------------------------------+
                                          |
   CLI: python -m reader ---------------->|
                                          v
 pipeline.py:  fetch.py -> pages.py -> extract.py ----------> verify.py -> decide.py
               download    PyMuPDF     chunks of 6 pages      5-gram       bid / no-bid
               40 MB cap   per-page    N concurrent calls     quote check  ids checked
               no private  text        |                                    |
               hosts                   v                                    v
                                    llm.py  -- OpenAI-compatible HTTPS -->  Nebius Token Factory
                                    (NebiusClient | FakeLLM)                READER_MODEL  (Nemotron 3 Super)
                                                                            DECIDER_MODEL (Nemotron 3 Ultra)
```

Modules (all in `reader/`): `config.py` (env settings), `models.py` (pydantic shapes), `find.py`,
`fetch.py`, `pages.py`, `extract.py`, `verify.py`, `decide.py`, `llm.py`, `pipeline.py`, `app.py`,
`__main__.py` (CLI), `static/index.html`.

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

Default models, taken from the Nebius model catalog <https://tokenfactory.nebius.com/model-catalog.md>
(checked 15 Oct 2026):

| Role | Env var | Default id |
|---|---|---|
| Reader (per chunk) | `READER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` |
| Decider (once) | `DECIDER_MODEL` | `nvidia/Nemotron-3-Ultra-550b-a55b` |

The live list for your account is at `GET {NEBIUS_BASE_URL}models`; override the env vars if an id differs.

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

Tests (offline: a fake LLM, a PDF generated in the test with PyMuPDF, mocked HTTP):

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
| `TAVILY_API_KEY` | none | enables Find |
| `READER_FAKE_LLM` | `0` | `1` uses the offline fake client |
| `READER_CONCURRENCY` | `4` | chunks read at the same time |
| `READER_CHUNK_PAGES` | `6` | pages per chunk |
| `READER_MAX_PDF_MB` | `40` | download and upload size cap |
| `READER_FETCH_TIMEOUT` | `60` | seconds for the PDF download |
| `READER_LLM_TIMEOUT` | `180` | seconds per model call |
| `READER_MAX_PAGES` | `500` | longer PDFs are refused |
| `READER_VERIFY_THRESHOLD` | `0.8` | share of quote 5-grams that must be on the page |
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
public demo page, use `--auth none` and rely on the app's own limits, or put the endpoint behind a
proxy that adds the header. Runs are held in memory (last 20, one hour), so run a single replica.

## Limits and honest notes

- Scanned PDFs without a text layer are not read (no OCR); the progress log says how many pages are empty.
- The SSRF guard checks every redirect hop and every resolved address, but DNS can change between the
  check and the connection; run the service without access to internal networks.
- The verify step proves a quote is on the page. It does not prove the plain-English requirement is a
  fair summary of the quote; read the quote.
- Verdicts are advice from a model, not a legal or commercial opinion.

## License

Apache License 2.0. See `LICENSE`.
