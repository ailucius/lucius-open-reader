"""Bid / no-bid call with the decider model (Nemotron 3 Ultra by default), grounded in the list only.

Three steps, so no single answer has to be long:
1. scope: from the first pages, what the tender buys and whether that is the company's line of work;
2. assess: every mandatory requirement judged against the profile (fails / not stated / met /
   routine), in batches of BATCH run side by side; ids the model skipped are asked again once;
3. summary: a few reasons and the conditions to meet, from the judgments.

The model never picks the verdict. It follows from the judgments: work outside the company's line
-> no bid; any "fails" -> no bid; any "not stated" -> bid with conditions; otherwise bid.
Measured 15 Oct 2026 on a 44-page tender: one free-form call gave bid 0.92, no bid 0.95 and bid
with conditions 0.75 on the same list, and one call judging all 78 items at once skipped two thirds.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .llm import LLMClient
from .models import Assessment, Decision, DecidingRequirement, Page, Requirement, Scope
from .verify import normalise

BATCH = 20
MAX_QUOTE_CHARS = 300
MAX_DECIDING = 8
SCOPE_PAGES = 3
SCOPE_CHARS_PER_PAGE = 4000
STATUSES: tuple[str, ...] = ("fails", "not stated", "met", "routine")
FITS: tuple[str, ...] = ("fits", "partly", "outside")
# which unmet requirements to show first: the ones that are hardest to put right before the deadline
CATEGORY_ORDER = {"eligibility": 0, "technical": 1, "commercial": 2, "legal": 3, "submission": 4, "other": 5}

SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "buyer", "buys", "fit", "why"],
    "properties": {
        "title": {"type": "string"},
        "buyer": {"type": "string"},
        "buys": {"type": "string"},
        "fit": {"type": "string", "enum": list(FITS)},
        "why": {"type": "string"},
    },
}

SCOPE_SYSTEM = """You read the first pages of a public tender for a company deciding whether to bid.
Return:
- title: the tender's title, copied from the pages ("" if none is given).
- buyer: the buying organisation's name, copied from the pages ("" if none is given).
- buys: one plain sentence on what the tender buys (goods, works or services, and where).
- fit: "fits" if that is the company's line of work, "partly" if only some of it is,
  "outside" if it is not the company's line of work.
- why: one plain sentence on the fit, about THIS company.
Use only the pages and the profile. Never invent names, numbers or dates. Return JSON only."""

ASSESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assessments"],
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "status", "hard", "why"],
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "hard": {"type": "boolean"},
                    "why": {"type": "string"},
                },
            },
        }
    },
}

ASSESS_SYSTEM = """You check a company against some of a public tender's mandatory requirements.
Return exactly one assessment for EVERY requirement id listed, in the order listed:
- "fails": the profile shows the company cannot meet it (wrong line of work, wrong grade or
  class, wrong country or region, too small, a licence it cannot get in time).
- "not stated": it needs a certificate, registration, document, capacity or experience the
  profile does not mention. The company must confirm or obtain it.
- "met": the profile says it is met.
- "routine": a form, signature, declaration, price format or submission rule any eligible
  bidder completes while preparing the bid.
hard: true if meeting it takes weeks, money or a third party (an approval, a certification, a
  guarantee, a bank rating, proven experience, a site visit); false for papers a registered
  company normally has or can fill in.
why: one short plain sentence about THIS company ("" is fine for routine).
Use only the requirements and the profile. Never invent facts. Return JSON only."""

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confidence", "reasons", "conditions"],
    "properties": {
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {"type": "string"}},
    },
}

SUMMARY_SYSTEM = """You explain a bid / no-bid call on a public tender to a company.
You get the company profile, the tender's scope, the call, and the requirements the company
fails or must still confirm, each with an id like [R12].
Return:
- reasons: at most 4 short plain sentences on why the call is what it is; cite at most 3 ids in
  one sentence.
- conditions: at most 8 things that must be true or done before bidding, most important first,
  each citing its ids; group similar documents into one item. Empty for a no-bid call.
- confidence: 0 to 1, how sure you are that the call is right for this company.
Use only what you are given. Never invent requirements, numbers or dates. Return JSON only."""


def _profile_block(profile: str) -> list[str]:
    return ["Company profile:", "<<<", profile.strip(), ">>>", ""]


def build_scope_prompt(profile: str, pages: list[Page]) -> str:
    lines = _profile_block(profile)
    for page in pages[:SCOPE_PAGES]:
        lines.append(f"=== PAGE {page.number} ===")
        lines.append(page.text[:SCOPE_CHARS_PER_PAGE].strip())
    lines.append("=== END OF PAGES ===")
    return "\n".join(lines)


def build_assess_prompt(profile: str, scope: Scope | None, batch: list[Requirement]) -> str:
    lines = _profile_block(profile)
    if scope and scope.buys:
        lines += [f"The tender buys: {scope.buys}", ""]
    lines.append(f"Requirements to assess ({len(batch)}): {', '.join(r.id for r in batch)}")
    for r in batch:
        flag = "verified" if r.verified else "UNVERIFIED"
        quote = r.quote if len(r.quote) <= MAX_QUOTE_CHARS else r.quote[:MAX_QUOTE_CHARS] + "..."
        lines.append(f"[{r.id}] page {r.page} | {flag} | {r.category}")
        lines.append(f"  Requirement: {r.requirement}")
        lines.append(f'  Quote: "{quote}"')
    return "\n".join(lines)


def build_summary_prompt(profile: str, scope: Scope | None, verdict: str, items: list[DecidingRequirement]) -> str:
    lines = _profile_block(profile)
    if scope:
        lines += [f"The tender buys: {scope.buys}", f"Fit with the company's line of work: {scope.fit} ({scope.why})", ""]
    lines += [f"Call: {verdict}", "", f"Requirements the company fails or must confirm ({len(items)}):"]
    for x in items:
        lines.append(f"[{x.id}] {x.status}{' | hard to obtain' if x.hard else ''} | page {x.page} | {x.requirement} | {x.why}")
    if not items:
        lines.append("(none)")
    return "\n".join(lines)


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _choice(value: Any, allowed: tuple[str, ...], aliases: dict[str, str]) -> str | None:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    text = aliases.get(text, text)
    return text if text in allowed else None


_STATUS_ALIASES = {"not met": "fails", "fail": "fails", "failed": "fails", "unknown": "not stated", "unclear": "not stated"}
_FIT_ALIASES = {"fit": "fits", "partial": "partly", "partially": "partly", "out of scope": "outside", "no": "outside"}


def parse_scope(raw: dict[str, Any], pages: list[Page]) -> Scope:
    """Title and buyer are kept only if they appear on the pages read (no invented names)."""
    seen = " " + " ".join(normalise(" ".join(p.text for p in pages[:SCOPE_PAGES]))) + " "

    def on_pages(value: Any) -> str | None:
        text = str(value or "").strip()
        words = normalise(text)
        return text if words and " " + " ".join(words) + " " in seen else None

    return Scope(
        title=on_pages(raw.get("title")),
        buyer=on_pages(raw.get("buyer")),
        buys=str(raw.get("buys") or "").strip(),
        fit=_choice(raw.get("fit"), FITS, _FIT_ALIASES) or "partly",  # type: ignore[arg-type]
        why=str(raw.get("why") or "").strip(),
    )


def parse_assessments(raw: dict[str, Any], batch: list[Requirement]) -> tuple[list[Assessment], list[str]]:
    """Returns (assessments for ids in this batch, ids the model cited that are not in the batch)."""
    wanted = {r.id for r in batch}
    out: list[Assessment] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for item in raw.get("assessments") or []:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or "").strip().strip("[]")
        status = _choice(item.get("status"), STATUSES, _STATUS_ALIASES)
        if status is None or rid in seen:
            continue
        if rid not in wanted:
            rejected.append(rid)
            continue
        seen.add(rid)
        hard = item.get("hard")
        out.append(
            Assessment(
                id=rid,
                status=status,  # type: ignore[arg-type]
                hard=hard is True or str(hard).strip().lower() == "true",
                why=str(item.get("why") or "").strip(),
            )
        )
    return out, rejected


def verdict_from(scope: Scope | None, assessments: list[Assessment]) -> str:
    if scope is not None and scope.fit == "outside":
        return "no bid"
    if any(a.status == "fails" for a in assessments):
        return "no bid"
    if any(a.status == "not stated" for a in assessments):
        return "bid with conditions"
    return "bid"


def deciding_from(assessments: list[Assessment], reqs: list[Requirement]) -> list[DecidingRequirement]:
    by_id = {r.id: r for r in reqs}
    unmet = [a for a in assessments if a.status in ("fails", "not stated")]

    def rank(a: Assessment) -> tuple[int, int, int, int]:
        r = by_id[a.id]
        return (0 if a.status == "fails" else 1, 0 if a.hard else 1, CATEGORY_ORDER.get(r.category, 9), r.page)

    return [
        DecidingRequirement(
            id=a.id,
            requirement=by_id[a.id].requirement,
            page=by_id[a.id].page,
            verified=by_id[a.id].verified,
            status=a.status,
            hard=a.hard,
            why=a.why,
        )
        for a in sorted(unmet, key=rank)
    ]


async def decide(
    profile: str,
    reqs: list[Requirement],
    llm: LLMClient,
    model: str,
    effort: str | None = None,
    pages: list[Page] | None = None,
    concurrency: int = 4,
) -> tuple[Decision, int]:
    """Return (decision, number of mandatory requirements the model gave no judgment for)."""
    sem = asyncio.Semaphore(concurrency)

    async def ask(system: str, user: str, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await llm.chat_json(model=model, system=system, user=user, schema_name=name, schema=schema, effort=effort)

    scope: Scope | None = None
    if pages:
        try:
            scope = parse_scope(await ask(SCOPE_SYSTEM, build_scope_prompt(profile, pages), "scope", SCOPE_SCHEMA), pages)
        except Exception:  # the scope step is a help, not a requirement: the assessments still decide
            scope = None

    mandatory = [r for r in reqs if r.mandatory]
    batches = [mandatory[i : i + BATCH] for i in range(0, len(mandatory), BATCH)]
    assessments: list[Assessment] = []
    rejected: list[str] = []

    async def run_batch(batch: list[Requirement]) -> None:
        todo = batch
        for _attempt in range(2):  # ask once more for the ids the model skipped
            try:
                raw = await ask(ASSESS_SYSTEM, build_assess_prompt(profile, scope, todo), "assessments", ASSESS_SCHEMA)
            except Exception:
                continue
            got, bad = parse_assessments(raw, todo)
            assessments.extend(got)
            rejected.extend(bad)
            done = {a.id for a in got}
            todo = [r for r in todo if r.id not in done]
            if not todo:
                return

    await asyncio.gather(*(run_batch(b) for b in batches))
    order = {r.id: i for i, r in enumerate(reqs)}
    assessments.sort(key=lambda a: order[a.id])
    assessed = {a.id for a in assessments}
    unassessed = [r.id for r in mandatory if r.id not in assessed]

    verdict = verdict_from(scope, assessments)
    unmet = deciding_from(assessments, reqs)
    summary: dict[str, Any] = {}
    try:
        summary = await ask(SUMMARY_SYSTEM, build_summary_prompt(profile, scope, verdict, unmet), "summary", SUMMARY_SCHEMA)
    except Exception:
        summary = {}
    decision = Decision(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=_confidence(summary.get("confidence")),
        scope=scope,
        deciding=unmet[:MAX_DECIDING],
        unmet_total=len(unmet),
        reasons=[str(x).strip() for x in summary.get("reasons") or [] if str(x).strip()],
        conditions=[] if verdict == "no bid" else [str(x).strip() for x in summary.get("conditions") or [] if str(x).strip()],
        assessments=assessments,
        counts={s: sum(1 for a in assessments if a.status == s) for s in STATUSES},
        unassessed=unassessed,
        rejected_ids=sorted(set(rejected)),
        model=model,
    )
    return decision, len(unassessed)
