"""Bid / no-bid call with the decider model (Nemotron 3 Ultra by default), grounded in the list only."""

from __future__ import annotations

from typing import Any

from .llm import LLMClient, LLMError
from .models import VERDICTS, Decision, DecidingRequirement, Requirement

MAX_ROWS = 400
MAX_QUOTE_CHARS = 300

DECIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence", "deciding", "reasons", "conditions"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "confidence": {"type": "number"},
        "deciding": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "why"],
                "properties": {"id": {"type": "string"}, "why": {"type": "string"}},
            },
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {"type": "string"}},
    },
}

DECIDE_SYSTEM = """You advise a company on whether to bid for a public tender.
You get the company profile and the tender's requirements, each with an id like [R12].

Rules:
- Use ONLY the listed requirements. Never invent a requirement, a number or a date.
- Refer to requirements by their id. "deciding" lists the ids that drive the call (at most 8).
- A requirement marked UNVERIFIED may be misquoted; if one decides the call, say so in reasons.
- If the profile does not say whether a mandatory requirement is met, make it a condition,
  not a failure.

verdict:
- "bid": the profile meets every mandatory requirement that matters.
- "bid with conditions": a bid is sensible once the listed conditions are met or confirmed.
- "no bid": the profile clearly fails a mandatory requirement, or the work is outside it.
confidence: 0 to 1. reasons: short plain sentences. conditions: what must be true to bid.
Return JSON only."""


def build_decide_prompt(profile: str, reqs: list[Requirement]) -> tuple[str, int]:
    ordered = sorted(reqs, key=lambda r: (not r.mandatory, r.page))[:MAX_ROWS]
    lines = ["Company profile:", "<<<", profile.strip(), ">>>", "", f"Requirements ({len(ordered)}):"]
    for r in sorted(ordered, key=lambda r: r.page):
        flag = "verified" if r.verified else "UNVERIFIED"
        need = "mandatory" if r.mandatory else "optional"
        quote = r.quote if len(r.quote) <= MAX_QUOTE_CHARS else r.quote[:MAX_QUOTE_CHARS] + "..."
        lines.append(f"[{r.id}] page {r.page} | {need} | {flag} | {r.category}")
        lines.append(f"  Requirement: {r.requirement}")
        lines.append(f'  Quote: "{quote}"')
    return "\n".join(lines), len(reqs) - len(ordered)


def _verdict(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    aliases = {"nobid": "no bid", "no go": "no bid", "bid with condition": "bid with conditions", "go": "bid"}
    text = aliases.get(text, text)
    if text not in VERDICTS:
        raise LLMError(f"decider returned an unknown verdict: {value!r}")
    return text


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_decision(raw: dict[str, Any], reqs: list[Requirement], model: str) -> Decision:
    by_id = {r.id: r for r in reqs}
    deciding: list[DecidingRequirement] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for item in raw.get("deciding") or []:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or "").strip().strip("[]")
        if rid in seen:
            continue
        seen.add(rid)
        req = by_id.get(rid)
        if req is None:
            rejected.append(rid)  # the model pointed at something not in the list
            continue
        deciding.append(
            DecidingRequirement(
                id=rid,
                requirement=req.requirement,
                page=req.page,
                verified=req.verified,
                why=str(item.get("why") or "").strip(),
            )
        )
    return Decision(
        verdict=_verdict(raw.get("verdict")),  # type: ignore[arg-type]
        confidence=_confidence(raw.get("confidence")),
        deciding=deciding,
        reasons=[str(x).strip() for x in raw.get("reasons") or [] if str(x).strip()],
        conditions=[str(x).strip() for x in raw.get("conditions") or [] if str(x).strip()],
        rejected_ids=rejected,
        model=model,
    )


async def decide(profile: str, reqs: list[Requirement], llm: LLMClient, model: str) -> tuple[Decision, int]:
    """Return (decision, number of requirements left out of the prompt)."""
    user, left_out = build_decide_prompt(profile, reqs)
    raw = await llm.chat_json(
        model=model, system=DECIDE_SYSTEM, user=user, schema_name="decision", schema=DECIDE_SCHEMA
    )
    return parse_decision(raw, reqs, model), left_out
