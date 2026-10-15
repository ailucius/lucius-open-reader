"""LLM access: NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API), plus an offline fake.

The Nebius client speaks the OpenAI chat-completions wire format directly over httpx:
POST {NEBIUS_BASE_URL}chat/completions with `Authorization: Bearer $NEBIUS_API_KEY` and
`response_format={"type": "json_schema", ...}` (falls back to `json_object` if the model refuses).

Nemotron 3 models think before they answer. At the default effort a dense tender chunk can use the
whole token budget on thinking and come back with an empty answer (finish_reason "length"), so the
client sends `reasoning_effort` and, if an answer still comes back empty, asks once more with
thinking switched off (`chat_template_kwargs={"enable_thinking": false}`).
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]: ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_content(content: str) -> dict[str, Any]:
    """Pull one JSON object out of a model reply (tolerates reasoning blocks and code fences)."""
    text = _THINK.sub("", content).strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"model reply is not JSON: {content[:200]!r}") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model reply is not JSON: {content[:200]!r}") from exc
    if not isinstance(value, dict):
        raise LLMError("model reply JSON is not an object")
    return value


class NebiusClient:
    """OpenAI-compatible chat client for Nebius Token Factory."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_s: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_tokens: int = 16384,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._timeout = timeout_s
        self._transport = transport
        self._max_tokens = max_tokens
        self.thinking_retries = 0  # answers that came back empty and were asked again without thinking

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            return await client.post(self._url, headers=self._headers, json=body)

    async def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        }
        _set_thinking(body, effort)
        resp = await self._post(body)
        if resp.status_code in (400, 422):
            # Some deployments reject json_schema; plain JSON mode is documented for all text models.
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] = (
                system + "\n\nReply with one JSON object matching this JSON Schema:\n" + json.dumps(schema)
            )
            resp = await self._post(body)
        content, finish = self._answer(resp)
        if not content.strip() and effort != "none":
            # thinking used the whole budget (finish_reason "length") or produced no answer: ask again
            # without thinking rather than lose the pages
            self.thinking_retries += 1
            _set_thinking(body, "none")
            content, finish = self._answer(await self._post(body))
        if not content.strip():
            raise LLMError(f"model reply is empty (finish_reason {finish!r})")
        return parse_json_content(content)

    @staticmethod
    def _answer(resp: httpx.Response) -> tuple[str, str | None]:
        if resp.status_code >= 400:
            raise LLMError(f"Nebius API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            choice = data["choices"][0]
            return choice["message"]["content"] or "", choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected API response: {str(data)[:300]}") from exc


def _set_thinking(body: dict[str, Any], effort: str | None) -> None:
    """`effort` is low, medium or high (sent as reasoning_effort), "none" (thinking off) or None (model default)."""
    body.pop("reasoning_effort", None)
    body.pop("chat_template_kwargs", None)
    if effort == "none":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort:
        body["reasoning_effort"] = effort


# --------------------------------------------------------------------------- fake

_PAGE_MARK = re.compile(r"^=== PAGE (\d+) ===$", re.MULTILINE)
_SENTENCE = re.compile(r"[^.;\n]+(?:[.;]|$)")
_SIGNAL = re.compile(r"\b(must|shall|required|mandatory|minimum|at least|submit|no later than)\b", re.I)
_CATEGORY_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("submission", ("submit", "deadline", "envelope", "portal", "no later than", "closing")),
    ("eligibility", ("registered", "registration", "experience", "turnover", "eligible", "certificate")),
    ("commercial", ("price", "payment", "guarantee", "bond", "insurance", "vat", "currency")),
    ("legal", ("law", "contract", "liability", "dispute", "declaration", "conflict of interest")),
    ("technical", ("pipe", "pump", "specification", "standard", "iso", "design", "works", "capacity")),
]


def _guess_category(sentence: str) -> str:
    low = sentence.lower()
    for name, words in _CATEGORY_WORDS:
        if any(w in low for w in words):
            return name
    return "other"


class FakeLLM:
    """Deterministic offline stand-in: rule-based extraction and a rule-based decision.

    Used by tests and by `READER_FAKE_LLM=1` demos. It never calls the network.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "schema_name": schema_name, "user": user, "effort": effort})
        if schema_name == "requirements":
            return self._extract(user)
        if schema_name == "scope":
            return {"title": "", "buyer": "", "buys": "The work described on the first pages.", "fit": "fits", "why": "Fake decider."}
        if schema_name == "assessments":
            return self._assess(user)
        if schema_name == "summary":
            return {
                "confidence": 0.5,
                "reasons": ["Fake decider: rule-based, for offline demos only."],
                "conditions": ["Confirm every mandatory requirement against your own documents."],
            }
        raise LLMError(f"fake LLM has no rule for {schema_name}")

    @staticmethod
    def _extract(user: str) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        marks = list(_PAGE_MARK.finditer(user))
        for i, mark in enumerate(marks):
            page = int(mark.group(1))
            end = marks[i + 1].start() if i + 1 < len(marks) else len(user)
            body = user[mark.end() : end]
            body = body.split("=== END OF PAGES ===")[0]
            for raw in _SENTENCE.findall(" ".join(body.split())):
                sentence = raw.strip()
                if len(sentence) < 12 or not _SIGNAL.search(sentence):
                    continue
                items.append(
                    {
                        "requirement": sentence.rstrip(".;"),
                        "quote": sentence,
                        "page": page,
                        "category": _guess_category(sentence),
                        "mandatory": bool(re.search(r"\b(must|shall|mandatory|required)\b", sentence, re.I)),
                    }
                )
        return {"requirements": items}

    @staticmethod
    def _assess(user: str) -> dict[str, Any]:
        rows = re.findall(r"^\[(R\d+)\] page \d+ \| (?:verified|UNVERIFIED) \| (\w+)", user, re.M)
        return {
            "assessments": [
                {
                    "id": rid,
                    "status": "not stated" if category in ("eligibility", "technical") else "routine",
                    "hard": category == "technical",
                    "why": "Fake decider: eligibility and technical items need confirming.",
                }
                for rid, category in rows
            ]
        }


def get_client(settings: Settings) -> LLMClient:
    if settings.fake_llm:
        return FakeLLM()
    if not settings.nebius_api_key:
        raise LLMError("NEBIUS_API_KEY is not set (or set READER_FAKE_LLM=1 for an offline demo)")
    return NebiusClient(
        settings.nebius_api_key, settings.nebius_base_url, settings.llm_timeout_s, max_tokens=settings.max_tokens
    )
