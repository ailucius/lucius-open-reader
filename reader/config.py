"""Runtime settings, read from environment variables only (no secrets in files)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Verified 2026-10-15 against https://docs.tokenfactory.nebius.com/api-reference/introduction.md
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Verified 2026-10-15 against https://tokenfactory.nebius.com/model-catalog.md
DEFAULT_READER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_DECIDER_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
# Measured 2026-10-15 on a 44-page tender: at the default or "medium" effort the reader thought for
# 8k-12k tokens on dense pages and returned nothing; "low" answered every chunk in about 8 seconds.
DEFAULT_READER_EFFORT = "low"
DEFAULT_DECIDER_EFFORT = "low"
EFFORTS = ("none", "low", "medium", "high")
# Verified 2026-10-15 against https://docs.tavily.com/documentation/api-reference/endpoint/search
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _effort(name: str, default: str) -> str | None:
    """low / medium / high / none; "default" leaves the model's own default."""
    raw = (os.environ.get(name) or default).strip().lower()
    if raw == "default":
        return None
    if raw not in EFFORTS:
        raise ValueError(f"{name} must be one of {', '.join(EFFORTS)} or default, not {raw!r}")
    return raw


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str | None = None
    nebius_base_url: str = DEFAULT_NEBIUS_BASE_URL
    reader_model: str = DEFAULT_READER_MODEL
    decider_model: str = DEFAULT_DECIDER_MODEL
    reader_effort: str | None = DEFAULT_READER_EFFORT
    decider_effort: str | None = DEFAULT_DECIDER_EFFORT
    max_tokens: int = 16384
    tavily_api_key: str | None = None
    fake_llm: bool = False
    concurrency: int = 4
    chunk_pages: int = 6
    max_pdf_mb: int = 40
    fetch_timeout_s: float = 60.0
    llm_timeout_s: float = 180.0
    max_pages: int = 500
    allow_private_urls: bool = False
    verify_threshold: float = 0.8

    @property
    def max_pdf_bytes(self) -> int:
        return self.max_pdf_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            nebius_api_key=os.environ.get("NEBIUS_API_KEY") or None,
            nebius_base_url=os.environ.get("NEBIUS_BASE_URL") or DEFAULT_NEBIUS_BASE_URL,
            reader_model=os.environ.get("READER_MODEL") or DEFAULT_READER_MODEL,
            decider_model=os.environ.get("DECIDER_MODEL") or DEFAULT_DECIDER_MODEL,
            reader_effort=_effort("READER_EFFORT", DEFAULT_READER_EFFORT),
            decider_effort=_effort("DECIDER_EFFORT", DEFAULT_DECIDER_EFFORT),
            max_tokens=max(1024, _int("READER_MAX_TOKENS", 16384)),
            tavily_api_key=os.environ.get("TAVILY_API_KEY") or None,
            fake_llm=_flag("READER_FAKE_LLM"),
            concurrency=max(1, _int("READER_CONCURRENCY", 4)),
            chunk_pages=max(1, _int("READER_CHUNK_PAGES", 6)),
            max_pdf_mb=max(1, _int("READER_MAX_PDF_MB", 40)),
            fetch_timeout_s=_float("READER_FETCH_TIMEOUT", 60.0),
            llm_timeout_s=_float("READER_LLM_TIMEOUT", 180.0),
            max_pages=max(1, _int("READER_MAX_PAGES", 500)),
            allow_private_urls=_flag("READER_ALLOW_PRIVATE_URLS"),
            verify_threshold=_float("READER_VERIFY_THRESHOLD", 0.8),
        )
