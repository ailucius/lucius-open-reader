"""CLI: python -m reader <pdf-url-or-path> --profile "..." [--fake]  (prints JSON to stdout)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from .config import Settings
from .llm import get_client
from .pipeline import Event, run_reader


async def _progress(event: Event) -> None:
    print(json.dumps(event), file=sys.stderr)


async def _main(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.fake:
        settings = dataclasses.replace(settings, fake_llm=True)
    llm = get_client(settings)
    source: str = args.source
    pdf: bytes | None = None
    url: str | None = None
    if source.startswith(("http://", "https://")):
        url = source
    else:
        pdf = Path(source).read_bytes()
    result, _ = await run_reader(
        settings=settings,
        llm=llm,
        profile=args.profile,
        url=url,
        pdf=pdf,
        source_name=Path(source).name if pdf is not None else None,
        emit=_progress if args.progress else _silent,
    )
    print(result.model_dump_json(indent=2))
    if args.progress and getattr(llm, "usage", None):
        await _progress({"type": "usage", "tokens": llm.usage})  # tokens billed per model
    return 0


async def _silent(_: Event) -> None:
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reader", description="Read a public tender before you bid.")
    parser.add_argument("source", help="public PDF URL (http/https) or a local PDF path")
    parser.add_argument("--profile", default="", help="short company profile for the bid / no-bid call")
    parser.add_argument("--fake", action="store_true", help="use the offline fake LLM (same as READER_FAKE_LLM=1)")
    parser.add_argument("--progress", action="store_true", help="print progress events to stderr")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_main(args))
    except Exception as exc:  # report cleanly for CLI users
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
