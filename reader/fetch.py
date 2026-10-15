"""Download a public PDF safely: http/https only, size cap, timeout, no private addresses."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from .config import Settings

MAX_REDIRECTS = 5
USER_AGENT = "LuciusOpenReader/0.1 (reads public tender PDFs)"


class FetchError(ValueError):
    pass


def check_pdf_bytes(data: bytes) -> None:
    if b"%PDF-" not in data[:1024]:
        raise FetchError("the file is not a PDF (no %PDF- header)")


async def _assert_public_host(url: str, allow_private: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http and https URLs are allowed")
    host = parts.hostname
    if not host:
        raise FetchError("URL has no host")
    if allow_private:
        return
    try:
        ip = ipaddress.ip_address(host)
        addrs = [ip]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise FetchError(f"cannot resolve host {host}") from exc
        addrs = [ipaddress.ip_address(info[4][0]) for info in infos]
    for addr in addrs:
        if not addr.is_global:
            raise FetchError(f"host {host} resolves to a non-public address")


async def fetch_pdf(
    url: str,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bytes:
    """Return the PDF bytes at `url`, following up to 5 redirects (each hop re-checked)."""
    cap = settings.max_pdf_bytes
    timeout = httpx.Timeout(settings.fetch_timeout_s, connect=15.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        transport=transport,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"},
    ) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await _assert_public_host(current, settings.allow_private_urls)
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError("redirect without a location")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise FetchError(f"download failed: HTTP {resp.status_code}")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > cap:
                    raise FetchError(f"PDF is larger than {settings.max_pdf_mb} MB")
                buf = bytearray()
                async for part in resp.aiter_bytes():
                    buf.extend(part)
                    if len(buf) > cap:
                        raise FetchError(f"PDF is larger than {settings.max_pdf_mb} MB")
                data = bytes(buf)
                check_pdf_bytes(data)
                return data
        raise FetchError("too many redirects")
