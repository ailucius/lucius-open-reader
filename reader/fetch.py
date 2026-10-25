"""Download a public PDF safely: http/https only, size cap, timeout, no private addresses."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from .config import Settings

MAX_REDIRECTS = 5
USER_AGENT = "LuciusOpenReader/0.1 (reads public tender PDFs)"


class FetchError(ValueError):
    pass


def check_pdf_bytes(data: bytes) -> None:
    if b"%PDF-" not in data[:1024]:
        raise FetchError("the file is not a PDF (no %PDF- header)")


async def _public_addresses(host: str, port: int) -> list[str]:
    """The addresses `host` resolves to, or FetchError if any of them is not public."""
    try:
        ip = ipaddress.ip_address(host)
        addrs = [ip]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port)
        except socket.gaierror as exc:
            raise FetchError(f"cannot resolve host {host}") from exc
        addrs = [ipaddress.ip_address(info[4][0]) for info in infos]
    for addr in addrs:
        if not addr.is_global:
            raise FetchError(f"host {host} resolves to a non-public address")
    return list(dict.fromkeys(str(addr) for addr in addrs))


async def _assert_public_host(url: str, allow_private: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http and https URLs are allowed")
    host = parts.hostname
    if not host:
        raise FetchError("URL has no host")
    if allow_private:
        return
    await _public_addresses(host, parts.port or (443 if parts.scheme == "https" else 80))


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """Connects only to an address it has just checked is public.

    `_assert_public_host` checks each URL before its request, but the client resolves the host again
    to connect, and a DNS server can answer a public address to the check and a private one to the
    connection (DNS rebinding). Here the address that passed the check is the one connected to; TLS
    still verifies the certificate against the host name in the URL.
    """

    def __init__(self) -> None:
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        error: Exception = httpcore.ConnectError(f"cannot connect to {host}")
        for addr in await _public_addresses(host, port):
            try:
                return await self._inner.connect_tcp(
                    addr, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                error = exc
        raise error

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PublicTransport(httpx.AsyncHTTPTransport):
    """httpx's transport with its connection pool connecting through `_PublicOnlyBackend`."""

    def __init__(self) -> None:
        super().__init__()
        # httpx 0.28 takes no network backend, so the pool is rebuilt with one (httpx is pinned).
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(), network_backend=_PublicOnlyBackend()
        )


def _transport(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncBaseTransport | None:
    if transport is not None or settings.allow_private_urls:
        return transport
    return _PublicTransport()


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
        transport=_transport(settings, transport),
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


async def probe_document(
    url: str,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bool, str, int | None]:
    """Read the first 2 KB at `url` (same address checks as fetch_pdf).

    Returns (is a PDF, final URL after redirects, total size in bytes when the server says).
    A link's name says little: "/download/3001" can be a PDF and "tender.pdf" an HTML error page.
    """
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        transport=_transport(settings, transport),
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*", "Range": "bytes=0-2047"},
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
                    raise FetchError(f"HTTP {resp.status_code}")
                head = bytearray()
                async for part in resp.aiter_bytes():
                    head.extend(part)
                    if len(head) >= 2048:
                        break
                size: int | None = None
                content_range = resp.headers.get("content-range", "")
                if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                    size = int(content_range.rsplit("/", 1)[1])
                elif resp.status_code == 200 and (resp.headers.get("content-length") or "").isdigit():
                    size = int(resp.headers["content-length"])
                return b"%PDF-" in bytes(head[:1024]), current, size
        raise FetchError("too many redirects")
