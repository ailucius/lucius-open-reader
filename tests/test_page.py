"""The page itself: link checkers and preview bots (18 Oct 2026)."""

from fastapi.testclient import TestClient

from reader.app import create_app
from reader.config import Settings


def test_the_page_answers_head_and_carries_its_preview() -> None:
    with TestClient(create_app(Settings(fake_llm=True))) as client:
        assert client.head("/").status_code == 200
        page = client.get("/").text
        for tag in ('property="og:title"', 'property="og:description"', 'property="og:image"', 'name="twitter:card"'):
            assert tag in page
        image = client.get("/og.png")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert image.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert client.head("/og.png").status_code == 200
