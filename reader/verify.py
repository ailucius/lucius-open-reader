"""Check every quote against the page text. Unverified items are kept and flagged, never dropped."""

from __future__ import annotations

import re
import unicodedata

from .models import Page, Requirement

NGRAM = 5
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalise(text: str) -> list[str]:
    """Lower-case words with punctuation and whitespace differences removed."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _NON_WORD.sub(" ", text.lower()).split()


def _grams(words: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def quote_score(quote_words: list[str], page_words: list[str], page_grams: set[tuple[str, ...]]) -> float:
    """Share of the quote's word 5-grams found on the page (exact phrase match for short quotes)."""
    if not quote_words:
        return 0.0
    if len(quote_words) < NGRAM:
        needle = " " + " ".join(quote_words) + " "
        return 1.0 if needle in " " + " ".join(page_words) + " " else 0.0
    grams = [tuple(quote_words[i : i + NGRAM]) for i in range(len(quote_words) - NGRAM + 1)]
    return sum(1 for g in grams if g in page_grams) / len(grams)


class PageIndex:
    def __init__(self, pages: list[Page]) -> None:
        self.words: dict[int, list[str]] = {p.number: normalise(p.text) for p in pages}
        self.grams: dict[int, set[tuple[str, ...]]] = {n: _grams(w) for n, w in self.words.items()}

    def score(self, quote_words: list[str], page: int) -> float:
        if page not in self.words:
            return 0.0
        return quote_score(quote_words, self.words[page], self.grams[page])

    def score_span(self, quote_words: list[str], first: int) -> float:
        """Score a quote that may run across the break between `first` and `first + 1`."""
        if first not in self.words or first + 1 not in self.words:
            return 0.0
        joined = self.words[first] + self.words[first + 1]
        return quote_score(quote_words, joined, _grams(joined))


def verify_requirements(reqs: list[Requirement], pages: list[Page], threshold: float = 0.8) -> list[Requirement]:
    index = PageIndex(pages)
    out: list[Requirement] = []
    for req in reqs:
        words = normalise(req.quote)
        cited = req.page
        best_page, best = cited, index.score(words, cited)
        if best < threshold:
            for offset in (-1, 1, -2, 2):
                candidate = cited + offset
                score = index.score(words, candidate)
                if score > best:
                    best_page, best = candidate, score
            if best < threshold:
                for first in (cited, cited - 1):
                    score = index.score_span(words, first)
                    if score > best:
                        # credit the page where the quote starts
                        best_page, best = first, score
        verified = best >= threshold
        page = best_page if verified else cited
        out.append(
            req.model_copy(
                update={
                    "verified": verified,
                    "match_score": round(best, 3),
                    "page": page,
                    "cited_page": cited if page != cited else None,
                }
            )
        )
    return out
