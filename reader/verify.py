"""Check every quote against the page text. Unverified items are kept and flagged, never dropped."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .models import Page, Requirement

NGRAM = 5
# A quote the model stitched together (it skipped a clause or a list lead-in) is still checkable:
# it passes as "joined" only if it is made, in page order, of at most MAX_PIECES verbatim passages
# of at least MIN_PIECE words each, with no word left over.
MIN_PIECE = 4
MAX_PIECES = 3
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


def joined_pieces(quote_words: list[str], page_words: list[str]) -> int:
    """How many verbatim passages of the page, in order, make up the whole quote (0: it cannot be built)."""
    if len(quote_words) < MIN_PIECE:
        return 0
    positions: dict[str, list[int]] = defaultdict(list)
    for i, word in enumerate(page_words):
        positions[word].append(i)
    i, after, pieces = 0, 0, 0
    while i < len(quote_words):
        best = 0
        best_end = after
        for start in positions.get(quote_words[i], ()):
            if start < after:
                continue
            k = 0
            while (
                i + k < len(quote_words)
                and start + k < len(page_words)
                and page_words[start + k] == quote_words[i + k]
            ):
                k += 1
            if k > best:
                best, best_end = k, start + k
        if best < MIN_PIECE:
            return 0
        pieces += 1
        if pieces > MAX_PIECES:
            return 0
        i += best
        after = best_end
    return pieces


class PageIndex:
    def __init__(self, pages: list[Page]) -> None:
        self.words: dict[int, list[str]] = {p.number: normalise(p.text) for p in pages}
        self.grams: dict[int, set[tuple[str, ...]]] = {n: _grams(w) for n, w in self.words.items()}
        self.transcribed: set[int] = {p.number for p in pages if p.transcribed}

    def score(self, quote_words: list[str], page: int) -> float:
        if page not in self.words:
            return 0.0
        return quote_score(quote_words, self.words[page], self.grams[page])

    def pieces(self, quote_words: list[str], page: int) -> int:
        if page not in self.words:
            return 0
        return joined_pieces(quote_words, self.words[page])

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
        used = [cited]  # the pages whose text the best match was found in
        if best < threshold:
            for offset in (-1, 1, -2, 2):
                candidate = cited + offset
                score = index.score(words, candidate)
                if score > best:
                    best_page, best, used = candidate, score, [candidate]
            if best < threshold:
                for first in (cited, cited - 1):
                    score = index.score_span(words, first)
                    if score > best:
                        # credit the page where the quote starts; the match used both pages
                        best_page, best, used = first, score, [first, first + 1]
        verified = best >= threshold
        joined = False
        if not verified:
            for candidate in (cited, cited - 1, cited + 1):
                if index.pieces(words, candidate) >= 2:
                    verified, joined, best_page, used = True, True, candidate, [candidate]
                    break
        page = best_page if verified else cited
        # "scanned": the check ran against the vision model's transcription of a page image, on the
        # page the quote is credited to or, for a quote across a page break, on either of the two
        checked = used if verified else [cited]
        out.append(
            req.model_copy(
                update={
                    "verified": verified,
                    "joined": joined,
                    "match_score": round(best, 3),
                    "page": page,
                    "cited_page": cited if page != cited else None,
                    "scanned": any(n in index.transcribed for n in checked),
                }
            )
        )
    return out
