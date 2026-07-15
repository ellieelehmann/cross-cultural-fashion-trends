"""Per-market keyword and phrase extraction.

Answers the question: "what words tend to co-occur with this
aesthetic in each culture?" Uses TF-IDF against a within-corpus
baseline so that terms which are common in every market drop out
and market-specific vocabulary rises.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd


# Simple, permissive tokenizer that treats CJK characters as single-char
# tokens. This is intentional: it lets us surface Korean/Japanese words
# without depending on a language-specific tokenizer.
TOKEN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+|[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+",
    re.UNICODE,
)

STOPWORDS = {
    # English
    "the", "and", "a", "to", "of", "in", "is", "for", "it", "on", "that",
    "this", "with", "as", "at", "be", "have", "has", "was", "are", "you",
    "your", "my", "just", "so", "not", "but", "or", "an", "if", "all",
    "like", "one", "im", "its", "get", "really", "even", "some",
    # French
    "le", "la", "les", "de", "des", "un", "une", "et", "est", "pas",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "ce", "cette", "ces", "qui", "que", "dans", "en", "avec", "sur",
    "mais", "au", "aux", "du", "pour", "par", "plus", "moins", "être",
    "avoir", "trop", "bien", "aussi", "fait", "faire", "cest", "cétait",
    # Standalone tokens we don't care about
    "https", "http", "www",
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def top_terms_by_market(
    frame: pd.DataFrame,
    top_n: int = 15,
    min_market_freq: int = 3,
) -> pd.DataFrame:
    """TF-IDF-style ranking of terms per market.

    Frame must include `market` and `clean_text` columns.
    """
    if frame.empty:
        return pd.DataFrame(columns=["market", "term", "score", "count"])

    market_counts: dict[str, Counter] = {}
    for market, group in frame.groupby("market"):
        counter: Counter = Counter()
        for text in group["clean_text"]:
            counter.update(_tokenize(text))
        market_counts[market] = counter

    global_counter: Counter = Counter()
    for counter in market_counts.values():
        for term, count in counter.items():
            global_counter[term] += count

    total_markets = len(market_counts)
    rows: list[dict] = []
    for market, counter in market_counts.items():
        market_total = sum(counter.values()) or 1
        for term, count in counter.items():
            if count < min_market_freq:
                continue
            markets_with_term = sum(1 for c in market_counts.values() if c[term] > 0)
            idf = (total_markets + 1) / (markets_with_term + 1)
            tfidf = (count / market_total) * idf
            rows.append(
                {"market": market, "term": term, "count": count, "score": tfidf}
            )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(["market", "score"], ascending=[True, False])
    return (
        ranked.groupby("market", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
