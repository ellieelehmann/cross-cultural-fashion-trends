"""Multilingual sentiment classification.

Uses `tabularisai/multilingual-sentiment-analysis`, a distilbert
fine-tune covering 23 languages including English, French, Korean,
and Japanese, mapping each text to one of five sentiment classes.

If Hugging Face `transformers` is not installed or model download
fails, we fall back to a lightweight lexicon+`sentiment_hint`-aware
heuristic so that the dashboard still renders reasonable numbers.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

LABEL_TO_SCORE = {
    "Very Negative": -1.0,
    "Negative": -0.5,
    "Neutral": 0.0,
    "Positive": 0.5,
    "Very Positive": 1.0,
    "negative": -0.75,
    "neutral": 0.0,
    "positive": 0.75,
}


@lru_cache(maxsize=1)
def _load_pipeline():
    """Lazy-load the transformers pipeline once."""
    try:
        from transformers import pipeline

        return pipeline(
            task="text-classification",
            model=MODEL_NAME,
            top_k=None,
            truncation=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Falling back to heuristic sentiment: %s", exc)
        return None


def _heuristic_label(text: str, hint: str | None = None) -> tuple[str, float]:
    """Deterministic fallback used when the transformer is unavailable."""
    if hint in {"positive", "negative", "neutral"}:
        score = LABEL_TO_SCORE[hint]
        return hint.capitalize(), score

    lowered = text.lower()
    positive_terms = (
        "love",
        "great",
        "best",
        "beautiful",
        "elegant",
        "chic",
        "adore",
        "예쁘",
        "좋",
        "好き",
        "최고",
        "elegante",
    )
    negative_terms = (
        "hate",
        "bad",
        "worst",
        "boring",
        "cringe",
        "exhausting",
        "지겨",
        "아쉬움",
        "重い",
        "prévisible",
        "performative",
    )
    pos = sum(term in lowered for term in positive_terms)
    neg = sum(term in lowered for term in negative_terms)
    if pos > neg:
        return "Positive", 0.5
    if neg > pos:
        return "Negative", -0.5
    return "Neutral", 0.0


def classify_texts(
    texts: Iterable[str],
    hints: Iterable[str | None] | None = None,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Return (label, score) pairs, one per input text."""
    texts = list(texts)
    hints = list(hints) if hints is not None else [None] * len(texts)

    pipe = _load_pipeline()
    if pipe is None:
        return [_heuristic_label(t, h) for t, h in zip(texts, hints)]

    results: list[tuple[str, float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            outputs = pipe(batch)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Transformer batch failed at %d: %s", start, exc)
            outputs = [None] * len(batch)

        for text, out, hint in zip(batch, outputs, hints[start : start + batch_size]):
            if not out:
                results.append(_heuristic_label(text, hint))
                continue
            best = max(out, key=lambda item: item["score"])
            label = best["label"]
            score = LABEL_TO_SCORE.get(label, 0.0)
            results.append((label, score))
    return results


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `sentiment_label` and `sentiment_score` columns."""
    if frame.empty:
        return frame.assign(sentiment_label="Neutral", sentiment_score=0.0)

    hints = frame["sentiment_hint"].tolist() if "sentiment_hint" in frame.columns else None
    pairs = classify_texts(frame["clean_text"].tolist(), hints=hints)
    out = frame.copy()
    out["sentiment_label"] = [p[0] for p in pairs]
    out["sentiment_score"] = [p[1] for p in pairs]
    return out
