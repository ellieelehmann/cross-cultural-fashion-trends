"""Text cleaning and language tagging for multilingual fashion text."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")
WHITESPACE_PATTERN = re.compile(r"\s+")
BOILERPLATE_TRAIL = re.compile(r"\s*-\s*[A-Za-z0-9\.\s]{2,40}$")  # trailing publisher, e.g. " - Vogue"


def clean_text(raw: str) -> str:
    """Lightweight, language-agnostic cleaning suitable for transformers."""
    if not isinstance(raw, str):
        return ""
    text = HTML_TAG_PATTERN.sub(" ", raw)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    # Remove Google News RSS duplicate: "Title - Publisher Title Publisher"
    # by collapsing repeated adjacent substrings of >=20 chars.
    if len(text) > 40:
        half = len(text) // 2
        for split in (half, half + 1, half - 1):
            left = text[:split].strip()
            right = text[split:].strip()
            if len(left) >= 20 and left in right:
                text = left
                break
    return text


@lru_cache(maxsize=1)
def _load_langdetect():
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42
        return detect
    except Exception:  # noqa: BLE001
        return None


def detect_language(text: str, fallback: str) -> str:
    """Detect language; fall back to the configured language on failure."""
    detect = _load_langdetect()
    if not detect or not text or len(text) < 4:
        return fallback
    try:
        return detect(text)
    except Exception:  # noqa: BLE001
        return fallback


def preprocess_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `clean_text` and `detected_language` columns.

    Expected columns: `text` and `language` (the configured market language).
    """
    if frame.empty:
        return frame.assign(clean_text="", detected_language="")

    out = frame.copy()
    combined = out.get("title", "").astype(str) + " " + out["text"].astype(str) \
        if "title" in out.columns else out["text"].astype(str)
    out["clean_text"] = combined.map(clean_text)
    out["detected_language"] = [
        detect_language(txt, fallback=lang)
        for txt, lang in zip(out["clean_text"], out["language"])
    ]
    out["char_count"] = out["clean_text"].str.len()
    return out
