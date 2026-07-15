"""Orchestrator: raw sources -> processed analytical tables.

Produces four artifacts under `data/processed/`:

    trends.parquet          Interest-over-time by market/keyword
    posts.parquet           Cleaned + language-tagged + sentiment-scored text
    market_keywords.parquet TF-IDF per-market vocabulary
    embeddings.parquet      2D coordinates + cluster labels

Prefers live data in `data/raw/` and falls back to the sample corpus in
`data/sample/`. When both Reddit and News are present they are merged
into one text corpus; each row is tagged with `data_source` (reddit /
news / sample) and `kind` (aesthetic / brand).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .cluster import cluster_frame
from .config import PROCESSED_DIR, RAW_DIR, SAMPLE_DIR, ensure_dirs, load_case_study
from .keywords import top_terms_by_market
from .preprocess import preprocess_text_frame
from .sentiment import score_frame


LOGGER = logging.getLogger(__name__)


def _load_trends() -> pd.DataFrame:
    raw = RAW_DIR / "trends.csv"
    sample = SAMPLE_DIR / "trends_sample.csv"
    path = raw if raw.exists() else sample
    LOGGER.info("Loading trends from %s", path)
    frame = pd.read_csv(path, parse_dates=["date"])
    if "kind" not in frame.columns:
        frame["kind"] = "aesthetic"
    return frame


def _load_reddit() -> pd.DataFrame | None:
    path = RAW_DIR / "reddit.csv"
    if not path.exists():
        return None
    LOGGER.info("Loading Reddit corpus from %s", path)
    frame = pd.read_csv(path)
    frame["text"] = frame["title"].fillna("") + " " + frame.get("selftext", "").fillna("")
    frame["data_source"] = "reddit"
    if "kind" not in frame.columns:
        frame["kind"] = "aesthetic"
    frame["sentiment_hint"] = None
    keep = [
        "aesthetic", "market", "language", "keyword", "kind",
        "post_id", "text", "score", "num_comments", "created_utc",
        "data_source", "sentiment_hint",
    ]
    return frame[[c for c in keep if c in frame.columns]]


def _load_news() -> pd.DataFrame | None:
    path = RAW_DIR / "news.csv"
    if not path.exists():
        return None
    LOGGER.info("Loading News corpus from %s", path)
    frame = pd.read_csv(path)
    title = frame["title"].fillna("").astype(str)
    summary = frame.get("summary", pd.Series([""] * len(frame))).fillna("").astype(str)
    # Google News summaries often include repeated titles + publisher chain.
    # Prefer summary if it is longer, else fall back to the title.
    frame["text"] = [
        s if len(s) > len(t) else t for t, s in zip(title, summary)
    ]
    frame["data_source"] = "news"
    if "kind" not in frame.columns:
        frame["kind"] = "aesthetic"
    frame["sentiment_hint"] = None
    frame["post_id"] = ("news:" + frame["link"].fillna("").astype(str)).str.slice(0, 60)
    frame["score"] = 0
    frame["num_comments"] = 0
    frame["created_utc"] = 0.0
    keep = [
        "aesthetic", "market", "language", "keyword", "kind",
        "post_id", "text", "score", "num_comments", "created_utc",
        "data_source", "sentiment_hint", "source", "link", "published",
    ]
    return frame[[c for c in keep if c in frame.columns]]


def _load_sample_text() -> pd.DataFrame:
    path = SAMPLE_DIR / "text_sample.csv"
    LOGGER.info("Loading sample multilingual corpus from %s", path)
    frame = pd.read_csv(path)
    frame["data_source"] = "sample"
    if "kind" not in frame.columns:
        frame["kind"] = "aesthetic"
    return frame


def _load_text(case) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    reddit = _load_reddit()
    if reddit is not None and not reddit.empty:
        frames.append(reddit)
    news = _load_news()
    if news is not None and not news.empty:
        frames.append(news)
    if frames:
        LOGGER.info(
            "Live corpus: %d rows (%s)",
            sum(len(f) for f in frames),
            ", ".join(f["data_source"].iloc[0] for f in frames),
        )
        return pd.concat(frames, ignore_index=True)
    return _load_sample_text()


def run() -> dict[str, Path]:
    ensure_dirs()
    case = load_case_study()

    trends = _load_trends()
    trends_out = PROCESSED_DIR / "trends.parquet"
    trends.to_parquet(trends_out, index=False)

    text = _load_text(case)
    if "text" not in text.columns:
        text["text"] = ""
    text["text"] = text["text"].fillna("").astype(str)
    text = preprocess_text_frame(text)
    text = score_frame(text)
    posts_out = PROCESSED_DIR / "posts.parquet"
    text.to_parquet(posts_out, index=False)

    # Per-aesthetic vocabulary so brands from one aesthetic do not leak into
    # another aesthetic's top-terms ranking.
    keyword_frames = []
    for aesthetic_key in text["aesthetic"].dropna().unique():
        subset = text[text["aesthetic"] == aesthetic_key]
        ranked = top_terms_by_market(subset, top_n=15)
        if not ranked.empty:
            ranked = ranked.assign(aesthetic=aesthetic_key)
            keyword_frames.append(ranked)
    keywords = (
        pd.concat(keyword_frames, ignore_index=True)
        if keyword_frames
        else pd.DataFrame(columns=["aesthetic", "market", "term", "count", "score"])
    )
    keywords_out = PROCESSED_DIR / "market_keywords.parquet"
    keywords.to_parquet(keywords_out, index=False)

    embeddings = cluster_frame(text)
    embeddings_cols = [
        "post_id", "aesthetic", "market", "language", "keyword", "kind",
        "data_source", "clean_text", "sentiment_label", "sentiment_score",
        "x", "y", "cluster",
    ]
    embeddings = embeddings[[c for c in embeddings_cols if c in embeddings.columns]]
    embeddings_out = PROCESSED_DIR / "embeddings.parquet"
    embeddings.to_parquet(embeddings_out, index=False)

    LOGGER.info(
        "Wrote %d trend rows, %d posts, %d keyword rows",
        len(trends),
        len(text),
        len(keywords),
    )
    return {
        "trends": trends_out,
        "posts": posts_out,
        "keywords": keywords_out,
        "embeddings": embeddings_out,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
