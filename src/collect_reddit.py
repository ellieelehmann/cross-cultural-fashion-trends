"""Collect Reddit search results as a multilingual text corpus.

Reddit is the only major platform whose Terms of Service explicitly permit
scraping of public content and which exposes public JSON feeds. That makes
it the safest primary text source for a portfolio project. This module
uses the public JSON endpoint so no API key is required.

Iterates over both aesthetic vibe terms (per market) and brand aliases
(per market's language) so each row is tagged as either 'aesthetic' or
'brand'.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import RAW_DIR, CaseStudy, ensure_dirs, load_case_study


LOGGER = logging.getLogger(__name__)

USER_AGENT = "cross-cultural-fashion-trends/0.1 (portfolio project)"
SEARCH_URL = "https://www.reddit.com/search.json"


@dataclass
class RedditPost:
    aesthetic: str
    market: str
    language: str
    keyword: str
    kind: str  # 'aesthetic' | 'brand'
    subreddit: str
    post_id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str


def _search(keyword: str, subreddit: str | None, limit: int) -> Iterable[dict]:
    params = {"q": keyword, "limit": limit, "sort": "relevance", "type": "link"}
    if subreddit:
        params["restrict_sr"] = "on"
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
    else:
        url = SEARCH_URL
    response = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20
    )
    response.raise_for_status()
    payload = response.json()
    for child in payload.get("data", {}).get("children", []):
        yield child.get("data", {})


def collect(
    case: CaseStudy | None = None,
    sleep_between: float = 3.0,
    global_only: bool = True,
    stop_on_repeated_429: int = 5,
) -> Path:
    """Fetch Reddit posts for every keyword and write `data/raw/reddit.csv`.

    Checkpoints every 20 queries so a mid-run 429 does not lose the buffer.
    `global_only=True` skips per-subreddit search which massively reduces
    request volume and avoids Reddit's per-endpoint throttling.
    """
    case = case or load_case_study()
    ensure_dirs()

    reddit_cfg = case.text_sources.get("reddit", {})
    if not reddit_cfg.get("enabled", False):
        raise RuntimeError("Reddit source is disabled in config.")

    if global_only:
        subreddits: list[str | None] = [None]
    else:
        subreddits = list(reddit_cfg.get("subreddits", [])) or [None]

    limit = int(reddit_cfg.get("post_limit_per_keyword", 25))
    queries = list(case.all_queries())
    LOGGER.info(
        "Reddit: %d queries x %d subreddits = %d requests",
        len(queries),
        len(subreddits),
        len(queries) * len(subreddits),
    )

    out = RAW_DIR / "reddit.csv"
    collected: list[RedditPost] = []
    consecutive_429 = 0
    processed = 0
    for aesthetic_key, market, language, keyword, kind in queries:
        for subreddit in subreddits:
            try:
                LOGGER.info("[%s] '%s' in r/%s", market, keyword, subreddit or "all")
                for post in _search(keyword, subreddit, limit):
                    collected.append(
                        RedditPost(
                            aesthetic=aesthetic_key,
                            market=market,
                            language=language,
                            keyword=keyword,
                            kind=kind,
                            subreddit=post.get("subreddit", subreddit or ""),
                            post_id=post.get("id", ""),
                            title=post.get("title", ""),
                            selftext=post.get("selftext", ""),
                            score=int(post.get("score", 0)),
                            num_comments=int(post.get("num_comments", 0)),
                            created_utc=float(post.get("created_utc", 0.0)),
                            permalink=post.get("permalink", ""),
                        )
                    )
                consecutive_429 = 0
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                LOGGER.warning("Skip %s / r/%s: %s", keyword, subreddit, message)
                if "429" in message:
                    consecutive_429 += 1
                    time.sleep(min(60, 5 * consecutive_429))
                    if consecutive_429 >= stop_on_repeated_429:
                        LOGGER.error(
                            "Bailing after %d consecutive 429s. Saving buffer.",
                            consecutive_429,
                        )
                        _flush(collected, out)
                        return out
            time.sleep(sleep_between)
            processed += 1
            if processed % 20 == 0:
                _flush(collected, out)

    _flush(collected, out)
    return out


def _flush(collected: list[RedditPost], out: Path) -> None:
    if not collected:
        LOGGER.warning("No rows to flush yet.")
        return
    frame = pd.DataFrame([asdict(p) for p in collected]).drop_duplicates("post_id")
    frame.to_csv(out, index=False)
    LOGGER.info("Checkpoint: wrote %d unique reddit rows to %s", len(frame), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect()
