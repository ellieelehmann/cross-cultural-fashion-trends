"""Collect editorial coverage via Google News RSS.

Google News exposes a public RSS endpoint that is language- and
country-scoped. It requires no API key and returns real editorial
headlines from magazines and newspapers indexed by Google News:

    https://news.google.com/rss/search
        ?q=<query>
        &hl=<lang>-<COUNTRY>
        &gl=<COUNTRY>
        &ceid=<COUNTRY>:<lang>

This gives us a third data source alongside Google Trends (search
interest) and Reddit (community discourse): fashion-media voice, in
the target market's own language.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import pandas as pd

from .config import RAW_DIR, CaseStudy, ensure_dirs, load_case_study


LOGGER = logging.getLogger(__name__)

BASE_URL = "https://news.google.com/rss/search"


@dataclass
class NewsItem:
    aesthetic: str
    market: str
    language: str
    keyword: str
    kind: str
    source: str
    title: str
    summary: str
    published: str
    link: str


def _rss_url(query: str, language: str, country: str) -> str:
    params = {
        "q": query,
        "hl": f"{language}-{country}",
        "gl": country,
        "ceid": f"{country}:{language}",
    }
    return f"{BASE_URL}?{urlencode(params)}"


def collect(
    case: CaseStudy | None = None,
    sleep_between: float = 0.6,
    per_query_limit: int = 20,
) -> Path:
    """Fetch Google News RSS results for every configured query."""
    case = case or load_case_study()
    ensure_dirs()

    queries = list(case.all_queries())
    LOGGER.info("Google News RSS: %d queries planned", len(queries))

    market_country: dict[str, str] = {
        m.market: m.country_code
        for a in case.aesthetics.values()
        for m in a.markets
    }

    collected: list[NewsItem] = []
    for aesthetic_key, market, language, keyword, kind in queries:
        country = market_country[market]
        url = _rss_url(keyword, language, country)
        try:
            LOGGER.info("[%s / %s] news '%s' (%s)", country, language, keyword, kind)
            feed = feedparser.parse(url)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skip %s / %s: %s", keyword, country, exc)
            continue

        entries = feed.entries[:per_query_limit] if hasattr(feed, "entries") else []
        for entry in entries:
            source_name = ""
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source_name = entry.source.get("title", "")
            elif hasattr(entry, "source"):
                source_name = getattr(entry.source, "title", "")

            collected.append(
                NewsItem(
                    aesthetic=aesthetic_key,
                    market=market,
                    language=language,
                    keyword=keyword,
                    kind=kind,
                    source=source_name,
                    title=getattr(entry, "title", ""),
                    summary=getattr(entry, "summary", ""),
                    published=getattr(entry, "published", ""),
                    link=getattr(entry, "link", ""),
                )
            )
        time.sleep(sleep_between)

    if not collected:
        raise RuntimeError("No news items collected.")

    frame = pd.DataFrame([asdict(n) for n in collected]).drop_duplicates("link")
    out = RAW_DIR / "news.csv"
    frame.to_csv(out, index=False)
    LOGGER.info("Wrote %d news rows to %s", len(frame), out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect()
