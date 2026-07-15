"""Collect Google Trends search-interest data.

Uses `pytrends`, the unofficial Google Trends client. Google frequently
changes its backend, so this module fails gracefully: if a specific
keyword fails or `pytrends` is not installed, the caller can fall back
to any partial data plus the sample dataset.

Iterates over both aesthetic vibe terms and brand aliases per market,
tagging each row as either 'aesthetic' or 'brand'.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, CaseStudy, ensure_dirs, load_case_study


LOGGER = logging.getLogger(__name__)


def fetch_interest_over_time(
    keyword: str,
    country_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return an interest-over-time frame for a single keyword+country."""
    from pytrends.request import TrendReq

    trend = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.5)
    timeframe = f"{start} {end}"
    trend.build_payload([keyword], timeframe=timeframe, geo=country_code)
    frame = trend.interest_over_time()
    if frame.empty:
        return frame
    frame = frame.reset_index().rename(columns={keyword: "interest"})
    frame["keyword"] = keyword
    frame["country_code"] = country_code
    return frame[["date", "keyword", "country_code", "interest"]]


def collect(
    case: CaseStudy | None = None,
    sleep_between: float = 3.0,
    resume: bool = True,
    cooldown_on_429: float = 45.0,
    max_consecutive_429: int = 6,
) -> Path:
    """Collect Google Trends data for every configured query.

    Resumable by default: already-collected (aesthetic, market, keyword)
    triples in `data/raw/trends.csv` are skipped so a re-run backfills gaps
    without re-hitting rate limits. On a 429 the collector cools down
    briefly, and bails cleanly if 429s happen back-to-back.
    """
    case = case or load_case_study()
    ensure_dirs()

    start = case.time_window.get("start", "2023-01-01")
    end = case.time_window.get("end", "today")

    queries = list(case.all_queries())
    LOGGER.info("Google Trends: %d queries planned", len(queries))

    market_country: dict[str, str] = {
        m.market: m.country_code
        for a in case.aesthetics.values()
        for m in a.markets
    }

    out = RAW_DIR / "trends.csv"
    existing: pd.DataFrame | None = None
    already: set[tuple[str, str, str]] = set()
    if resume and out.exists():
        try:
            existing = pd.read_csv(out)
            if {"aesthetic", "market", "keyword"}.issubset(existing.columns):
                already = set(
                    zip(existing["aesthetic"], existing["market"], existing["keyword"])
                )
                LOGGER.info(
                    "Resume: %d rows already on disk covering %d unique queries",
                    len(existing),
                    len(already),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read existing trends.csv (%s); starting fresh", exc)
            existing = None

    rows: list[pd.DataFrame] = []
    consecutive_429 = 0
    for aesthetic_key, market, language, keyword, kind in queries:
        if (aesthetic_key, market, keyword) in already:
            continue
        country_code = market_country[market]
        try:
            LOGGER.info("[%s / %s] %s (%s)", country_code, language, keyword, kind)
            frame = fetch_interest_over_time(
                keyword=keyword,
                country_code=country_code,
                start=start,
                end=end,
            )
            consecutive_429 = 0
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            LOGGER.warning("Skip %s / %s: %s", market, keyword, message[:180])
            if "429" in message or "too many" in message.lower():
                consecutive_429 += 1
                if consecutive_429 >= max_consecutive_429:
                    LOGGER.warning(
                        "Bailing after %d consecutive 429s; run again later to resume.",
                        consecutive_429,
                    )
                    break
                LOGGER.info("Cooling down for %.0fs after 429...", cooldown_on_429)
                time.sleep(cooldown_on_429)
            continue

        if frame.empty:
            continue
        frame["aesthetic"] = aesthetic_key
        frame["market"] = market
        frame["language"] = language
        frame["kind"] = kind
        rows.append(frame)
        time.sleep(sleep_between)

    fresh = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if existing is not None and not existing.empty and not fresh.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
    elif not fresh.empty:
        combined = fresh
    elif existing is not None:
        combined = existing
    else:
        raise RuntimeError(
            "No Google Trends data collected. "
            "Install `pytrends`, check network access, or use the sample dataset."
        )

    combined.to_csv(out, index=False)
    LOGGER.info(
        "Wrote %d trend rows to %s (added %d new rows this pass)",
        len(combined),
        out,
        len(fresh),
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect()
