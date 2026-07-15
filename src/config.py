"""Load and normalize the aesthetics case-study config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "aesthetics.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"


@dataclass(frozen=True)
class Market:
    market: str
    country_code: str
    language: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Brand:
    name: str
    origin: str
    aliases: dict[str, str]  # language code -> localized brand name


@dataclass(frozen=True)
class Aesthetic:
    key: str
    display_name: str
    description: str
    markets: tuple[Market, ...]
    brands: tuple[Brand, ...] = ()


@dataclass(frozen=True)
class CaseStudy:
    primary_aesthetic: str
    aesthetics: dict[str, Aesthetic]
    time_window: dict[str, str]
    text_sources: dict[str, dict]

    def primary(self) -> Aesthetic:
        return self.aesthetics[self.primary_aesthetic]

    def all_keywords(self) -> Iterable[tuple[str, str, str, str]]:
        """Yield (aesthetic_key, market, language, keyword) rows."""
        for key, aesthetic in self.aesthetics.items():
            for market in aesthetic.markets:
                for keyword in market.keywords:
                    yield key, market.market, market.language, keyword

    def all_queries(self) -> Iterable[tuple[str, str, str, str, str]]:
        """Yield (aesthetic_key, market, language, query, kind) rows.

        `kind` is 'aesthetic' for vibe terms configured in market.keywords,
        or 'brand' for a localized brand alias in that market's language.
        """
        for key, aesthetic in self.aesthetics.items():
            for market in aesthetic.markets:
                for keyword in market.keywords:
                    yield key, market.market, market.language, keyword, "aesthetic"
                for brand in aesthetic.brands:
                    alias = brand.aliases.get(market.language, brand.name)
                    yield key, market.market, market.language, alias, "brand"


def load_case_study(path: Path = CONFIG_PATH) -> CaseStudy:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    aesthetics: dict[str, Aesthetic] = {}
    for key, entry in raw.get("aesthetics", {}).items():
        markets = tuple(
            Market(
                market=m["market"],
                country_code=m["country_code"],
                language=m["language"],
                keywords=tuple(m["keywords"]),
            )
            for m in entry.get("markets", [])
        )
        brands = tuple(
            Brand(
                name=b["name"],
                origin=b.get("origin", ""),
                aliases=dict(b.get("aliases", {})),
            )
            for b in entry.get("brands", [])
        )
        aesthetics[key] = Aesthetic(
            key=key,
            display_name=entry.get("display_name", key.replace("_", " ").title()),
            description=entry.get("description", ""),
            markets=markets,
            brands=brands,
        )

    return CaseStudy(
        primary_aesthetic=raw["primary_aesthetic"],
        aesthetics=aesthetics,
        time_window=raw.get("time_window", {}),
        text_sources=raw.get("text_sources", {}),
    )


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
