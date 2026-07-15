"""Generate deterministic sample data for the dashboard.

Real Google Trends and Reddit access require network calls that can
break in CI or without API keys. The sample data is realistic enough
to demo the dashboard, exercise the full NLP pipeline, and let
reviewers explore the interface end-to-end.

Text snippets are hand-crafted per market to mimic how each language
community actually talks about quiet luxury (slang, hashtags, tone).
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAMPLE_DIR, ensure_dirs, load_case_study


RNG = np.random.default_rng(seed=42)
random.seed(42)


# Curated, realistic-sounding posts for each market/keyword. Multilingual
# on purpose: this dataset is what the NLP layer will be tested against.
SEED_POSTS: dict[str, list[dict]] = {
    "US": [
        {
            "text": "Quiet luxury is really just having a good tailor and a cashmere addiction. The Row lookbook lives in my head rent free.",
            "sentiment_hint": "positive",
        },
        {
            "text": "Old money aesthetic feels performative on TikTok now. It's just Ralph Lauren cosplay for people with debt.",
            "sentiment_hint": "negative",
        },
        {
            "text": "Stealth wealth basics: navy crewneck, gray flannels, unbranded loafers. Nothing screams, everything whispers.",
            "sentiment_hint": "positive",
        },
        {
            "text": "Honestly quiet luxury is a marketing term. Rich people always dressed like this.",
            "sentiment_hint": "neutral",
        },
        {
            "text": "Understated luxury > logo mania. My closet finally feels like mine.",
            "sentiment_hint": "positive",
        },
        {
            "text": "The stealth wealth thing is exhausting. It's just beige.",
            "sentiment_hint": "negative",
        },
        {
            "text": "Best quiet luxury piece I own: an unlined cashmere blazer. Wears like a sweatshirt.",
            "sentiment_hint": "positive",
        },
        {
            "text": "Old money aesthetic on Pinterest is 90% white girls at horse farms. Where is the range.",
            "sentiment_hint": "negative",
        },
    ],
    "FR": [
        {
            "text": "Le luxe discret, c'est la vraie élégance. Pas de logo, juste une coupe impeccable.",
            "sentiment_hint": "positive",
        },
        {
            "text": "Franchement le style old money version française c'est juste le vestiaire APC + Hermès accessible. Rien de neuf.",
            "sentiment_hint": "neutral",
        },
        {
            "text": "Luxe silencieux mon oeil, c'est encore une tendance TikTok qui va disparaître.",
            "sentiment_hint": "negative",
        },
        {
            "text": "J'adore ce retour au luxe discret, on respire enfin après des années de streetwear tape-à-l'oeil.",
            "sentiment_hint": "positive",
        },
        {
            "text": "Le luxe discret parisien c'est un manteau en cachemire beige et des mocassins. Chic mais tellement prévisible.",
            "sentiment_hint": "neutral",
        },
        {
            "text": "Style old money = privilège déguisé en minimalisme.",
            "sentiment_hint": "negative",
        },
        {
            "text": "Enfin des marques françaises qui portent le luxe discret sans en faire des tonnes.",
            "sentiment_hint": "positive",
        },
    ],
    "KR": [
        {
            "text": "올드머니룩 진짜 예쁘다. 니트 하나에 슬랙스만 입어도 분위기 다름.",
            "sentiment_hint": "positive",
        },
        {
            "text": "조용한 럭셔리 트렌드 그냥 비싸 보이려는 마케팅 아님?",
            "sentiment_hint": "negative",
        },
        {
            "text": "미니멀 럭셔리 좋아하는데 한국에서는 어울리는 브랜드가 많이 없어서 아쉬움.",
            "sentiment_hint": "neutral",
        },
        {
            "text": "요즘 셀럽들 다 올드머니룩. 지겨울 정도.",
            "sentiment_hint": "negative",
        },
        {
            "text": "조용한 럭셔리 코디 하나만 잘하면 데일리로 계속 입을 수 있어서 좋음.",
            "sentiment_hint": "positive",
        },
        {
            "text": "미니멀 럭셔리는 결국 캐시미어랑 좋은 재단이 답이다.",
            "sentiment_hint": "positive",
        },
        {
            "text": "올드머니룩 하려면 결국 돈이 필요함. 조용한 게 아니라 조용하게 돈 자랑.",
            "sentiment_hint": "negative",
        },
    ],
    "JP": [
        {
            "text": "クワイエットラグジュアリー、めっちゃ好き。ロゴなしのカシミアが最強。",
            "sentiment_hint": "positive",
        },
        {
            "text": "オールドマネー ファッション、日本人には少し重いかも。着こなしが難しい。",
            "sentiment_hint": "neutral",
        },
        {
            "text": "上品カジュアル、実は一番難しいスタイル。全部の質が問われる。",
            "sentiment_hint": "positive",
        },
        {
            "text": "クワイエットラグジュアリーって結局高い服買うだけでしょ。",
            "sentiment_hint": "negative",
        },
        {
            "text": "オールドマネー ファッションのシンプルさがちょうどいい。飽きない。",
            "sentiment_hint": "positive",
        },
        {
            "text": "上品カジュアルを目指すと結局ユニクロ+Theoryになる問題。",
            "sentiment_hint": "neutral",
        },
        {
            "text": "静かなラグジュアリーは日本の美意識と相性がいい。",
            "sentiment_hint": "positive",
        },
    ],
}


def _generate_trends(case) -> pd.DataFrame:
    """Realistic monthly interest-over-time per (market, keyword)."""
    start = datetime.fromisoformat(case.time_window.get("start", "2023-01-01"))
    end = datetime.fromisoformat(case.time_window.get("end", "2026-06-30"))
    months = pd.date_range(start=start, end=end, freq="MS")

    rows: list[dict] = []
    for aesthetic_key, aesthetic in case.aesthetics.items():
        for market in aesthetic.markets:
            for keyword in market.keywords:
                base = RNG.integers(20, 60)
                growth = RNG.uniform(0.15, 1.2)
                noise_scale = RNG.uniform(4, 12)
                trend = np.linspace(0, growth * 40, len(months))
                seasonal = 6 * np.sin(np.linspace(0, 6 * np.pi, len(months)))
                noise = RNG.normal(0, noise_scale, len(months))
                interest = np.clip(base + trend + seasonal + noise, 1, 100).round().astype(int)
                for date, value in zip(months, interest):
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "aesthetic": aesthetic_key,
                            "market": market.market,
                            "country_code": market.country_code,
                            "language": market.language,
                            "keyword": keyword,
                            "interest": int(value),
                        }
                    )
    return pd.DataFrame(rows)


def _generate_text(case) -> pd.DataFrame:
    """Multilingual text corpus with realistic per-market voice."""
    rows: list[dict] = []
    post_id = 0
    for aesthetic_key, aesthetic in case.aesthetics.items():
        for market in aesthetic.markets:
            seeds = SEED_POSTS.get(market.market, [])
            for keyword in market.keywords:
                repeats = 6
                for _ in range(repeats):
                    for seed in seeds:
                        post_id += 1
                        month_offset = RNG.integers(0, 30)
                        created = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(RNG.integers(0, 720)))
                        rows.append(
                            {
                                "post_id": f"s{post_id:06d}",
                                "aesthetic": aesthetic_key,
                                "market": market.market,
                                "language": market.language,
                                "keyword": keyword,
                                "text": seed["text"],
                                "sentiment_hint": seed["sentiment_hint"],
                                "score": int(RNG.integers(1, 500)),
                                "num_comments": int(RNG.integers(0, 120)),
                                "created_utc": created.timestamp() + month_offset * 3600,
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    case = load_case_study()

    trends = _generate_trends(case)
    text = _generate_text(case)

    trends_path = SAMPLE_DIR / "trends_sample.csv"
    text_path = SAMPLE_DIR / "text_sample.csv"
    trends.to_csv(trends_path, index=False)
    text.to_csv(text_path, index=False)

    print(f"Wrote {len(trends):,} trend rows to {trends_path}")
    print(f"Wrote {len(text):,} text rows to {text_path}")


if __name__ == "__main__":
    main()
