"""Streamlit dashboard for cross-cultural fashion trend mapping.

Editorial-magazine visual language: serif display typography, warm
neutral palette, generous white space, card-based layout. Every view
answers a question a fashion analyst would actually ask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, load_case_study  # noqa: E402


# ---------- Design tokens ----------

# Warm editorial palette. Cream paper, deep ink, cognac accent,
# and four market colors that read as fashion-magazine rather than
# spreadsheet.
PALETTE = {
    "paper": "#f7f2e9",
    "paper_soft": "#efe8db",
    "card": "#ffffff",
    "ink": "#1a1918",
    "ink_soft": "#4a4744",
    "muted": "#8a8378",
    "rule": "#d9cfbb",
    "accent": "#8b5a2b",
    "accent_soft": "#c9a97a",
}

MARKET_COLORS = {
    "US": "#8b1a1a",   # deep crimson  – New York
    "FR": "#c9a97a",   # muted gold    – Paris
    "KR": "#4a5d4e",   # forest green  – Seoul
    "JP": "#3d4a5c",   # ink blue      – Tokyo
}

MARKET_LABELS = {
    "US": "New York",
    "FR": "Paris",
    "KR": "Seoul",
    "JP": "Tokyo",
}

MARKET_ORDER = ["US", "FR", "KR", "JP"]

SENTIMENT_ORDER = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
]

SENTIMENT_COLORS = {
    "Very Negative": "#6b1e2e",
    "Negative": "#b47a7a",
    "Neutral": "#c9c1b0",
    "Positive": "#8ba888",
    "Very Positive": "#4a6b52",
}

MARKET_TAGLINES = {
    "quiet_luxury": {
        "US": "loud opinions, quiet clothes",
        "FR": "the discreet inheritance",
        "KR": "minimalism as status",
        "JP": "quality as vocabulary",
    },
    "punk_grunge": {
        "US": "raw, ripped, and refused",
        "FR": "elegance with a snarl",
        "KR": "streetwear meets subculture",
        "JP": "avant-garde as tradition",
    },
}

AESTHETIC_LEAD = {
    "quiet_luxury": (
        "Tracing how <em>Quiet Luxury</em> is searched for, discussed, and "
        "emotionally received across <strong>{cities}</strong>."
    ),
    "punk_grunge": (
        "Tracing how <em>Punk / Grunge</em> is searched for, discussed, and "
        "emotionally received across <strong>{cities}</strong>."
    ),
}

ANALYST_TONE = {
    "quiet_luxury": {
        "positive": "Language tends to focus on craft, restraint, and quality.",
        "negative": "Common critiques frame the aesthetic as marketing, class performance, or a passing micro-trend.",
    },
    "punk_grunge": {
        "positive": "Language tends to celebrate subculture, deconstruction, and designer icons.",
        "negative": "Common critiques frame the aesthetic as commodified rebellion or 90s revival fatigue.",
    },
}


st.set_page_config(
    page_title="Cross-Cultural Fashion Trend Mapping",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&family=Cormorant+Garamond:ital,wght@0,300;0,400;1,400&family=Inter:wght@300;400;500;600&display=swap");
        :root {{
            --paper: {PALETTE['paper']};
            --paper-soft: {PALETTE['paper_soft']};
            --card: {PALETTE['card']};
            --ink: {PALETTE['ink']};
            --ink-soft: {PALETTE['ink_soft']};
            --muted: {PALETTE['muted']};
            --rule: {PALETTE['rule']};
            --accent: {PALETTE['accent']};
            --accent-soft: {PALETTE['accent_soft']};
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: var(--paper) !important;
            color: var(--ink);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 400;
        }}

        [data-testid="stHeader"] {{ background: transparent; }}

        [data-testid="stSidebar"] {{
            background-color: var(--paper-soft) !important;
            border-right: 1px solid var(--rule);
        }}
        [data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
        [data-testid="stSidebar"] .stMarkdown p {{
            font-family: "Inter", sans-serif;
            font-size: 0.85rem;
            color: var(--ink-soft) !important;
        }}

        /* Sidebar multiselect chips: deep-red pill, crisp white text */
        [data-testid="stSidebar"] [data-baseweb="tag"] {{
            background-color: #8b1a1a !important;
            border: 1px solid #6b1414 !important;
            color: #ffffff !important;
            font-family: "Inter", sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.15rem 0.25rem 0.15rem 0.55rem !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="tag"] * {{
            color: #ffffff !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="tag"] span {{
            color: #ffffff !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="tag"] svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
            opacity: 0.95;
        }}
        [data-testid="stSidebar"] [data-baseweb="tag"]:hover {{
            background-color: #a02222 !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="tag"] [role="button"]:hover svg {{
            fill: #ffffff !important;
            opacity: 1;
        }}

        /* Selectbox value text and dropdown chevron */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: #ffffff !important;
            border: 1px solid var(--rule) !important;
            color: var(--ink) !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] div[role="button"] {{
            color: var(--ink) !important;
        }}

        /* Typography */
        h1, h2, h3, h4 {{
            font-family: "Playfair Display", "Didot", serif;
            font-weight: 500;
            letter-spacing: -0.01em;
            color: var(--ink);
        }}
        h1 {{
            font-size: 3.4rem;
            line-height: 1.05;
            font-weight: 500;
            margin-bottom: 0.4rem;
        }}
        h2 {{ font-size: 2rem; }}
        h3 {{ font-size: 1.55rem; font-weight: 500; }}
        h4 {{ font-size: 1.15rem; font-weight: 500; }}

        .stCaption, .st-emotion-cache-1wivap2 p, [data-testid="stCaptionContainer"] {{
            font-family: "Inter", sans-serif;
            color: var(--muted) !important;
            font-size: 0.88rem;
            letter-spacing: 0.01em;
        }}

        /* Hero */
        .hero {{
            border-bottom: 1px solid var(--rule);
            padding-bottom: 1.4rem;
            margin-bottom: 1.8rem;
        }}
        .hero .kicker {{
            font-family: "Inter", sans-serif;
            font-size: 0.72rem;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 500;
        }}
        .hero .lead {{
            font-family: "Cormorant Garamond", serif;
            font-size: 1.35rem;
            font-weight: 300;
            line-height: 1.5;
            color: var(--ink-soft);
            max-width: 780px;
            margin-top: 0.6rem;
            font-style: italic;
        }}

        /* KPI cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0 2rem 0;
        }}
        .kpi {{
            background: var(--card);
            border: 1px solid var(--rule);
            border-radius: 2px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 1px 2px rgba(26, 25, 24, 0.02);
        }}
        .kpi .label {{
            font-family: "Inter", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 500;
        }}
        .kpi .value {{
            font-family: "Playfair Display", serif;
            font-size: 2.1rem;
            font-weight: 500;
            color: var(--ink);
            line-height: 1.1;
            margin-top: 0.55rem;
        }}
        .kpi .sub {{
            font-family: "Inter", sans-serif;
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }}

        /* Section headers */
        .section-head {{
            display: flex;
            align-items: baseline;
            gap: 0.9rem;
            margin: 2.2rem 0 0.4rem 0;
        }}
        .section-head .num {{
            font-family: "Playfair Display", serif;
            font-style: italic;
            font-weight: 400;
            color: var(--accent);
            font-size: 1.3rem;
        }}
        .section-head h3 {{
            margin: 0;
            padding: 0;
        }}
        .section-caption {{
            font-family: "Cormorant Garamond", serif;
            font-style: italic;
            font-size: 1.05rem;
            color: var(--ink-soft);
            margin-bottom: 1.1rem;
            max-width: 780px;
        }}

        /* Insight card */
        .insight-card {{
            background: var(--card);
            border: 1px solid var(--rule);
            border-left: 3px solid var(--accent);
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.9rem;
            border-radius: 2px;
        }}
        .insight-card h4 {{
            margin: 0 0 0.5rem 0;
            font-family: "Playfair Display", serif;
        }}
        .insight-card p {{
            margin: 0;
            font-family: "Inter", sans-serif;
            font-size: 0.94rem;
            color: var(--ink-soft);
            line-height: 1.55;
        }}

        /* Editorial-style inline note (used for coverage caveats) */
        .editorial-note {{
            font-family: "Cormorant Garamond", serif;
            font-size: 1.02rem;
            font-style: italic;
            color: var(--ink-soft);
            background: var(--card);
            border-left: 3px solid var(--accent);
            padding: 0.9rem 1.2rem;
            margin: 0.4rem 0 1.4rem 0;
            line-height: 1.55;
        }}
        .editorial-note strong {{
            font-style: normal;
            font-family: "Inter", sans-serif;
            font-weight: 600;
            color: var(--ink);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
        }}
        .editorial-note code {{
            font-family: "SF Mono", "Menlo", monospace;
            font-size: 0.82rem;
            background: rgba(139, 26, 26, 0.06);
            padding: 0.05rem 0.35rem;
            border-radius: 2px;
            color: var(--accent);
            font-style: normal;
        }}

        /* Voice cards */
        .voice-card {{
            background: var(--card);
            border: 1px solid var(--rule);
            border-radius: 2px;
            padding: 1.5rem 1.5rem 1.3rem 1.5rem;
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        .voice-card .voice-market {{
            font-family: "Inter", sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }}
        .voice-card .voice-title {{
            font-family: "Playfair Display", serif;
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 0.15rem;
        }}
        .voice-card .voice-tagline {{
            font-family: "Cormorant Garamond", serif;
            font-style: italic;
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 1rem;
        }}
        .voice-card .voice-quote {{
            font-family: "Cormorant Garamond", serif;
            font-size: 1.1rem;
            font-style: italic;
            line-height: 1.55;
            color: var(--ink);
            border-left: 2px solid var(--accent-soft);
            padding-left: 0.9rem;
            margin-bottom: 0.8rem;
        }}
        .voice-card .voice-attr {{
            font-family: "Inter", sans-serif;
            font-size: 0.75rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .voice-card .voice-pill {{
            display: inline-block;
            font-family: "Inter", sans-serif;
            font-size: 0.7rem;
            font-weight: 500;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            margin-top: 0.3rem;
            margin-right: 0.3rem;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2rem;
            border-bottom: 1px solid var(--rule);
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: "Inter", sans-serif !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: var(--muted) !important;
            padding: 0.6rem 0 !important;
            font-weight: 500 !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--ink) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: var(--accent) !important;
            height: 2px !important;
        }}

        /* Widgets */
        .stSelectbox, .stMultiSelect {{
            font-family: "Inter", sans-serif;
        }}
        .stSelectbox label, .stMultiSelect label {{
            font-size: 0.7rem !important;
            letter-spacing: 0.22em !important;
            text-transform: uppercase !important;
            color: var(--muted) !important;
            font-weight: 500 !important;
        }}

        /* Divider rule */
        .rule {{
            border: none;
            border-top: 1px solid var(--rule);
            margin: 2.2rem 0 1.6rem 0;
        }}

        .footer {{
            border-top: 1px solid var(--rule);
            margin-top: 3rem;
            padding-top: 1.4rem;
            font-family: "Inter", sans-serif;
            font-size: 0.78rem;
            color: var(--muted);
            letter-spacing: 0.04em;
        }}

        /* Hide default Streamlit chrome we do not need */
        #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _register_plotly_template() -> None:
    """Editorial Plotly template used by every chart."""
    pio.templates["editorial"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, sans-serif", size=13, color=PALETTE["ink"]),
            title=dict(font=dict(family="Playfair Display, serif", size=18, color=PALETTE["ink"])),
            paper_bgcolor=PALETTE["card"],
            plot_bgcolor=PALETTE["card"],
            colorway=[MARKET_COLORS[m] for m in MARKET_ORDER],
            margin=dict(l=48, r=24, t=32, b=48),
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor=PALETTE["rule"],
                linewidth=1,
                ticks="outside",
                tickcolor=PALETTE["rule"],
                tickfont=dict(size=11, color=PALETTE["ink_soft"]),
                title=dict(font=dict(size=11, color=PALETTE["muted"])),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=PALETTE["rule"],
                gridwidth=0.5,
                showline=False,
                zeroline=False,
                tickfont=dict(size=11, color=PALETTE["ink_soft"]),
                title=dict(font=dict(size=11, color=PALETTE["muted"])),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color=PALETTE["ink_soft"]),
                itemsizing="constant",
                title=dict(text=""),
            ),
            hoverlabel=dict(
                bgcolor=PALETTE["card"],
                bordercolor=PALETTE["rule"],
                font=dict(family="Inter, sans-serif", size=12, color=PALETTE["ink"]),
            ),
        )
    )


def market_label(code: str) -> str:
    return f"{MARKET_LABELS.get(code, code)}"


def market_color_map(codes: list[str]) -> dict[str, str]:
    return {market_label(c): MARKET_COLORS[c] for c in codes if c in MARKET_COLORS}


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "trends": pd.read_parquet(PROCESSED_DIR / "trends.parquet"),
        "posts": pd.read_parquet(PROCESSED_DIR / "posts.parquet"),
        "keywords": pd.read_parquet(PROCESSED_DIR / "market_keywords.parquet"),
        "embeddings": pd.read_parquet(PROCESSED_DIR / "embeddings.parquet"),
    }


def render_hero(case, selected_aesthetic: str) -> None:
    aesthetic = case.aesthetics[selected_aesthetic]
    cities = ", ".join(MARKET_LABELS[m.market] for m in aesthetic.markets)
    lead_template = AESTHETIC_LEAD.get(
        selected_aesthetic,
        "Tracing how <em>{name}</em> is searched for, discussed, and "
        "emotionally received across <strong>{cities}</strong>.",
    )
    lead = lead_template.format(cities=cities, name=aesthetic.display_name)
    st.markdown(
        f"""
        <div class="hero">
            <div class="kicker">A multilingual NLP study</div>
            <h1>Cross-Cultural<br>Fashion Trend Mapping</h1>
            <div class="lead">{lead}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(trends: pd.DataFrame, posts: pd.DataFrame) -> None:
    total_posts = len(posts)
    market_sources = [
        s["market"] for s in (trends, posts) if not s.empty and "market" in s.columns
    ]
    if market_sources:
        n_markets = pd.concat(market_sources).nunique()
    else:
        n_markets = 0
    n_languages = posts["language"].nunique() if not posts.empty else 0
    peak = "—"
    peak_sub = ""
    if not trends.empty:
        peak_row = trends.loc[trends["interest"].idxmax()]
        peak = str(peak_row["keyword"])
        peak_sub = market_label(peak_row["market"])
    elif not posts.empty and "sentiment_score" in posts.columns:
        by_kw = (
            posts.assign(abs_score=posts["sentiment_score"].abs())
            .groupby("keyword")["abs_score"]
            .mean()
            .sort_values(ascending=False)
        )
        if not by_kw.empty:
            peak = str(by_kw.index[0])
            peak_sub = "strongest editorial signal"

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi">
                <div class="label">Markets</div>
                <div class="value">{n_markets}</div>
                <div class="sub">cities compared</div>
            </div>
            <div class="kpi">
                <div class="label">Languages</div>
                <div class="value">{n_languages}</div>
                <div class="sub">natively processed</div>
            </div>
            <div class="kpi">
                <div class="label">Posts analyzed</div>
                <div class="value">{total_posts:,}</div>
                <div class="sub">multilingual corpus</div>
            </div>
            <div class="kpi">
                <div class="label">Peak signal</div>
                <div class="value" style="font-size:1.55rem;">{peak}</div>
                <div class="sub">{peak_sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_head(num: str, title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <span class="num">{num}</span>
            <h3>{title}</h3>
        </div>
        <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_tab(trends: pd.DataFrame, selected_markets: list[str]) -> None:
    if not selected_markets:
        st.info("Select at least one market from the sidebar.")
        return
    if trends.empty:
        st.info(
            "Google Trends has not been collected for this aesthetic yet. "
            "Run `make collect-trends` to populate the search-interest signal."
        )
        return

    available_markets = sorted(trends["market"].unique())
    expected_markets = [m for m in MARKET_ORDER if m in selected_markets]
    missing_from_data = [m for m in expected_markets if m not in available_markets]
    if missing_from_data:
        missing_labels = ", ".join(MARKET_LABELS.get(m, m) for m in missing_from_data)
        st.markdown(
            f"""
            <div class="editorial-note">
                <strong>A note on coverage &mdash;</strong>
                Google Trends rate-limited this aesthetic during collection, so
                search-interest for <strong>{missing_labels}</strong> could not be pulled
                in this pass. The chart below shows the markets Google did return
                ({', '.join(MARKET_LABELS.get(m, m) for m in available_markets)}).
                Re-run <code>make collect-trends</code> later to backfill &mdash;
                already-collected queries are skipped automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered = trends[trends["market"].isin(selected_markets)].copy()
    if filtered.empty:
        available = ", ".join(MARKET_LABELS.get(m, m) for m in available_markets)
        st.info(
            f"No Google Trends rows for the selected markets in this aesthetic. "
            f"Trends data is currently available for: **{available}**."
        )
        return
    filtered["market_label"] = filtered["market"].map(market_label)

    _section_head(
        "I.",
        "Search interest over time",
        "Monthly Google Trends interest, indexed 0-100. Each region is normalized "
        "to itself, so the shape of the curve matters more than absolute height.",
    )

    monthly = (
        filtered.groupby(["date", "market_label"], as_index=False)["interest"]
        .mean()
        .sort_values("date")
    )
    fig = px.line(
        monthly,
        x="date",
        y="interest",
        color="market_label",
        color_discrete_map=market_color_map(selected_markets),
        template="editorial",
    )
    fig.update_traces(line=dict(width=2.2), mode="lines")
    fig.update_layout(
        yaxis_title="Interest (0-100)",
        xaxis_title=None,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    _section_head(
        "II.",
        "Keyword-level heat map",
        "Which specific translation is the loudest signal in each region?",
    )
    pivot = (
        filtered.groupby(["market_label", "keyword"], as_index=False)["interest"]
        .mean()
        .pivot(index="keyword", columns="market_label", values="interest")
        .fillna(0)
    )
    fig2 = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[
            [0.0, PALETTE["paper_soft"]],
            [0.4, PALETTE["accent_soft"]],
            [0.75, PALETTE["accent"]],
            [1.0, PALETTE["ink"]],
        ],
        template="editorial",
        labels=dict(color="Interest"),
    )
    fig2.update_layout(
        height=420,
        coloraxis_colorbar=dict(
            thickness=10,
            tickfont=dict(size=10, color=PALETTE["ink_soft"]),
            outlinewidth=0,
        ),
        xaxis=dict(showline=False, side="top", tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})


def render_sentiment_tab(posts: pd.DataFrame, selected_markets: list[str]) -> None:
    filtered = posts[posts["market"].isin(selected_markets)].copy()
    if filtered.empty:
        st.info("Select at least one market from the sidebar.")
        return
    filtered["market_label"] = filtered["market"].map(market_label)

    _section_head(
        "III.",
        "Emotional tone by market",
        "Sentiment distribution from a multilingual transformer classifier. "
        "Cultural differences show up here even when the search-volume story looks similar.",
    )

    counts = (
        filtered.groupby(["market_label", "sentiment_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["share"] = counts.groupby("market_label")["count"].transform(
        lambda s: s / s.sum() * 100
    )
    counts["sentiment_label"] = pd.Categorical(
        counts["sentiment_label"], categories=SENTIMENT_ORDER, ordered=True
    )
    counts = counts.sort_values(["market_label", "sentiment_label"])

    fig = px.bar(
        counts,
        x="market_label",
        y="share",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        template="editorial",
        category_orders={
            "sentiment_label": SENTIMENT_ORDER,
            "market_label": [market_label(m) for m in selected_markets],
        },
        custom_data=["count", "sentiment_label"],
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{customdata[1]}: %{y:.1f}%%<br>%{customdata[0]} posts<extra></extra>",
    )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Share of posts (%)",
        xaxis_title=None,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    _section_head(
        "IV.",
        "Sentiment index",
        "Average sentiment score per market on a -1 to +1 scale. Above zero = net positive reception.",
    )
    avg = (
        filtered.groupby("market_label", as_index=False)["sentiment_score"]
        .mean()
        .sort_values("sentiment_score")
    )
    market_by_label = {market_label(c): c for c in selected_markets}
    bar_colors = [MARKET_COLORS[market_by_label[m]] for m in avg["market_label"]]
    fig2 = go.Figure(
        go.Bar(
            x=avg["sentiment_score"],
            y=avg["market_label"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=PALETTE["ink"], width=0)),
            text=[f"{v:+.2f}" for v in avg["sentiment_score"]],
            textposition="outside",
            textfont=dict(family="Playfair Display, serif", size=14, color=PALETTE["ink"]),
            hovertemplate="<b>%{y}</b><br>%{x:+.2f}<extra></extra>",
        )
    )
    fig2.add_vline(x=0, line_dash="dot", line_color=PALETTE["muted"], line_width=1)
    fig2.update_layout(
        template="editorial",
        xaxis=dict(title="", range=[-1, 1], tickformat="+.1f"),
        yaxis=dict(title=""),
        height=280,
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})


def render_language_tab(keywords: pd.DataFrame, selected_markets: list[str]) -> None:
    if not selected_markets:
        st.info("Select at least one market from the sidebar.")
        return
    filtered = keywords[keywords["market"].isin(selected_markets)].copy()

    _section_head(
        "V.",
        "Vocabulary of the aesthetic",
        "Top terms per market ranked by TF-IDF against the overall corpus. "
        "Terms that appear across every market drop out; culturally specific words rise.",
    )

    cols = st.columns(len(selected_markets))
    for col, market in zip(cols, selected_markets):
        market_slice = (
            filtered[filtered["market"] == market]
            .sort_values("score", ascending=False)
            .head(10)
        )
        with col:
            st.markdown(
                f"""
                <div style="margin-bottom:0.4rem;">
                    <div style="font-family:'Inter',sans-serif;font-size:0.65rem;letter-spacing:0.28em;text-transform:uppercase;color:{PALETTE['accent']};font-weight:500;">
                        {market}
                    </div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:500;color:{PALETTE['ink']};">
                        {market_label(market)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if market_slice.empty:
                st.caption("No terms available yet.")
                continue
            fig = px.bar(
                market_slice.iloc[::-1],
                x="score",
                y="term",
                orientation="h",
                template="editorial",
                color_discrete_sequence=[MARKET_COLORS[market]],
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>TF-IDF %{x:.3f}<extra></extra>",
                marker_line_width=0,
            )
            fig.update_layout(
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(showline=False, showticklabels=False, title=None),
                yaxis=dict(title=None, tickfont=dict(size=12, color=PALETTE["ink"])),
                height=360,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def render_semantic_tab(embeddings: pd.DataFrame, selected_markets: list[str]) -> None:
    filtered = embeddings[embeddings["market"].isin(selected_markets)].copy()
    if filtered.empty:
        st.info("Select at least one market from the sidebar.")
        return
    filtered["market_label"] = filtered["market"].map(market_label)
    filtered["preview"] = filtered["clean_text"].str.slice(0, 140)

    _section_head(
        "VI.",
        "Semantic space",
        "Each dot is one post placed by its multilingual embedding, reduced to 2D. "
        "Overlapping regions = shared meaning across cultures; isolated regions = "
        "culture-specific framing.",
    )

    fig = px.scatter(
        filtered,
        x="x",
        y="y",
        color="market_label",
        color_discrete_map=market_color_map(selected_markets),
        hover_data={
            "preview": True,
            "sentiment_label": True,
            "keyword": True,
            "x": False,
            "y": False,
            "market_label": False,
        },
        template="editorial",
    )
    fig.update_traces(
        marker=dict(
            size=11,
            opacity=0.75,
            line=dict(width=0.6, color=PALETTE["paper"]),
        )
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
        xaxis=dict(showticklabels=False, title=None, showgrid=False, zeroline=False, showline=False),
        yaxis=dict(showticklabels=False, title=None, showgrid=False, zeroline=False, showline=False),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def render_voices(
    posts: pd.DataFrame,
    selected_markets: list[str],
    selected_aesthetic: str,
) -> None:
    _section_head(
        "VII.",
        "Voices from each market",
        "Representative multilingual quotes from the corpus. The most emotionally "
        "charged post per market, alongside the market's overall tone.",
    )

    if posts.empty:
        st.info("No posts available.")
        return

    taglines = MARKET_TAGLINES.get(selected_aesthetic, {})
    cols = st.columns(len(selected_markets) or 1)
    for col, market in zip(cols, selected_markets):
        market_posts = posts[posts["market"] == market]
        if market_posts.empty:
            continue

        market_posts = market_posts.assign(abs_score=market_posts["sentiment_score"].abs())
        top = market_posts.sort_values("abs_score", ascending=False).iloc[0]
        mean_score = market_posts["sentiment_score"].mean()
        tone = "warm" if mean_score > 0.15 else "skeptical" if mean_score < -0.15 else "measured"
        color = MARKET_COLORS[market]

        with col:
            st.markdown(
                f"""
                <div class="voice-card">
                    <div class="voice-market">{market} &middot; {tone} tone</div>
                    <div class="voice-title">{market_label(market)}</div>
                    <div class="voice-tagline">&mdash; {taglines.get(market, '')}</div>
                    <div class="voice-quote" style="border-left-color:{color};">
                        &ldquo;{top['clean_text']}&rdquo;
                    </div>
                    <div>
                        <span class="voice-pill" style="background:{color}18;color:{color};">
                            {top['sentiment_label']}
                        </span>
                        <span class="voice-pill" style="background:{PALETTE['paper_soft']};color:{PALETTE['ink_soft']};">
                            {top['keyword']}
                        </span>
                    </div>
                    <div class="voice-attr" style="margin-top:0.9rem;">
                        Mean sentiment {mean_score:+.2f} &middot; {len(market_posts):,} posts
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insights(
    posts: pd.DataFrame,
    trends: pd.DataFrame,
    selected_aesthetic: str,
) -> None:
    _section_head(
        "VIII.",
        "Analyst take",
        "Auto-generated market-level insights, refreshed every time the data pipeline runs.",
    )

    if posts.empty:
        st.info("No posts to summarize yet.")
        return

    sentiment_by_market = (
        posts.groupby("market")["sentiment_score"].mean().sort_values()
    )
    most_negative = sentiment_by_market.index[0]
    most_positive = sentiment_by_market.index[-1]

    growth_card: tuple[str, str] | None = None
    if not trends.empty and "date" in trends.columns:
        trend_growth = (
            trends.assign(year=trends["date"].dt.year)
            .dropna(subset=["year"])
            .groupby(["market", "year"], as_index=False)["interest"]
            .mean()
            .pivot(index="market", columns="year", values="interest")
        )
        years = [y for y in trend_growth.columns if pd.notna(y)]
        if len(years) >= 2 and not trend_growth.empty:
            latest_year = max(years)
            earliest_year = min(years)
            growth = (
                trend_growth[latest_year] - trend_growth[earliest_year]
            ).dropna().sort_values()
            if len(growth) >= 2:
                fastest = growth.index[-1]
                growth_card = (
                    f"Fastest-growing market: {MARKET_LABELS.get(fastest, fastest)}",
                    f"Google Trends interest for this aesthetic rose the most "
                    f"between {int(earliest_year)} and {int(latest_year)} in this region.",
                )
            elif len(growth) == 1:
                sole = growth.index[0]
                delta = growth.iloc[0]
                direction = "rose" if delta >= 0 else "cooled"
                growth_card = (
                    f"{MARKET_LABELS.get(sole, sole)} carries the search signal",
                    f"Google Trends rate-limited the other markets for this aesthetic, "
                    f"so cross-market growth is not comparable yet. In "
                    f"{MARKET_LABELS.get(sole, sole)}, interest {direction} "
                    f"{delta:+.1f} points between {int(earliest_year)} and {int(latest_year)}.",
                )

    if growth_card is None:
        growth_card = (
            "Search interest not yet collected",
            "Google Trends has not been pulled for this aesthetic yet &mdash; "
            "run <code>make collect-trends</code> to fill in the growth signal.",
        )

    tone = ANALYST_TONE.get(selected_aesthetic, ANALYST_TONE["quiet_luxury"])
    cards = [
        (
            f"{MARKET_LABELS.get(most_positive, most_positive)} embraces the aesthetic",
            f"Highest average sentiment score across all analyzed posts. {tone['positive']}",
        ),
        (
            f"{MARKET_LABELS.get(most_negative, most_negative)} is the skeptic",
            f"Lowest average sentiment. {tone['negative']}",
        ),
        growth_card,
        (
            "Same word, different meanings",
            "The semantic scatter plot shows partial cluster overlap between markets, "
            "but each culture keeps its own tail of vocabulary that does not translate.",
        ),
    ]

    left, right = st.columns(2)
    containers = [left, right, left, right]
    for container, (title, body) in zip(containers, cards):
        with container:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer(posts: pd.DataFrame, trends: pd.DataFrame) -> None:
    n_posts = len(posts)
    n_publishers = posts["source"].nunique() if "source" in posts.columns else 0
    n_trend_rows = len(trends)
    st.markdown(
        f"""
        <div class="footer">
            Built with Python, pandas, Hugging Face Transformers
            (<code>tabularisai/multilingual-sentiment-analysis</code>),
            sentence-transformers
            (<code>paraphrase-multilingual-MiniLM-L12-v2</code>),
            UMAP, Plotly, and Streamlit.<br>
            Data: <strong>{n_posts:,}</strong> live news items collected from
            <strong>{n_publishers}</strong> publishers via Google News RSS
            &middot; <strong>{n_trend_rows:,}</strong> Google Trends rows via
            pytrends. No synthetic content.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    _register_plotly_template()
    case = load_case_study()
    data = load_data()

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.4rem 0 1rem 0;">
                <div style="font-family:'Inter',sans-serif;font-size:0.68rem;letter-spacing:0.28em;text-transform:uppercase;color:{PALETTE['accent']};font-weight:500;">
                    Filters
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:500;color:{PALETTE['ink']};margin-top:0.2rem;">
                    Curate the study
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        aesthetic_names = {k: v.display_name for k, v in case.aesthetics.items()}
        selected_aesthetic = st.selectbox(
            "Aesthetic",
            options=list(aesthetic_names.keys()),
            format_func=lambda k: aesthetic_names[k],
        )
        available_markets = sorted(
            {m.market for a in case.aesthetics.values() for m in a.markets},
            key=lambda m: MARKET_ORDER.index(m) if m in MARKET_ORDER else 99,
        )
        selected_markets = st.multiselect(
            "Markets",
            options=available_markets,
            default=available_markets,
            format_func=lambda m: f"{market_label(m)}",
        )
        st.markdown(
            f"""
            <hr style="border:none;border-top:1px solid {PALETTE['rule']};margin:1.4rem 0;">
            <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:1rem;color:{PALETTE['ink_soft']};line-height:1.5;">
                &ldquo;Fashion is architecture: it is a matter of proportions.&rdquo;
            </div>
            <div style="font-family:'Inter',sans-serif;font-size:0.72rem;letter-spacing:0.15em;text-transform:uppercase;color:{PALETTE['muted']};margin-top:0.5rem;">
                Coco Chanel
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_hero(case, selected_aesthetic)

    trends = data["trends"].query("aesthetic == @selected_aesthetic")
    posts = data["posts"].query("aesthetic == @selected_aesthetic")
    keywords = data["keywords"]
    if "aesthetic" in keywords.columns:
        keywords = keywords.query("aesthetic == @selected_aesthetic")
    embeddings = data["embeddings"].query("aesthetic == @selected_aesthetic")

    render_kpis(trends, posts)

    tabs = st.tabs(
        ["Trends", "Sentiment", "Vocabulary", "Voices", "Semantic Map", "Analyst Take"]
    )
    with tabs[0]:
        render_trend_tab(trends, selected_markets)
    with tabs[1]:
        render_sentiment_tab(posts, selected_markets)
    with tabs[2]:
        render_language_tab(keywords, selected_markets)
    with tabs[3]:
        render_voices(posts, selected_markets, selected_aesthetic)
    with tabs[4]:
        render_semantic_tab(embeddings, selected_markets)
    with tabs[5]:
        render_insights(posts, trends, selected_aesthetic)

    render_footer(data["posts"], data["trends"])


if __name__ == "__main__":
    main()
