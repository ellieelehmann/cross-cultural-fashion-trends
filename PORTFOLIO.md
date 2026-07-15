# Portfolio Integration

Copy-ready snippets for the portfolio site and job applications.

---

## One-line pitch

> A multilingual NLP dashboard that traces how "quiet luxury" is searched
> for and discussed across New York, Paris, Seoul, and Tokyo.

## 3-line project card

> **Cross-Cultural Fashion Trend Mapping**
> Python · Hugging Face · Streamlit · Plotly
> Traces one fashion aesthetic across four regional markets and languages
> using Google Trends signals, transformer-based multilingual sentiment,
> and TF-IDF vocabulary analysis.

## Resume bullet

> Built a multilingual NLP dashboard mapping how one fashion aesthetic is
> discussed across four regional markets and languages (English, French,
> Korean, Japanese), combining Google Trends signals, Hugging Face
> transformer sentiment (23 languages), sentence-transformer embeddings,
> and TF-IDF vocabulary analysis. Delivered as a reproducible Python
> pipeline and an interactive Streamlit dashboard.

## Skills to tag

`Python` · `pandas` · `Hugging Face Transformers` · `sentence-transformers` ·
`scikit-learn` · `UMAP` · `Streamlit` · `Plotly` · `NLP` · `Sentiment analysis` ·
`Multilingual text processing` · `Data visualization` · `Dashboarding`

## Suggested Next.js portfolio card

```tsx
<ProjectCard
  title="Cross-Cultural Fashion Trend Mapping"
  slug="fashion-trend-mapping"
  stack={["Python", "Hugging Face", "Streamlit", "Plotly"]}
  cover="/projects/fashion-trend-mapping/dashboard-trends.png"
  blurb="A multilingual NLP dashboard tracing how quiet luxury is discussed across New York, Paris, Seoul, and Tokyo."
  demo="https://<your-streamlit-cloud-url>"
  code="https://github.com/<you>/cross-cultural-fashion-trends"
/>
```

## Deploying the live demo

The dashboard runs on Streamlit Community Cloud with zero configuration:

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `app/streamlit_app.py` and select `requirements-lite.txt`.
3. The checked-in sample dataset makes the demo self-contained.

## Screenshots to feature

- `assets/dashboard-trends.png` — hero image (recommended cover)
- `assets/dashboard-voices.png` — strongest single image; multilingual quotes side-by-side
- `assets/dashboard-vocabulary.png` — per-market vocabulary comparison
- `assets/dashboard-sentiment.png` — sentiment distributions
- `assets/dashboard-semantic.png` — multilingual embedding scatter
- `assets/dashboard-analyst.png` — auto-generated insight cards
