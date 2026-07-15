"""Semantic clustering of multilingual fashion text.

Uses `sentence-transformers` (paraphrase-multilingual-MiniLM) to place
posts in a shared multilingual embedding space, then reduces to 2D with
UMAP for visualization and clusters with KMeans for interpretability.

Falls back to a simple hashed-bag-of-words + TruncatedSVD projection
if sentence-transformers is not installed.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Falling back to hashed embeddings: %s", exc)
        return None


def _hashed_embed(texts: list[str], dim: int = 128) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")
    return vec.transform(texts).toarray()


def embed(texts: list[str]) -> np.ndarray:
    encoder = _load_encoder()
    if encoder is None:
        return _hashed_embed(texts)
    return np.asarray(encoder.encode(texts, show_progress_bar=False, batch_size=32))


def reduce_2d(matrix: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D. Prefer UMAP; fall back to PCA."""
    if len(matrix) < 3:
        return np.zeros((len(matrix), 2))
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, len(matrix) - 1)),
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(matrix)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Falling back to PCA: %s", exc)
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(matrix)


def cluster_frame(frame: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """Return the input frame with `x`, `y`, and `cluster` columns."""
    if frame.empty:
        return frame.assign(x=[], y=[], cluster=[])

    texts = frame["clean_text"].fillna("").tolist()
    matrix = embed(texts)
    coords = reduce_2d(matrix)

    from sklearn.cluster import KMeans

    k = max(2, min(n_clusters, len(frame) // 8 or 2))
    labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(matrix)

    out = frame.copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out["cluster"] = labels
    return out
