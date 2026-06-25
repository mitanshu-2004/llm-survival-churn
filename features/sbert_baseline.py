"""
Sentence-Transformer (SBERT) frozen-embedding baseline.

Encodes the review text with `sentence-transformers/all-MiniLM-L6-v2`
(already cached locally), then reduces 384 → 16 dims via PCA so the
embeddings can be plugged into Cox PH alongside the behavioural features.

The raw 384-dim embeddings are cached to `results/embeddings_sbert.npy` , 
on a re-run we read the cache instead of re-encoding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR  # noqa: E402

SBERT_MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_EMBEDDINGS_PATH  = RESULTS_DIR / "embeddings_sbert.npy"
SBERT_FEATURES_PATH    = RESULTS_DIR / "features_sbert.parquet"


def encode_reviews(df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
    """Return (n, 384) float32 SBERT embeddings, cached to disk."""
    if use_cache and SBERT_EMBEDDINGS_PATH.exists():
        arr = np.load(SBERT_EMBEDDINGS_PATH)
        if arr.shape[0] == len(df):
            print(f"Loaded cached SBERT embeddings from {SBERT_EMBEDDINGS_PATH}")
            return arr

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"Encoding {len(df):,} reviews with {SBERT_MODEL_NAME} (CPU)…")
    emb = model.encode(
        df["review_text"].fillna("").tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    if use_cache:
        SBERT_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(SBERT_EMBEDDINGS_PATH, emb)
        print(f"  Cached → {SBERT_EMBEDDINGS_PATH}")
    return emb


def add_sbert_features(
    df: pd.DataFrame,
    train_mask: np.ndarray | None = None,
    n_components: int = 64,
    save: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """Add sbert_0..sbert_{n-1} columns derived from PCA(SBERT-embedding).

    PCA is fit on the training mask only (when provided). The default
    ``n_components=64`` is chosen so that the bottleneck captures most of
    the embedding's variance (typically >85% for ``all-MiniLM-L6-v2`` on
    short reviews), the earlier 16-component setting was too aggressive and
    invited the criticism that the dense baseline was being starved of
    predictive dimensions.
    """
    emb = encode_reviews(df)
    pca = PCA(n_components=n_components, random_state=random_state)

    if train_mask is not None:
        pca.fit(emb[train_mask])
        X = pca.transform(emb)
    else:
        X = pca.fit_transform(emb)

    cols = [f"sbert_{i}" for i in range(n_components)]
    feat = pd.DataFrame(X, columns=cols, index=df.index)
    out  = pd.concat([df, feat], axis=1)

    if save:
        out.to_parquet(SBERT_FEATURES_PATH, index=False)
    return out, pca
