"""
TF-IDF (n-gram) baseline for the ablation table.

Builds a low-dimensional numerical embedding from review text by fitting
TfidfVectorizer + TruncatedSVD on the training slice only. The reduced
components are used as Cox PH covariates so the model stays well-posed
(2 000+ sparse TF-IDF features would be unstable for a 10k-row Cox fit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR  # noqa: E402

TFIDF_FEATURES_PATH = RESULTS_DIR / "features_tfidf.parquet"


def build_tfidf_pipeline(n_components: int = 64, random_state: int = 42) -> Pipeline:
    """Return an un-fitted TF-IDF → TruncatedSVD pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
    ])


def add_tfidf_features(
    df: pd.DataFrame,
    train_mask: np.ndarray | None = None,
    n_components: int = 64,
    save: bool = True,
) -> tuple[pd.DataFrame, Pipeline]:
    """Add tfidf_0..tfidf_{n-1} columns to df.

    If `train_mask` is provided, fit the vectorizer only on those rows
    (avoids leaking test-set vocabulary). Otherwise fit on full df , 
    fine for diagnostics, never for the headline ablation. The default
    ``n_components=64`` matches the SBERT baseline so the two text
    representations get the same dimensionality budget.
    """
    pipeline = build_tfidf_pipeline(n_components=n_components)
    text = df["review_text"].fillna("").tolist()
    if train_mask is not None:
        pipeline.fit([t for t, keep in zip(text, train_mask) if keep])
        X = pipeline.transform(text)
    else:
        X = pipeline.fit_transform(text)

    cols = [f"tfidf_{i}" for i in range(n_components)]
    feat = pd.DataFrame(X, columns=cols, index=df.index)
    out  = pd.concat([df, feat], axis=1)

    if save:
        out.to_parquet(TFIDF_FEATURES_PATH, index=False)
    return out, pipeline
