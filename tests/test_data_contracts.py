"""Data-layer contract tests, guard against accidentally reintroducing leaky features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT     = Path(__file__).resolve().parent.parent
_FEATURES = _ROOT / "results" / "features_llm.parquet"


def test_no_leaky_features_in_model_inputs():
    """Leaky features may exist in the parquet for diagnostics, but must NOT be in the
    feature lists actually consumed by the Cox model."""
    from models.experiment import AUGMENTED_FEATURES, BEHAVIORAL_FEATURES

    leaky = {"log_duration", "playtime_2wk_ratio", "duration_ratio", "duration"}
    used = set(BEHAVIORAL_FEATURES) | set(AUGMENTED_FEATURES)
    overlap = used & leaky
    assert not overlap, f"Leaky features in model feature lists: {overlap}"


@pytest.mark.skipif(not _FEATURES.exists(), reason="features_llm.parquet not built yet")
def test_event_and_duration_columns():
    df = pd.read_parquet(_FEATURES)
    assert "duration" in df.columns
    assert "event" in df.columns
    assert df["event"].isin([0, 1]).all()
    assert (df["duration"] > 0).all()


@pytest.mark.skipif(not _FEATURES.exists(), reason="features_llm.parquet not built yet")
def test_llm_features_in_expected_ranges():
    df = pd.read_parquet(_FEATURES)
    assert df["sentiment_score"].between(-1.0, 1.0).all()
    assert df["frustration_level"].between(0.0, 1.0).all()
    for col in ("technical_issue", "value_complaint", "engagement_dropped", "positive_signal"):
        assert df[col].isin([0, 1]).all(), f"{col} contains non-binary values"


def test_holdout_split_is_deterministic():
    """Same seed → same indices, zero overlap."""
    from sklearn.model_selection import train_test_split

    rng_df = pd.DataFrame({
        "event":  np.array([0, 1] * 50),
        "x":      np.arange(100),
    })
    train_a, test_a = train_test_split(
        rng_df, test_size=0.2, stratify=rng_df["event"], random_state=42,
    )
    train_b, test_b = train_test_split(
        rng_df, test_size=0.2, stratify=rng_df["event"], random_state=42,
    )
    assert list(train_a.index) == list(train_b.index)
    assert list(test_a.index) == list(test_b.index)
    assert not (set(train_a.index) & set(test_a.index))
