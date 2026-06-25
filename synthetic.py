"""Offline synthetic smoke test for the survival-modelling pipeline.

The full pipeline needs the McAuley Steam dataset and a Groq key for LLM feature
extraction, so a reviewer cannot reproduce it end-to-end out of the box. This module
generates a synthetic survival dataset whose churn risk genuinely depends on the
(synthetic) LLM-style signals, then fits the same Cox models used in the real
experiment and reports held-out concordance. It is fully self-contained, no network,
no Groq, no committed artifacts touched, so it proves the modelling code runs and
that the augmented model recovers the planted signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from models.cox import CoxSurvivalModel
from models.experiment import BEHAVIORAL_FEATURES, AUGMENTED_FEATURES


def make_synthetic_survival(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic survival dataset matching the real schema.

    Churn risk is a known linear function of the features (more frustration and
    disengagement raise it; positive sentiment lowers it), so the augmented Cox
    model should achieve concordance clearly above 0.5.
    """
    rng = np.random.default_rng(seed)

    log_playtime_2weeks = rng.normal(1.0, 1.0, n)
    log_items_count = rng.normal(2.0, 1.0, n)
    sentiment_score = rng.uniform(-1.0, 1.0, n)
    frustration_level = rng.uniform(0.0, 1.0, n)
    technical_issue = rng.integers(0, 2, n).astype(float)
    value_complaint = rng.integers(0, 2, n).astype(float)
    engagement_dropped = rng.integers(0, 2, n).astype(float)
    positive_signal = rng.integers(0, 2, n).astype(float)

    risk = (
        0.8 * frustration_level
        - 0.6 * sentiment_score
        + 0.5 * engagement_dropped
        - 0.4 * positive_signal
        + 0.3 * technical_issue
        - 0.2 * log_playtime_2weeks
    )
    risk = (risk - risk.mean()) / risk.std()

    # Survival time shrinks as risk rises; censoring is independent of risk.
    scale = 1.0 / (0.1 * np.exp(0.7 * risk))
    true_time = rng.exponential(scale)
    censor_time = rng.exponential(scale.mean() * 1.5, n)
    duration = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)

    return pd.DataFrame({
        "duration": duration,
        "event": event,
        "log_playtime_2weeks": log_playtime_2weeks,
        "log_items_count": log_items_count,
        "sentiment_score": sentiment_score,
        "frustration_level": frustration_level,
        "technical_issue": technical_issue,
        "value_complaint": value_complaint,
        "engagement_dropped": engagement_dropped,
        "positive_signal": positive_signal,
    })


def run_synthetic_smoke(n: int = 1000, seed: int = 0) -> dict:
    """Fit baseline + augmented Cox models on synthetic data; return held-out C-index."""
    df = make_synthetic_survival(n=n, seed=seed)
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["event"], random_state=1729
    )

    baseline = CoxSurvivalModel(label="baseline").fit(train, BEHAVIORAL_FEATURES)
    augmented = CoxSurvivalModel(label="augmented").fit(train, AUGMENTED_FEATURES)

    result = {
        "n": n,
        "churn_rate": float(df["event"].mean()),
        "baseline_holdout_cindex": float(baseline.cindex_on(test)),
        "augmented_holdout_cindex": float(augmented.cindex_on(test)),
    }
    print(
        f"Synthetic smoke (n={n}): churn={result['churn_rate']:.1%} | "
        f"baseline C-index={result['baseline_holdout_cindex']:.3f} | "
        f"augmented C-index={result['augmented_holdout_cindex']:.3f}"
    )
    return result


if __name__ == "__main__":
    run_synthetic_smoke()
