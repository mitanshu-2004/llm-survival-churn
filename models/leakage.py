"""
Leakage decomposition of the augmented Cox model.

Runs four Cox fits with progressively more LLM features to quantify which part
of the augmented C-index lift is plausibly predictive (non-polarity behavioural
text signals) vs descriptive (polarity features that re-encode the label).

Result is saved to `results/leakage_results.json` so the README and any
downstream consumer can render it without re-fitting four Cox models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES_PATH, RESULTS_DIR  # noqa: E402
from models.cox import CoxSurvivalModel  # noqa: E402
from models.experiment import _prepare_features  # noqa: E402

LEAKAGE_PATH = RESULTS_DIR / "leakage_results.json"

BEHAVIORAL    = ["log_playtime_2weeks", "log_items_count"]
NON_POLARITY  = ["technical_issue", "engagement_dropped"]
POLARITY      = ["sentiment_score", "frustration_level", "positive_signal", "value_complaint"]
ALL_LLM       = NON_POLARITY + POLARITY


def run_leakage_audit(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Fit the 4 variants and save the decomposition table."""
    if df is None:
        df = pd.read_parquet(FEATURES_PATH)
    df = _prepare_features(df).reset_index(drop=True)

    variants = {
        "baseline":        BEHAVIORAL,
        "non_polarity":    BEHAVIORAL + NON_POLARITY,
        "polarity_only":   BEHAVIORAL + POLARITY,
        "full_augmented":  BEHAVIORAL + ALL_LLM,
    }

    rows = {}
    fitted = {}
    for label, feats in variants.items():
        m = CoxSurvivalModel(label=label).fit(df, feats)
        cv = m.cv_cindex(df)
        rows[label] = {
            "n_features":  len(feats),
            "features":    feats,
            "cv_cindex":   cv,
        }
        fitted[label] = m

    base_mean = rows["baseline"]["cv_cindex"]["mean"]
    for label, row in rows.items():
        row["delta_vs_baseline"] = row["cv_cindex"]["mean"] - base_mean

    lr_results = {
        "polarity_only_vs_baseline":  fitted["polarity_only"].likelihood_ratio_test(fitted["baseline"]),
        "non_polarity_vs_baseline":   fitted["non_polarity"].likelihood_ratio_test(fitted["baseline"]),
        "full_vs_baseline":           fitted["full_augmented"].likelihood_ratio_test(fitted["baseline"]),
    }

    out = {
        "variants":    rows,
        "lr_tests":    lr_results,
        "decomposition": {
            "defensible_predictive_lift":  rows["non_polarity"]["delta_vs_baseline"],
            "polarity_lift":               rows["polarity_only"]["delta_vs_baseline"],
            "full_augmented_lift":         rows["full_augmented"]["delta_vs_baseline"],
        },
    }

    if save:
        with open(LEAKAGE_PATH, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Saved → {LEAKAGE_PATH}")
    return out


if __name__ == "__main__":
    run_leakage_audit()
