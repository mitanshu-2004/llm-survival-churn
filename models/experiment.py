"""
Run the full experiment:
  1. Train Cox PH baseline (behavioral features only)
  2. Train Cox PH augmented (behavioral + LLM features)
  3. Compare via CV C-index + likelihood ratio test
  4. Save results to JSON
"""
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_PATH, MODEL_BASELINE, MODEL_AUGMENTED,
    EXPERIMENT_PATH, RANDOM_STATE
)
from models.cox import CoxSurvivalModel

# ── Feature sets ──────────────────────────────────────────────────────────
# NOTE: log_duration was removed, it IS the survival time (dependent variable).
# Using it as a predictor is circular and inflates C-index artificially.
# playtime_2wk_ratio was also removed, dividing by duration leaks the target.

BEHAVIORAL_FEATURES = [
    "log_playtime_2weeks",
    "log_items_count",
]

LLM_FEATURES = [
    "sentiment_score",
    "frustration_level",
    "technical_issue",
    "value_complaint",
    "engagement_dropped",
    "positive_signal",
]

AUGMENTED_FEATURES = BEHAVIORAL_FEATURES + LLM_FEATURES


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive missing behavioral columns from existing data (backward compat)."""
    if "log_playtime_2weeks" not in df.columns:
        if "playtime_2wk_ratio" in df.columns and "duration" in df.columns:
            # Reconstruct: playtime_2weeks ≈ ratio * duration
            pt2w = df["playtime_2wk_ratio"] * (df["duration"] + 1e-6)
            df["log_playtime_2weeks"] = np.log1p(pt2w)
        elif "playtime_2weeks" in df.columns:
            df["log_playtime_2weeks"] = np.log1p(df["playtime_2weeks"].fillna(0))
        else:
            df["log_playtime_2weeks"] = 0.0
    return df


HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED     = 1729  # different from RANDOM_STATE to keep CV/holdout entropy independent


def run_experiment(df: pd.DataFrame = None) -> dict:
    """
    Run full experiment. If df is None, loads from FEATURES_PATH.

    Reports three concordance numbers per model:
      - cv_cindex       : 5-fold CV on the 80% train slice (in-sample stability)
      - apparent_cindex : C-index on the full train slice  (in-sample optimism)
      - holdout_cindex  : C-index on the held-out 20% slice (out-of-sample)

    Saves the train/test indices to HOLDOUT_SPLIT_PATH so anyone can reproduce
    the split without rerunning the LLM extraction.
    """
    from sklearn.model_selection import train_test_split

    if df is None:
        print(f"Loading features from {FEATURES_PATH}...")
        df = pd.read_parquet(FEATURES_PATH)

    df = _prepare_features(df).reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows | Churn rate: {df['event'].mean():.1%}")

    # ── 80/20 stratified holdout ──────────────────────────────────────────
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=HOLDOUT_FRACTION,
        stratify=df["event"],
        random_state=HOLDOUT_SEED,
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} rows | Test: {len(df_test):,} rows (seed={HOLDOUT_SEED})\n")

    holdout_split_path = EXPERIMENT_PATH.parent / "holdout_split.json"
    with open(holdout_split_path, "w") as fh:
        json.dump(
            {
                "holdout_seed":     HOLDOUT_SEED,
                "holdout_fraction": HOLDOUT_FRACTION,
                "train_indices":    [int(i) for i in train_idx],
                "test_indices":     [int(i) for i in test_idx],
            },
            fh,
        )

    # ── Train models on the 80% slice ─────────────────────────────────────
    print("=" * 55)
    print("Training BASELINE model (behavioral features only)")
    print("=" * 55)
    baseline = CoxSurvivalModel(label="baseline")
    baseline.fit(df_train, BEHAVIORAL_FEATURES)

    print("\n" + "=" * 55)
    print("Training AUGMENTED model (behavioral + LLM features)")
    print("=" * 55)
    augmented = CoxSurvivalModel(label="augmented")
    augmented.fit(df_train, AUGMENTED_FEATURES)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Cross-validated C-index (train slice) + held-out C-index")
    print("=" * 55)
    baseline_cv  = baseline.cv_cindex(df_train)
    augmented_cv = augmented.cv_cindex(df_train)
    baseline_holdout  = baseline.cindex_on(df_test)
    augmented_holdout = augmented.cindex_on(df_test)
    baseline_apparent  = baseline.cindex_on(df_train)
    augmented_apparent = augmented.cindex_on(df_train)

    print(f"  Baseline  CV {baseline_cv['mean']:.4f} ± {baseline_cv['std']:.4f}"
          f"  | hold-out {baseline_holdout:.4f}"
          f"  | apparent {baseline_apparent:.4f}")
    print(f"  Augmented CV {augmented_cv['mean']:.4f} ± {augmented_cv['std']:.4f}"
          f"  | hold-out {augmented_holdout:.4f}"
          f"  | apparent {augmented_apparent:.4f}")

    delta_cv      = augmented_cv["mean"] - baseline_cv["mean"]
    delta_holdout = augmented_holdout    - baseline_holdout
    print(f"  Δ CV:      {delta_cv:+.4f}")
    print(f"  Δ holdout: {delta_holdout:+.4f}")

    # ── Likelihood ratio test ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Likelihood Ratio Test (LLM features add value?)")
    print("=" * 55)
    lr_test = augmented.likelihood_ratio_test(baseline)
    print(f"  LR statistic: {lr_test['lr_statistic']:.2f}")
    print(f"  df: {lr_test['df']}")
    print(f"  p-value: {lr_test['p_value']:.4f}")
    print(f"  log_p_value: {lr_test['log_p_value']:.2f} (natural log)")
    print(f"  Significant: {lr_test['significant']}")

    # ── Hazard ratios ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Augmented Model, Hazard Ratios")
    print("=" * 55)
    hr_df = augmented.hazard_ratios()
    print(hr_df.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "n_samples":       int(len(df)),
        "n_train":         int(len(df_train)),
        "n_test":          int(len(df_test)),
        "churn_rate":      float(df["event"].mean()),
        "holdout_seed":    HOLDOUT_SEED,
        "baseline": {
            "features":         BEHAVIORAL_FEATURES,
            "cv_cindex":        baseline_cv,
            "holdout_cindex":   float(baseline_holdout),
            "apparent_cindex":  float(baseline_apparent),
        },
        "augmented": {
            "features":         AUGMENTED_FEATURES,
            "cv_cindex":        augmented_cv,
            "holdout_cindex":   float(augmented_holdout),
            "apparent_cindex":  float(augmented_apparent),
        },
        "delta_cindex":         float(delta_cv),
        "delta_holdout_cindex": float(delta_holdout),
        "lr_test":              lr_test,
        "hazard_ratios":        hr_df.to_dict(orient="records"),
    }

    with open(EXPERIMENT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {EXPERIMENT_PATH}")
    print(f"  Hold-out split saved → {holdout_split_path}")

    baseline.save(MODEL_BASELINE)
    augmented.save(MODEL_AUGMENTED)

    return results
