"""
Multi-encoder ablation study.

Builds 4 text representations for the same 10K Steam reviews:

  * (none)               , behavioural-only baseline
  * TF-IDF (TruncatedSVD-64)
  * SBERT  (PCA-64 of all-MiniLM-L6-v2 embeddings)
  * LLM    (6 Pydantic-validated structured signals from Groq + Llama-4)

…then fits Cox PH with each (and a final "combined" row that stacks SBERT + LLM)
under an 80/20 stratified hold-out. Reports both an **isolation** table (each
encoder alone) and an **additive** table (encoders stacked).

Run:
    python -m models.ablation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES_PATH, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from features.sbert_baseline import add_sbert_features  # noqa: E402
from features.tfidf_baseline import add_tfidf_features  # noqa: E402
from models.cox import CoxSurvivalModel  # noqa: E402
from models.experiment import _prepare_features  # noqa: E402

ABLATION_PATH    = RESULTS_DIR / "ablation_results.json"
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED     = 1729

BEHAVIORAL = ["log_playtime_2weeks", "log_items_count"]
LLM_FEATS  = [
    "sentiment_score", "frustration_level",
    "technical_issue", "value_complaint",
    "engagement_dropped", "positive_signal",
]


def _eval_variant(label: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                  feature_cols: list[str]) -> dict:
    m  = CoxSurvivalModel(label=label).fit(df_train, feature_cols)
    cv = m.cv_cindex(df_train)
    return {
        "label":            label,
        "n_features":       len(feature_cols),
        "features":         feature_cols,
        "cv_cindex":        cv,
        "holdout_cindex":   float(m.cindex_on(df_test)),
        "apparent_cindex":  float(m.cindex_on(df_train)),
    }


def run_ablation(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(FEATURES_PATH)
    df = _prepare_features(df).reset_index(drop=True)

    # 80/20 stratified hold-out (same split convention as run_experiment)
    train_idx, test_idx = train_test_split(
        df.index, test_size=HOLDOUT_FRACTION,
        stratify=df["event"], random_state=HOLDOUT_SEED,
    )
    train_mask = df.index.isin(train_idx)

    # Build TF-IDF and SBERT features (fit on training mask only, no leakage).
    # The dim budget is 64 each, so the dense baseline isn't being starved of
    # predictive dimensions while the LLM gets its 6 prompt-targeted features.
    print("Building TF-IDF features…")
    df, _ = add_tfidf_features(df, train_mask=train_mask, n_components=64, save=False)
    print("Building SBERT features…")
    df, _ = add_sbert_features(df, train_mask=train_mask, n_components=64, save=False)

    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    sbert_cols = [c for c in df.columns if c.startswith("sbert_")]

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}\n")

    # ── Isolation table: each encoder alone, with the behavioural floor ──
    print("=" * 55)
    print("Isolation ablation, each text encoder alone (+ behavioural)")
    print("=" * 55)
    isolation = {
        "Baseline":        _eval_variant("Baseline",        df_train, df_test, BEHAVIORAL),
        "TF-IDF only":     _eval_variant("TF-IDF",          df_train, df_test, BEHAVIORAL + tfidf_cols),
        "SBERT only":      _eval_variant("SBERT",           df_train, df_test, BEHAVIORAL + sbert_cols),
        "LLM only":        _eval_variant("LLM",             df_train, df_test, BEHAVIORAL + LLM_FEATS),
    }
    for name, entry in isolation.items():
        cv = entry["cv_cindex"]
        print(f"  {name:14s} CV {cv['mean']:.4f} ± {cv['std']:.4f}  | hold-out {entry['holdout_cindex']:.4f}")

    # ── Additive table: stacked combinations ─────────────────────────────
    print("\n" + "=" * 55)
    print("Additive ablation, stacked text encoders (+ behavioural)")
    print("=" * 55)
    additive = {
        "Baseline":          isolation["Baseline"],
        "+TF-IDF":           isolation["TF-IDF only"],
        "+TF-IDF +SBERT":    _eval_variant("+TF-IDF +SBERT",    df_train, df_test,
                                           BEHAVIORAL + tfidf_cols + sbert_cols),
        "+SBERT +LLM":       _eval_variant("+SBERT +LLM",       df_train, df_test,
                                           BEHAVIORAL + sbert_cols + LLM_FEATS),
        "+TF-IDF +SBERT +LLM": _eval_variant("+TF-IDF +SBERT +LLM", df_train, df_test,
                                             BEHAVIORAL + tfidf_cols + sbert_cols + LLM_FEATS),
    }
    for name, entry in additive.items():
        cv = entry["cv_cindex"]
        print(f"  {name:22s} CV {cv['mean']:.4f} ± {cv['std']:.4f}  | hold-out {entry['holdout_cindex']:.4f}")

    out = {
        "n_train":   int(len(df_train)),
        "n_test":    int(len(df_test)),
        "holdout_seed": HOLDOUT_SEED,
        "isolation": isolation,
        "additive":  additive,
    }
    with open(ABLATION_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved → {ABLATION_PATH}")
    return out


if __name__ == "__main__":
    run_ablation()
