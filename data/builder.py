"""
Construct the survival analysis dataset.

Schema:
  user_id         str
  item_id         str
  duration        float   hours played (time-to-event / tenure)
  event           int     1 = churned (didn't recommend), 0 = censored (recommended)
  review_text     str     raw review for LLM extraction
  items_count     int     total games owned (behavioral covariate)
  playtime_2weeks float   recent activity (behavioral covariate)
  early_access    bool    game context
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    REVIEWS_PATH, ITEMS_PATH, SURVIVAL_DF_PATH,
    MAX_REVIEWS, MIN_DURATION, MAX_DURATION
)
from data.loader import load_reviews, load_items


def build_survival_dataset(save: bool = True) -> pd.DataFrame:
    print("Loading reviews...")
    reviews = load_reviews(REVIEWS_PATH)
    print(f"  {len(reviews):,} reviews with text")

    print("Loading items (playtime)...")
    items = load_items(ITEMS_PATH)
    print(f"  {len(items):,} user-item rows")

    # Join on user_id + item_id to get playtime for each reviewed game
    print("Joining reviews ↔ playtime...")
    df = reviews.merge(
        items[["user_id", "item_id", "playtime_forever", "playtime_2weeks", "items_count"]],
        on=["user_id", "item_id"],
        how="inner"
    )
    print(f"  {len(df):,} rows after join")

    # ── Survival variables ─────────────────────────────────────────────────
    df["duration"] = df["playtime_forever"]                   # hours played = time-to-event
    df["event"]    = (~df["recommend"]).astype(int)           # 1 = churned, 0 = censored

    # ── Filter ────────────────────────────────────────────────────────────
    df = df[(df["duration"] >= MIN_DURATION) & (df["duration"] <= MAX_DURATION)]
    df = df[df["review_text"].str.len() > 20]
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    print(f"  {len(df):,} rows after filtering (duration {MIN_DURATION}-{MAX_DURATION}h)")

    # ── Balance check ─────────────────────────────────────────────────────
    churn_rate = df["event"].mean()
    print(f"  Churn rate: {churn_rate:.1%}  ({df['event'].sum():,} churned / {(1-df['event']).sum():,} retained)")

    # ── Cap for LLM cost control ───────────────────────────────────────────
    if len(df) > MAX_REVIEWS:
        # Stratified sample to preserve churn rate. Clamp each stratum to the rows
        # actually available so an unusual churn rate can't request more retained
        # (or churned) rows than exist, which would raise "Cannot take a larger
        # sample than population".
        churned_pool  = df[df["event"] == 1]
        retained_pool = df[df["event"] == 0]
        n_churned  = min(int(MAX_REVIEWS * churn_rate), len(churned_pool))
        n_retained = min(MAX_REVIEWS - n_churned, len(retained_pool))
        churned  = churned_pool.sample(n_churned, random_state=42)
        retained = retained_pool.sample(n_retained, random_state=42)
        df = pd.concat([churned, retained]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Capped to {len(df):,} rows (stratified sample)")

    # ── Derived behavioral features ────────────────────────────────────────
    # NOTE: Do NOT use log_duration as a feature, it IS the survival time
    # (dependent variable). Using it as a predictor is circular.
    # Similarly, playtime_2wk_ratio = playtime_2weeks / duration leaks duration.
    df["log_playtime_2weeks"] = np.log1p(df["playtime_2weeks"].fillna(0))
    df["log_items_count"]     = np.log1p(df["items_count"].fillna(0))

    # Clean
    df = df.reset_index(drop=True)
    df = df[[
        "user_id", "item_id", "duration", "event", "review_text",
        "playtime_2weeks", "log_playtime_2weeks", "log_items_count",
    ]]

    if save:
        df.to_parquet(SURVIVAL_DF_PATH, index=False)
        print(f"  Saved → {SURVIVAL_DF_PATH}")

    return df
