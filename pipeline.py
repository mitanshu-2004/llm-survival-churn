"""
pipeline.py, Run the full research pipeline end-to-end.

Steps:
  1. Download McAuley Steam dataset
  2. Build survival dataset (join reviews + playtime)
  3. Extract LLM features via Groq (Llama-4-Scout-17B)
  4. Train baseline + augmented Cox models
  5. (optional) Run TF-IDF / SBERT / LLM ablation

Usage:
  python pipeline.py                        # full run (~35 min)
  python pipeline.py --skip-download        # if data already downloaded
  python pipeline.py --skip-llm             # if LLM features already extracted
  python pipeline.py --demo                 # fast 200-review demo (~3 min)
  python pipeline.py --ablation             # also run TF-IDF/SBERT/LLM ablation
  python pipeline.py --skip-experiment      # skip the Cox experiment (only useful with --ablation)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Augmented Survival Analysis Pipeline")
    parser.add_argument("--skip-download",  action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-llm",       action="store_true", help="Skip LLM feature extraction")
    parser.add_argument("--skip-experiment", action="store_true",
                        help="Skip the Cox baseline/augmented experiment")
    parser.add_argument("--ablation",       action="store_true",
                        help="Run TF-IDF / SBERT / LLM ablation after the experiment")
    parser.add_argument("--demo",           action="store_true",
                        help="Fast 200-review demo run (~3 min)")
    parser.add_argument("--max-reviews",    type=int, default=None,
                        help="Override MAX_REVIEWS env var (defaults to config.MAX_REVIEWS)")
    parser.add_argument("--synthetic",      action="store_true",
                        help="Offline smoke test on synthetic data (no dataset or Groq key needed)")
    args = parser.parse_args()

    # ── Offline smoke: prove the modelling pipeline runs with no external deps ──
    if args.synthetic:
        from synthetic import run_synthetic_smoke
        run_synthetic_smoke()
        return

    # Demo mode is a shorthand: smaller cap + skip download (assume cached)
    if args.demo:
        os.environ["MAX_REVIEWS"] = str(args.max_reviews or 200)
        args.skip_download = True
        print("→ Demo mode: MAX_REVIEWS=200, skipping data download")
    elif args.max_reviews is not None:
        os.environ["MAX_REVIEWS"] = str(args.max_reviews)

    # Late-import so MAX_REVIEWS override takes effect
    from config import FEATURES_PATH, ITEMS_PATH, REVIEWS_PATH, SURVIVAL_DF_PATH

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not args.skip_download:
        print("\n[1/5] Downloading McAuley Steam dataset...")
        from data.loader import download_all
        download_all()
    else:
        print("[1/5] Skipping download")
        if not REVIEWS_PATH.exists() or not ITEMS_PATH.exists():
            print("ERROR: Data files not found. Run without --skip-download first.")
            sys.exit(1)

    # ── Step 2: Build survival dataset ────────────────────────────────────
    if args.demo or not SURVIVAL_DF_PATH.exists():
        # In demo mode always rebuild so the row cap takes effect
        print("\n[2/5] Building survival dataset...")
        from data.builder import build_survival_dataset
        df = build_survival_dataset(save=not args.demo)
    else:
        print("\n[2/5] Survival dataset exists, loading...")
        import pandas as pd
        df = pd.read_parquet(SURVIVAL_DF_PATH)
        print(f"  {len(df):,} rows loaded")

    # ── Step 3: LLM feature extraction ────────────────────────────────────
    if args.demo:
        print("\n[3/5] Extracting LLM features (demo mode, no parquet cache)...")
        from features.llm_extractor import extract_llm_features
        df = extract_llm_features(df, save=False)
    elif not args.skip_llm and not FEATURES_PATH.exists():
        print("\n[3/5] Extracting LLM features...")
        from features.llm_extractor import extract_llm_features
        df = extract_llm_features(df, save=True)
    elif FEATURES_PATH.exists():
        print("\n[3/5] LLM features exist, loading...")
        import pandas as pd
        df = pd.read_parquet(FEATURES_PATH)
        print(f"  {len(df):,} rows with LLM features")
    else:
        print("\n[3/5] Skipping LLM extraction (--skip-llm)")

    # ── Step 4: Cox experiment ────────────────────────────────────────────
    results = None
    if not args.skip_experiment:
        print("\n[4/5] Running Cox PH baseline vs augmented...")
        from models.experiment import run_experiment
        results = run_experiment(df)
    else:
        print("[4/5] Skipping experiment (--skip-experiment)")

    # ── Step 5: Ablation ──────────────────────────────────────────────────
    if args.ablation:
        print("\n[5/5] Running TF-IDF / SBERT / LLM ablation...")
        from models.ablation import run_ablation
        ablation = run_ablation(df)
        print(f"  Ablation rows: {len(ablation['isolation']) + len(ablation['additive'])}")
    else:
        print("[5/5] Skipping ablation (use --ablation to enable)")

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    if results is not None:
        b_cv = results["baseline"]["cv_cindex"]
        a_cv = results["augmented"]["cv_cindex"]
        print(f"  Baseline  CV C-index: {b_cv['mean']:.4f} ± {b_cv['std']:.4f}")
        print(f"  Augmented CV C-index: {a_cv['mean']:.4f} ± {a_cv['std']:.4f}")
        print(f"  Augmented hold-out:   {results['augmented']['holdout_cindex']:.4f}")
        print(f"  Δ CV:                 {results['delta_cindex']:+.4f}")
        print(f"  LR log p-value:       {results['lr_test']['log_p_value']:.2f} (natural log)")
    print("\n  Results saved under results/.  See README.md for interpretation.")


if __name__ == "__main__":
    main()
