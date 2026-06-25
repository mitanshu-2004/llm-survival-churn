"""The synthetic smoke runs the Cox modelling offline (no dataset, no Groq) and the
augmented model should recover the planted signal. Run with: pytest"""
from synthetic import make_synthetic_survival, run_synthetic_smoke
from models.experiment import AUGMENTED_FEATURES


def test_synthetic_schema_matches_model_features():
    df = make_synthetic_survival(n=50, seed=0)
    for col in ["duration", "event", *AUGMENTED_FEATURES]:
        assert col in df.columns
    assert (df["duration"] > 0).all()
    assert set(df["event"].unique()) <= {0, 1}


def test_augmented_recovers_planted_signal():
    res = run_synthetic_smoke(n=1500, seed=0)
    # The planted risk is real, so concordance should beat chance, and the
    # LLM-augmented model should not be worse than the behavioural baseline.
    assert res["augmented_holdout_cindex"] > 0.6
    assert res["augmented_holdout_cindex"] >= res["baseline_holdout_cindex"] - 0.02
