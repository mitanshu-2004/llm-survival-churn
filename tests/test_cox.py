"""Regression tests for the Cox PH wrapper, specifically the chi2 tail bug."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.cox import CoxSurvivalModel


def _make_dataset(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # x2 is informative for both duration and event so the LR test gets a real signal
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    baseline_hazard = 0.02 * np.exp(0.6 * x2)
    duration = rng.exponential(1.0 / baseline_hazard)
    event    = (duration < np.percentile(duration, 70)).astype(int)
    return pd.DataFrame({"duration": duration + 1e-3, "event": event, "x1": x1, "x2": x2})


def test_likelihood_ratio_test_returns_finite_log_p():
    """When LR stat is huge the raw p_value underflows to 0.0 in float64; log_p_value must stay finite."""
    df = _make_dataset()
    restricted = CoxSurvivalModel("restricted").fit(df, ["x1"])
    full       = CoxSurvivalModel("full").fit(df, ["x1", "x2"])

    result = full.likelihood_ratio_test(restricted)
    assert {"lr_statistic", "df", "p_value", "log_p_value", "significant"} <= result.keys()
    assert result["p_value"] >= 0.0
    assert math.isfinite(result["log_p_value"]), (
        "log_p_value should be finite even when p_value underflows to 0"
    )


def test_asymptotic_log_p_for_huge_stat():
    """Manually push lr_stat past the float64 chi2.sf tail; the asymptotic fallback must engage."""
    df = _make_dataset()
    m = CoxSurvivalModel("test").fit(df, ["x1", "x2"])

    # Synthesize a 'restricted' partner whose log-likelihood is much lower so the
    # difference fakes a colossal LR stat. We do this by monkey-poking the
    # log_likelihood_ attribute on a small clone.
    restricted = CoxSurvivalModel("low").fit(df, ["x1"])
    restricted.model.log_likelihood_ = m.model.log_likelihood_ - 1000.0   # diff = 2000 in LR space

    result = m.likelihood_ratio_test(restricted)
    assert result["lr_statistic"] > 1500.0
    assert result["p_value"] in (0.0, pytest.approx(0.0, abs=1e-200))
    assert math.isfinite(result["log_p_value"]), "Asymptotic fallback must produce a finite log-p"
    assert result["log_p_value"] < math.log(0.05), "A real-world huge LR stat must be flagged significant"
    assert result["significant"] is True


def test_cv_cindex_near_half_on_random_features():
    """C-index for a feature that contains no signal must hover near 0.5, guard against leakage."""
    rng = np.random.default_rng(11)
    n = 600
    df = pd.DataFrame({
        "duration": rng.exponential(40, n) + 1e-3,
        "event":    rng.binomial(1, 0.2, n),
        "noise":    rng.standard_normal(n),
    })
    m = CoxSurvivalModel("noise").fit(df, ["noise"])
    cv = m.cv_cindex(df)
    assert 0.40 <= cv["mean"] <= 0.60, f"C-index on noise was {cv['mean']:.3f} (expected ~0.5)"
