"""
Cox Proportional Hazards model wrapper.

Wraps lifelines.CoxPHFitter with:
  - Cross-validated C-index
  - Held-out concordance via :py:meth:`cindex_on`
  - Likelihood-ratio test against a nested restricted model
  - Hazard ratio table with 95% CIs
  - Survival curve prediction for new observations
  - Serialization via joblib
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CV_FOLDS, RANDOM_STATE, PENALIZER


class CoxSurvivalModel:
    """
    Thin wrapper around lifelines.CoxPHFitter that adds CV evaluation,
    Brier score, and clean hazard ratio reporting.
    """

    DURATION_COL = "duration"
    EVENT_COL    = "event"

    def __init__(self, label: str, penalizer: float = PENALIZER):
        self.label      = label
        self.penalizer  = penalizer
        self.model      = CoxPHFitter(penalizer=penalizer)
        self.feature_cols: list[str] = []
        self.is_fitted   = False

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "CoxSurvivalModel":
        self.feature_cols = feature_cols
        train_df = df[[self.DURATION_COL, self.EVENT_COL] + feature_cols].dropna()
        self.model.fit(
            train_df,
            duration_col=self.DURATION_COL,
            event_col=self.EVENT_COL,
        )
        self.is_fitted = True
        print(f"[{self.label}] Fitted on {len(train_df):,} rows, {len(feature_cols)} features")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────

    def cv_cindex(self, df: pd.DataFrame) -> dict:
        """5-fold CV concordance index."""
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        subset = df[[self.DURATION_COL, self.EVENT_COL] + self.feature_cols].dropna()
        X = subset.reset_index(drop=True)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_model = CoxPHFitter(penalizer=self.penalizer)
            fold_model.fit(X.iloc[train_idx], duration_col=self.DURATION_COL, event_col=self.EVENT_COL)
            risk = fold_model.predict_partial_hazard(X.iloc[val_idx])
            ci = concordance_index(
                X.iloc[val_idx][self.DURATION_COL],
                -risk,
                X.iloc[val_idx][self.EVENT_COL]
            )
            scores.append(ci)

        return {
            "mean":   float(np.mean(scores)),
            "std":    float(np.std(scores)),
            "folds":  scores,
        }

    def cindex_on(self, df: pd.DataFrame) -> float:
        """C-index on a held-out set (or full set for reporting)."""
        subset = df[[self.DURATION_COL, self.EVENT_COL] + self.feature_cols].dropna()
        risk = self.model.predict_partial_hazard(subset)
        return concordance_index(subset[self.DURATION_COL], -risk, subset[self.EVENT_COL])

    def hazard_ratios(self) -> pd.DataFrame:
        """Return hazard ratios with 95% CI, sorted by abs(log HR)."""
        summary = self.model.summary.copy()
        hr_df = pd.DataFrame({
            "feature":    summary.index,
            "HR":         np.exp(summary["coef"]),
            "HR_lower":   np.exp(summary["coef lower 95%"]),
            "HR_upper":   np.exp(summary["coef upper 95%"]),
            "p_value":    summary["p"],
            "significant": summary["p"] < 0.05,
        }).reset_index(drop=True)
        hr_df["abs_log_hr"] = np.abs(np.log(hr_df["HR"]))
        return hr_df.sort_values("abs_log_hr", ascending=False).drop("abs_log_hr", axis=1)

    def likelihood_ratio_test(self, restricted_model: "CoxSurvivalModel") -> dict:
        """
        Likelihood ratio test: self (full) vs restricted_model (nested).
        H0: additional features in self add no predictive value.
        """
        import math
        from scipy import stats
        ll_full  = self.model.log_likelihood_
        ll_restr = restricted_model.model.log_likelihood_
        lr_stat  = float(2 * (ll_full - ll_restr))
        df_diff  = int(len(self.feature_cols) - len(restricted_model.feature_cols))

        # chi2.sf gives 0.0 once the tail probability falls below float64 (~1e-308).
        # chi2.logsf often returns -inf for the same regime. Fall back to the
        # asymptotic upper-tail expansion so we keep a reportable log-p:
        #   log P(X > x | df=k) ≈ (k/2 - 1)·log(x) - x/2 - (k/2)·log(2) - lgamma(k/2)
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff))
        log_p   = float(stats.chi2.logsf(lr_stat, df=df_diff))
        if not math.isfinite(log_p):
            log_p = (
                (df_diff / 2.0 - 1.0) * math.log(lr_stat)
                - lr_stat / 2.0
                - (df_diff / 2.0) * math.log(2.0)
                - math.lgamma(df_diff / 2.0)
            )
        return {
            "lr_statistic":  lr_stat,
            "df":            df_diff,
            "p_value":       p_value,
            "log_p_value":   log_p,
            "significant":   bool(log_p < math.log(0.05)),
        }

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_survival(self, obs: dict, times: np.ndarray = None) -> pd.DataFrame:
        """
        Given a dict of feature values, return survival curve S(t).
        Returns DataFrame with columns: time, survival_prob.
        """
        row = pd.DataFrame([obs])[self.feature_cols]
        sf  = self.model.predict_survival_function(row)
        if times is None:
            times = sf.index.values
        probs = np.interp(times, sf.index.values, sf.iloc[:, 0].values)
        return pd.DataFrame({"time": times, "survival_prob": probs})

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        print(f"  Model saved → {path}")

    @staticmethod
    def load(path: Path) -> "CoxSurvivalModel":
        return joblib.load(path)
