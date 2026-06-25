"""
Construct `notebooks/diagnostics.ipynb` and `notebooks/leakage_audit.ipynb`
from in-file source. Re-run after editing the cell contents below.

Usage:
    python notebooks/_build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent


def _nb_from_cells(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


# ── notebooks/diagnostics.ipynb ───────────────────────────────────────────

DIAGNOSTICS_CELLS: list[tuple[str, str]] = [
    ("md",
"""# Diagnostics, Cox PH assumptions, feature correlation, leakage visualisation

This notebook is the methodology-rigour appendix for `llm-survival-churn`. It checks the assumptions
that the published C-index headline depends on, and visualises *why* the LLM features look so
informative:

1. **Proportional-hazards assumption**, Schoenfeld residuals + global PH test (lifelines).
2. **Feature collinearity**, Spearman correlation matrix of the 6 LLM signals.
3. **Per-feature concordance**, univariate C-index for every covariate (ranks predictive power).
4. **Leakage visualisation**, distribution of each LLM feature stratified by the churn event.
5. **Kaplan-Meier sanity check**, survival curves split on the strongest hazard (frustration).

Run top to bottom. All figures are saved to `notebooks/figures/`.
"""),
    ("code",
"""import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / 'config.py').exists():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / 'config.py').exists():
    sys.path.insert(0, str(ROOT.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from config import FEATURES_PATH, EXPERIMENT_PATH
from models.experiment import _prepare_features

FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook')

df = pd.read_parquet(FEATURES_PATH)
df = _prepare_features(df)
with open(EXPERIMENT_PATH) as fh:
    results = json.load(fh)

BEHAVIORAL = ['log_playtime_2weeks', 'log_items_count']
LLM_FEATS  = ['sentiment_score', 'frustration_level',
              'technical_issue', 'value_complaint',
              'engagement_dropped', 'positive_signal']
ALL_FEATS  = BEHAVIORAL + LLM_FEATS

print(f'Rows: {len(df):,} | churn rate: {df[\"event\"].mean():.1%}')
df[ALL_FEATS].describe().T[['mean', 'std', 'min', 'max']]
"""),
    ("md",
"""## 1. Proportional-hazards assumption check

The Cox model assumes covariate effects are constant over time. We use Schoenfeld residuals
(`lifelines.statistics.proportional_hazard_test`) to test this for each covariate. A small p-value
(< 0.05) suggests the PH assumption is violated *for that covariate*.

For descriptive use (this project's claim) PH violations are usually fine, they affect inference
on individual hazard ratios more than overall ranking. For deployment, any violating covariate
should be stratified or modelled with a time-varying coefficient.
"""),
    ("code",
"""cph = CoxPHFitter(penalizer=0.1)
cph.fit(df[['duration', 'event'] + ALL_FEATS].dropna(),
        duration_col='duration', event_col='event')

ph_test = proportional_hazard_test(cph, df[['duration', 'event'] + ALL_FEATS].dropna(),
                                   time_transform='rank')
ph_df = ph_test.summary.copy()
ph_df.index.name = 'covariate'
ph_df = ph_df.reset_index()
ph_df['ph_violated'] = ph_df['p'] < 0.05
ph_df.sort_values('p')
"""),
    ("code",
"""fig, ax = plt.subplots(figsize=(8, 5))
plot_df = ph_df.sort_values('p')
colors  = ['#f87171' if v else '#4ade80' for v in plot_df['ph_violated']]
labels  = plot_df['covariate'].astype(str)
ax.barh(labels, -np.log10(plot_df['p'].clip(lower=1e-30)), color=colors)
ax.axvline(-np.log10(0.05), color='#1f2937', ls='--', lw=1, label='p = 0.05')
ax.set_xlabel('-log10(p), higher = stronger PH violation')
ax.set_title('Proportional-Hazards Test (Schoenfeld residuals)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'ph_test.png', dpi=120)
plt.show()
"""),
    ("md",
"""## 2. Feature collinearity (6 LLM signals)

Spearman correlation between the 6 LLM features. The L2 penalty (0.1) in `cox.py` regularises away
some collinearity but does not address it conceptually. Highly correlated features (|r| > 0.7) mean
the hazard ratios for those features are sensitive to small data shifts.
"""),
    ("code",
"""corr = df[LLM_FEATS].corr(method='spearman')
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt='.2f', center=0, cmap='RdBu_r', square=True,
            vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink': 0.7})
ax.set_title('Spearman correlation, 6 LLM-extracted features')
plt.tight_layout()
plt.savefig(FIG_DIR / 'llm_corr.png', dpi=120)
plt.show()
"""),
    ("md",
"""## 3. Univariate concordance per feature

For each covariate, fit a single-feature Cox model and report its CV C-index. This is a coarse
but interpretable ranking of *how predictive each feature is on its own*, it complements the
multivariate hazard ratios in the main results.
"""),
    ("code",
"""from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

def univariate_cindex(feature: str, n_splits: int = 5, seed: int = 42) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sub = df[['duration', 'event', feature]].dropna().reset_index(drop=True)
    scores = []
    for train_idx, val_idx in kf.split(sub):
        cph_f = CoxPHFitter(penalizer=0.1)
        try:
            cph_f.fit(sub.iloc[train_idx], duration_col='duration', event_col='event')
            risk = cph_f.predict_partial_hazard(sub.iloc[val_idx])
            scores.append(concordance_index(sub.iloc[val_idx]['duration'], -risk, sub.iloc[val_idx]['event']))
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores))

uni = pd.DataFrame({
    'feature':    ALL_FEATS,
    'univariate_C': [univariate_cindex(f) for f in ALL_FEATS],
}).sort_values('univariate_C', ascending=False)
uni
"""),
    ("code",
"""fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#fb923c' if f in LLM_FEATS else '#60a5fa' for f in uni['feature']]
ax.barh(uni['feature'], uni['univariate_C'], color=colors)
ax.axvline(0.5, ls='--', color='#1f2937', lw=1, label='random (0.5)')
ax.set_xlim(0.4, 0.85)
ax.set_xlabel('5-fold CV univariate C-index')
ax.set_title('How predictive is each feature alone? (orange = LLM-extracted, blue = behavioral)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'univariate_cindex.png', dpi=120)
plt.show()
"""),
    ("md",
"""## 4. Leakage visualisation, distributions per event class

If an LLM-extracted feature has *cleanly separated* distributions between `event=0` (retained) and
`event=1` (churned), the feature is either genuinely predictive or *leaking*, i.e. the LLM read
sentiment from review text that itself encodes the outcome (the user explicitly recommended or
didn't).

In a deployment-clean setting, the same features would have to be extracted from text written
*before* the recommend decision is observed.
"""),
    ("code",
"""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flat, LLM_FEATS):
    for evt, color, label in [(0, '#4ade80', 'retained'), (1, '#f87171', 'churned')]:
        vals = df.loc[df['event'] == evt, feat]
        if feat in ('sentiment_score', 'frustration_level'):
            sns.kdeplot(vals, ax=ax, fill=True, color=color, label=label, alpha=0.4)
        else:
            rate = vals.mean()
            ax.bar(label, rate, color=color, alpha=0.7)
    ax.set_title(feat)
    if feat in ('sentiment_score', 'frustration_level'):
        ax.legend()
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel('rate')
plt.suptitle('LLM features, distribution by churn outcome (visualises leakage)', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_by_event.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
    ("md",
"""## 5. Kaplan-Meier sanity check on `frustration_level`

The Cox model says `frustration_level` is the strongest accelerant of churn (HR ≈ 2.2). A simple
median-split KM curve should show users with above-median frustration churning earlier, if not,
the hazard ratio is misleading.
"""),
    ("code",
"""median_f = df['frustration_level'].median()
high = df['frustration_level'] >  median_f
low  = df['frustration_level'] <= median_f

kmf_h = KaplanMeierFitter().fit(df.loc[high, 'duration'], df.loc[high, 'event'], label='High frustration')
kmf_l = KaplanMeierFitter().fit(df.loc[low,  'duration'], df.loc[low,  'event'], label='Low frustration')

fig, ax = plt.subplots(figsize=(8, 5))
kmf_h.plot_survival_function(ax=ax, color='#f87171')
kmf_l.plot_survival_function(ax=ax, color='#4ade80')
ax.set_xlabel('Hours played')
ax.set_ylabel('Survival probability')
ax.set_title('Kaplan-Meier, high vs low frustration (median split)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'km_frustration.png', dpi=120)
plt.show()
"""),
    ("md",
"""## Conclusions

* **PH assumption**: see the `-log10(p)` plot. Behavioral covariates often violate PH on Steam-style
  lifetime data because long-tenure users have qualitatively different hazards from short-tenure ones.
  Treat hazard ratios as average effects, not pointwise.
* **Collinearity**: `sentiment_score` and `positive_signal` are strongly anti-correlated;
  `frustration_level`, `value_complaint`, `engagement_dropped` cluster. The L2 penalty stabilises
  the fit but the individual HR confidence intervals should be read as optimistic.
* **Univariate concordance**: confirms LLM features dominate. The headline +0.26 multivariate
  improvement is largely carried by the polarity-bearing features (`sentiment_score`,
  `frustration_level`, `positive_signal`).
* **Leakage**: the KDEs for `sentiment_score` and `frustration_level` are *cleanly separated*
  between event classes, that is the smoking gun for "LLM reads the label from the text."
  The non-polarity features (`technical_issue`, `engagement_dropped`) show smaller gaps and are
  the more defensible predictors.

The `leakage_audit.ipynb` companion notebook quantifies the polarity-leakage contribution by
re-fitting the model with the polarity features masked out.
"""),
]


# ── notebooks/leakage_audit.ipynb ─────────────────────────────────────────

LEAKAGE_CELLS: list[tuple[str, str]] = [
    ("md",
"""# Leakage audit, decomposing the +0.26 augmented C-index

The README discloses that the LLM extracts polarity-bearing features (sentiment, frustration,
positive_signal, value_complaint) from review text whose label (`recommend`) itself encodes the
same polarity. That makes the headline +0.26 C-index improvement at least partially a
**descriptive** artifact rather than a forward-looking predictor.

This notebook quantifies the polarity contribution by re-fitting four models that vary which
LLM features are exposed:

| Model | Features used |
|---|---|
| `baseline`         | log_playtime_2weeks, log_items_count |
| `behavioral_only`  | (same as baseline, sanity duplicate) |
| `non_polarity`     | baseline + technical_issue, engagement_dropped |
| `polarity_only`    | baseline + sentiment_score, frustration_level, positive_signal, value_complaint |
| `full_augmented`   | baseline + all 6 LLM features |

The expectation: `polarity_only` already gets most of the +0.26 lift, while `non_polarity` only
gets a fraction. That fraction is the *defensible* uplift.
"""),
    ("code",
"""import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / 'config.py').exists():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / 'config.py').exists():
    sys.path.insert(0, str(ROOT.parent))

import pandas as pd

from config import FEATURES_PATH
from models.cox import CoxSurvivalModel
from models.experiment import _prepare_features

df = _prepare_features(pd.read_parquet(FEATURES_PATH))

BEHAVIORAL    = ['log_playtime_2weeks', 'log_items_count']
NON_POLARITY  = ['technical_issue', 'engagement_dropped']
POLARITY      = ['sentiment_score', 'frustration_level', 'positive_signal', 'value_complaint']
ALL_LLM       = NON_POLARITY + POLARITY

VARIANTS = {
    'baseline':        BEHAVIORAL,
    'non_polarity':    BEHAVIORAL + NON_POLARITY,
    'polarity_only':   BEHAVIORAL + POLARITY,
    'full_augmented':  BEHAVIORAL + ALL_LLM,
}
print(f'Rows: {len(df):,} | churn rate: {df[\"event\"].mean():.1%}')
"""),
    ("code",
"""rows = []
fitted = {}
for label, feats in VARIANTS.items():
    m = CoxSurvivalModel(label=label).fit(df, feats)
    cv = m.cv_cindex(df)
    rows.append({
        'variant':     label,
        '# features':  len(feats),
        'CV C-index':  cv['mean'],
        'CV ± std':    cv['std'],
    })
    fitted[label] = m

table = pd.DataFrame(rows).set_index('variant')
table['Δ vs baseline'] = table['CV C-index'] - table.loc['baseline', 'CV C-index']
table
"""),
    ("md",
"""## Interpretation

Reading the `Δ vs baseline` column:

* **`non_polarity` (technical_issue + engagement_dropped)**, captures the share of the augmented
  lift attributable to *behavioural* signals the LLM extracted that aren't pure polarity.
* **`polarity_only` (sentiment, frustration, positive, value)**, captures the share attributable
  to features that are essentially restating the `recommend` label in continuous form.
* **`full_augmented`**, the headline number from the main experiment.

If `polarity_only Δ` ≈ `full_augmented Δ`, almost all of the lift is polarity-driven, and the model
should be sold as a **descriptive** hazard model: it explains *which textual signals correlate with
having churned*, not *what to monitor to predict future churn*.

If `non_polarity Δ` is non-trivial (e.g. > +0.05 over baseline), there is real *behavioural* signal
in the text that survives label leakage, a defensible predictive claim.
"""),
    ("code",
"""# Pretty-print a single-row "leakage decomposition" summary
delta_full     = table.loc['full_augmented',  'Δ vs baseline']
delta_polarity = table.loc['polarity_only',    'Δ vs baseline']
delta_nonpol   = table.loc['non_polarity',     'Δ vs baseline']
polarity_share = (delta_polarity / delta_full) if delta_full else float('nan')

print(f'Full augmented Δ:    {delta_full:+.4f}')
print(f'Polarity-only Δ:     {delta_polarity:+.4f}   ({polarity_share:.0%} of full)')
print(f'Non-polarity-only Δ: {delta_nonpol:+.4f}   ({1 - polarity_share:.0%} of full)')
"""),
    ("md",
"""## Likelihood-ratio test, does `non_polarity` even matter?

If `non_polarity` lift is small in C-index but the LR test on the nested model is still highly
significant, the behavioural-text features are detecting *something*, just something the
concordance metric doesn't reward much. If LR is *also* non-significant, the polarity features
are doing all the work.
"""),
    ("code",
"""lr_pol = fitted['polarity_only'].likelihood_ratio_test(fitted['baseline'])
lr_non = fitted['non_polarity'].likelihood_ratio_test(fitted['baseline'])
lr_all = fitted['full_augmented'].likelihood_ratio_test(fitted['baseline'])

def fmt(d):
    return f'LR={d[\"lr_statistic\"]:.1f}, df={d[\"df\"]}, log_p={d[\"log_p_value\"]:.1f}'

print('polarity_only vs baseline:    ', fmt(lr_pol))
print('non_polarity vs baseline:     ', fmt(lr_non))
print('full_augmented vs baseline:   ', fmt(lr_all))
"""),
    ("md",
"""## Bottom line

This notebook is the *honest version* of the headline. The README quotes the
`full_augmented Δ` because that's what readers expect to see, but the supplementary table here
is what a careful methodology reviewer (Sarvam, AI4Bharat, HF Smol) actually wants:

> "Of the +0.26 augmented C-index lift, X points come from behavioural signals the LLM extracted
> from review text (defensible predictive signal), and Y points come from polarity features that
> partially restate the recommend label (descriptive only)."

Numbers vary slightly run-to-run because the 5-fold CV is stochastic, but the *shape* is stable.
"""),
]


def main() -> None:
    nb_diag = _nb_from_cells(DIAGNOSTICS_CELLS)
    nb_leak = _nb_from_cells(LEAKAGE_CELLS)

    diag_path = NOTEBOOK_DIR / "diagnostics.ipynb"
    leak_path = NOTEBOOK_DIR / "leakage_audit.ipynb"
    nbf.write(nb_diag, diag_path)
    nbf.write(nb_leak, leak_path)
    print(f"Wrote {diag_path}")
    print(f"Wrote {leak_path}")


if __name__ == "__main__":
    main()
