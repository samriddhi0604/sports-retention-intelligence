"""
Step 2 -- Hypothesis test (the headline finding).

H0: sports_spike_engagement has the same distribution for churned and
    retained users.
H1: retained users have higher sports_spike_engagement than churned users.

Normality is checked (Shapiro-Wilk) before picking a test rather than
defaulting to a t-test -- if either group looks non-normal, a Welch's
t-test's mean-comparison assumption is questionable, so a non-parametric
Mann-Whitney U test is used instead. Variance equality (Levene's test) is
checked too, in case a t-test path is taken, so Welch's correction is
applied only when actually needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_PATH = PROCESSED_DIR / "hypothesis_test_results.json"

ALPHA = 0.05


def run_hypothesis_test() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "user_features.csv")

    retained = df.loc[df["churned"] == 0, "sports_spike_engagement"]
    churned = df.loc[df["churned"] == 1, "sports_spike_engagement"]

    # Shapiro-Wilk on each group (subsampled if large -- the test is only
    # reliable up to a few thousand points, and we just need a normality
    # read, not an exact p-value on the full set).
    def shapiro_p(sample: pd.Series) -> float:
        s = sample.sample(n=min(len(sample), 4999), random_state=42) if len(sample) > 4999 else sample
        return float(stats.shapiro(s).pvalue)

    retained_normal_p = shapiro_p(retained)
    churned_normal_p = shapiro_p(churned)
    both_normal = retained_normal_p > ALPHA and churned_normal_p > ALPHA

    levene_p = float(stats.levene(retained, churned).pvalue)
    equal_var = levene_p > ALPHA

    if both_normal:
        test_name = "Welch's t-test" if not equal_var else "Student's t-test"
        result = stats.ttest_ind(retained, churned, equal_var=equal_var)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        test_name = "Mann-Whitney U test"
        result = stats.mannwhitneyu(retained, churned, alternative="greater")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    output = {
        "test_used": test_name,
        "reason": (
            "both groups passed Shapiro-Wilk normality check (p > 0.05)"
            if both_normal
            else "at least one group failed Shapiro-Wilk normality check (p <= 0.05), "
            "so a non-parametric test was used instead of assuming normality"
        ),
        "shapiro_p_retained": retained_normal_p,
        "shapiro_p_churned": churned_normal_p,
        "levene_p_equal_variance": levene_p,
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_alpha_0.05": p_value < ALPHA,
        "n_retained": int(len(retained)),
        "n_churned": int(len(churned)),
        "mean_sports_spike_engagement_retained": float(retained.mean()),
        "mean_sports_spike_engagement_churned": float(churned.mean()),
        "std_sports_spike_engagement_retained": float(retained.std()),
        "std_sports_spike_engagement_churned": float(churned.std()),
    }

    print("=== Hypothesis test: sports_spike_engagement, retained vs. churned ===")
    for key, value in output.items():
        print(f"  {key}: {value}")

    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n[hypothesis_test] wrote {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    run_hypothesis_test()
