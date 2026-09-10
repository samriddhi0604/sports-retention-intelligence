"""
Step 7 -- test that the hypothesis test function returns a sane p-value on
a known synthetic case with an obvious effect, before trusting it on the
real dataset (already exercised for real in analysis/hypothesis_test.py).
"""

import numpy as np
import pandas as pd

from analysis.hypothesis_test import compare_groups


def test_obvious_effect_is_detected_as_significant():
    rng = np.random.default_rng(0)
    retained = pd.Series(rng.normal(0.7, 0.05, size=200))
    churned = pd.Series(rng.normal(0.3, 0.05, size=200))

    result = compare_groups(retained, churned)

    assert result["p_value"] < 0.001
    assert result["significant_at_alpha_0.05"] is True
    assert result["mean_sports_spike_engagement_retained"] > result["mean_sports_spike_engagement_churned"]


def test_no_effect_is_not_significant():
    rng = np.random.default_rng(1)
    retained = pd.Series(rng.normal(0.5, 0.1, size=200))
    churned = pd.Series(rng.normal(0.5, 0.1, size=200))

    result = compare_groups(retained, churned)

    assert result["p_value"] > 0.05
    assert result["significant_at_alpha_0.05"] is False


def test_picks_nonparametric_test_for_non_normal_data():
    rng = np.random.default_rng(2)
    # exponential distributions are clearly non-normal -- should fail
    # Shapiro-Wilk and route to Mann-Whitney, not a t-test
    retained = pd.Series(rng.exponential(1.0, size=300))
    churned = pd.Series(rng.exponential(1.0, size=300))

    result = compare_groups(retained, churned)

    assert result["test_used"] == "Mann-Whitney U test"
