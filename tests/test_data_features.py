"""
Step 1d -- unit tests for the pure feature-computation functions in
data/features.py, checked against small hand-constructed examples where the
expected answer is independently computed (not just re-deriving the
implementation), before trusting these on the full simulated dataset.
"""

import math

import numpy as np
import pandas as pd
import pytest

from data.features import (
    compute_avg_completion_rate,
    compute_genre_diversity,
    compute_sports_spike_engagement,
)


def test_sports_spike_engagement_known_case():
    session_dates = pd.Series(
        pd.to_datetime(["2024-03-22", "2024-03-22", "2024-03-23", "2024-04-01"])
    )
    match_days = {pd.Timestamp("2024-03-22"), pd.Timestamp("2024-04-01")}
    # 3 of 4 sessions fall on a match day
    assert compute_sports_spike_engagement(session_dates, match_days) == pytest.approx(0.75)


def test_sports_spike_engagement_no_matches():
    session_dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    match_days = {pd.Timestamp("2024-03-22")}
    assert compute_sports_spike_engagement(session_dates, match_days) == 0.0


def test_sports_spike_engagement_empty_is_nan():
    assert np.isnan(compute_sports_spike_engagement(pd.Series([], dtype="datetime64[ns]"), set()))


def test_genre_diversity_known_case():
    genre_lists = [["Action"], ["Action"], ["Drama"], ["Comedy"]]
    all_genres = ["Action", "Comedy", "Drama"]

    # independently computed expected value, not via the function under test
    counts = {"Action": 2, "Drama": 1, "Comedy": 1}
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    expected_entropy = -sum(p * math.log(p) for p in probs)
    expected = expected_entropy / math.log(len(all_genres))

    assert compute_genre_diversity(genre_lists, all_genres) == pytest.approx(expected)


def test_genre_diversity_single_genre_is_zero():
    genre_lists = [["Action"], ["Action"], ["Action"]]
    all_genres = ["Action", "Comedy", "Drama"]
    # only ever watching one genre -> zero diversity
    assert compute_genre_diversity(genre_lists, all_genres) == pytest.approx(0.0)


def test_genre_diversity_empty_is_nan():
    assert np.isnan(compute_genre_diversity([], ["Action", "Comedy"]))


def test_avg_completion_rate_known_case():
    ratings = pd.Series([8.0, 6.0, 10.0])
    assert compute_avg_completion_rate(ratings) == pytest.approx(0.8)


def test_avg_completion_rate_empty_is_nan():
    assert np.isnan(compute_avg_completion_rate(pd.Series([], dtype=float)))
