"""
Step 7 -- test the spike detector on a synthetic series with a known
injected spike, before trusting it on the real Wikipedia series (already
exercised for real in analysis/spike_detection.py).
"""

import numpy as np
import pandas as pd

from analysis.spike_detection import detect_spikes


def _make_synthetic_series(n_weeks: int = 8, spike_date: str | None = None, spike_multiplier: float = 10.0):
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    rng = np.random.default_rng(0)
    # stable baseline with a mild day-of-week pattern and small noise
    day_of_week_base = {0: 100, 1: 100, 2: 100, 3: 100, 4: 120, 5: 150, 6: 150}
    views = np.array([day_of_week_base[d.dayofweek] for d in dates], dtype=float)
    views += rng.normal(0, 3, size=len(dates))

    df = pd.DataFrame({"date": dates, "views": views})
    if spike_date is not None:
        idx = df.index[df["date"] == pd.Timestamp(spike_date)][0]
        df.loc[idx, "views"] *= spike_multiplier
    return df


def test_injected_spike_is_flagged():
    df = _make_synthetic_series(spike_date="2024-02-10", spike_multiplier=10.0)
    scored = detect_spikes(df)

    spike_row = scored.loc[scored["date"] == pd.Timestamp("2024-02-10")].iloc[0]
    assert spike_row["is_flagged"] == True  # noqa: E712


def test_normal_days_are_not_flagged():
    df = _make_synthetic_series(spike_date="2024-02-10", spike_multiplier=10.0)
    scored = detect_spikes(df)

    non_spike_days = scored[scored["date"] != pd.Timestamp("2024-02-10")]
    false_positive_rate = non_spike_days["is_flagged"].mean()
    assert false_positive_rate < 0.05


def test_no_spike_series_flags_almost_nothing():
    df = _make_synthetic_series(spike_date=None)
    scored = detect_spikes(df)
    assert scored["is_flagged"].mean() < 0.05
