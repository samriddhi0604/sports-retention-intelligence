"""
Step 7 -- tests for the gateway analysis functions on small hand-constructed
examples with a known expected answer, before trusting them on the full
simulated session log (already exercised for real in
recommendations/gateway_analysis.py).
"""

import pandas as pd
import pytest

from recommendations.gateway_analysis import classify_ramp_up, compute_lift


def test_classify_ramp_up_known_case():
    midpoint = pd.Timestamp("2024-02-01")
    sessions = pd.DataFrame(
        {
            "user_id": (
                [1] * 3 + [1] * 9  # user 1: 3 early, 9 late -> ramped up (9 > 2*3)
                + [2] * 4 + [2] * 5  # user 2: 4 early, 5 late -> not ramped up (5 <= 2*4)
                + [3] * 2 + [3] * 10  # user 3: only 2 early -> ineligible (below min_first_half=3)
            ),
            "timestamp": (
                [pd.Timestamp("2024-01-10")] * 3 + [pd.Timestamp("2024-02-10")] * 9
                + [pd.Timestamp("2024-01-10")] * 4 + [pd.Timestamp("2024-02-10")] * 5
                + [pd.Timestamp("2024-01-10")] * 2 + [pd.Timestamp("2024-02-10")] * 10
            ),
        }
    )

    result = classify_ramp_up(sessions, midpoint, min_first_half=3, multiplier=2.0).set_index("user_id")

    assert result.loc[1, "eligible"] == True  # noqa: E712
    assert result.loc[1, "ramped_up"] == True  # noqa: E712
    assert result.loc[2, "eligible"] == True  # noqa: E712
    assert result.loc[2, "ramped_up"] == False  # noqa: E712
    assert result.loc[3, "eligible"] == False  # noqa: E712
    assert result.loc[3, "ramped_up"] == False  # noqa: E712


def test_compute_lift_known_case():
    # "A" appears disproportionately in the ramped-up group; "B" is common
    # to both in similar proportion
    ramped = pd.Series(["A"] * 8 + ["B"] * 2)
    never = pd.Series(["A"] * 2 + ["B"] * 8)

    lift = compute_lift(ramped, never, min_occurrences=1, smoothing=0.0)
    lift = lift.set_index("category")

    assert lift.loc["A", "lift"] > 1.0
    assert lift.loc["B", "lift"] < 1.0
    # A's lift should be exactly the inverse ratio structure: (8/10)/(2/10) = 4.0
    assert lift.loc["A", "lift"] == pytest.approx(4.0)


def test_compute_lift_respects_min_occurrences():
    ramped = pd.Series(["A"] * 2 + ["B"] * 8)
    never = pd.Series(["A"] * 5 + ["B"] * 5)

    lift = compute_lift(ramped, never, min_occurrences=3)
    assert "A" not in set(lift["category"])  # only 2 occurrences in ramped, below the min
    assert "B" in set(lift["category"])
