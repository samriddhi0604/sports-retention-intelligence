"""
Step 3 -- Viewership spike detection.

Detects anomalous days in the daily Wikipedia pageview series (the
viewership proxy -- see data/README.md) using a seasonal baseline: the
median and MAD (median absolute deviation) of views for each day-of-week,
computed across the full 182-day window. A naive rolling window is avoided
deliberately -- IPL 2024 has matches on ~91% of days within the ~66-day
season itself, so a local rolling window falling inside the season would be
dominated by the spikes it's supposed to be measuring against, exactly the
distortion CLAUDE.md flags as a real bug from the prototype build. The
day-of-week median is far more robust to this: each weekday occurs ~26
times across the full window, and only ~9-10 of those occurrences fall
within the season, so the off-season majority anchors a realistic baseline
even for weekdays cricket is often played on.

Evaluated at DAY granularity, not hour granularity -- the Wikimedia REST API
only exposes daily/monthly granularity per article (see data/README.md), so
"recall"/"precision" here mean event DAYS caught / flagged, not event hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_PATH = PROCESSED_DIR / "spike_detection_results.json"

MODIFIED_Z_THRESHOLD = 3.5  # standard robust-outlier convention (Iglewicz & Hoaglin)


def detect_spikes(wiki: pd.DataFrame) -> pd.DataFrame:
    """Returns wiki with baseline/mad/modified_z/is_flagged columns added."""
    df = wiki.copy()
    df["day_of_week"] = df["date"].dt.dayofweek

    baseline = df.groupby("day_of_week")["views"].median().rename("baseline")
    mad = (
        df.merge(baseline, on="day_of_week")
        .assign(abs_dev=lambda d: (d["views"] - d["baseline"]).abs())
        .groupby("day_of_week")["abs_dev"]
        .median()
        .rename("mad")
    )

    df = df.merge(baseline, on="day_of_week").merge(mad, on="day_of_week")
    # guard against a zero MAD (a weekday with perfectly consistent views)
    # producing a divide-by-zero -- treat any deviation from baseline as
    # infinitely anomalous in that case rather than crashing.
    safe_mad = df["mad"].replace(0, np.nan)
    df["modified_z"] = 0.6745 * (df["views"] - df["baseline"]) / safe_mad
    fallback = pd.Series(
        np.where(df["views"] > df["baseline"], np.inf, 0.0), index=df.index
    )
    df["modified_z"] = df["modified_z"].fillna(fallback)
    df["is_flagged"] = df["modified_z"] > MODIFIED_Z_THRESHOLD

    return df.sort_values("date").reset_index(drop=True)


def evaluate(df: pd.DataFrame, match_days: set) -> dict:
    df = df.copy()
    df["is_match_day"] = df["date"].dt.normalize().isin(match_days)

    true_positives = int((df["is_flagged"] & df["is_match_day"]).sum())
    flagged = int(df["is_flagged"].sum())
    actual_event_days = int(df["is_match_day"].sum())

    recall = true_positives / actual_event_days if actual_event_days > 0 else float("nan")
    precision = true_positives / flagged if flagged > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )

    return {
        "modified_z_threshold": MODIFIED_Z_THRESHOLD,
        "n_days_total": int(len(df)),
        "n_actual_event_days": actual_event_days,
        "n_flagged_days": flagged,
        "n_true_positives": true_positives,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def main() -> dict:
    wiki = pd.read_csv(PROCESSED_DIR / "clean_wikipedia_pageviews.csv", parse_dates=["date"])
    matches = pd.read_csv(PROCESSED_DIR / "clean_cricsheet_matches.csv", parse_dates=["date"])
    match_days = set(matches["date"].dt.normalize())

    scored = detect_spikes(wiki)
    results = evaluate(scored, match_days)

    print("=== Spike detection: day-of-week seasonal baseline (median/MAD) ===")
    for key, value in results.items():
        print(f"  {key}: {value}")

    scored_out = PROCESSED_DIR / "spike_detection_scored_days.csv"
    scored.to_csv(scored_out, index=False)
    print(f"\n[spike_detection] wrote {scored_out}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[spike_detection] wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
