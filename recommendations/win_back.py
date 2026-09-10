"""
Win-back recommendation logic -- simple rule-based baseline.

Kept alongside the ML-based hybrid recommender (gateway_analysis.py,
collab_filter.py, hybrid_recommender.py, evaluate_recommender.py) as a
deliberately simple comparison point, per user request. This is the
original Step 5 design: for users flagged at-risk by the Step 4 churn
model, recommend sports content specifically to the segment whose
sports-spike engagement is below the retained-user average -- i.e. act
directly on the Step 2 hypothesis-test finding, not a generic "watch more"
nudge. Intentionally a rule, not a model -- see the root README for how
this compares to the ML recommender.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"
RESULTS_PATH = PROCESSED_DIR / "win_back_results.json"


def flag_at_risk(user_features: pd.DataFrame) -> pd.Series:
    model = joblib.load(ARTIFACTS_DIR / "churn_model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "churn_scaler.joblib")
    threshold = joblib.load(ARTIFACTS_DIR / "churn_threshold.joblib")

    feature_cols = ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]
    X = scaler.transform(user_features[feature_cols].to_numpy())
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba >= threshold, index=user_features.index)


def segment_at_risk_users(user_features: pd.DataFrame) -> pd.DataFrame:
    at_risk_mask = flag_at_risk(user_features)
    at_risk = user_features.loc[at_risk_mask].copy()

    retained_avg_sports_engagement = user_features.loc[
        user_features["churned"] == 0, "sports_spike_engagement"
    ].mean()

    at_risk["recommendation"] = at_risk["sports_spike_engagement"].apply(
        lambda x: "sports_content_winback" if x < retained_avg_sports_engagement else "general_winback"
    )
    at_risk.attrs["retained_avg_sports_engagement"] = retained_avg_sports_engagement
    return at_risk


def main() -> dict:
    user_features = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    segmented = segment_at_risk_users(user_features)

    n_at_risk = len(segmented)
    n_sports_segment = int((segmented["recommendation"] == "sports_content_winback").sum())
    n_general_segment = n_at_risk - n_sports_segment

    results = {
        "n_at_risk_users": n_at_risk,
        "retained_avg_sports_engagement": float(segmented.attrs["retained_avg_sports_engagement"]),
        "n_users_recommended_sports_content": n_sports_segment,
        "n_users_recommended_general_winback": n_general_segment,
        "pct_at_risk_below_retained_sports_avg": n_sports_segment / n_at_risk if n_at_risk else float("nan"),
    }

    print("=== Rule-based win-back targeting (baseline) ===")
    for key, value in results.items():
        print(f"  {key}: {value}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[win_back] wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
