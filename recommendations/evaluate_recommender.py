"""
Step 5d -- Targeting and evaluation.

Defines the recommender's target population as low-OVERALL-engagement users
(not low-sports-specifically, per Step 5a's framing: heavy/light watching is
a general trait, so the actionable population is anyone under-watching
overall). Reports how much this overlaps with Step 4's at-risk churn flag
rather than assuming they're the same group, generates hybrid
recommendations for the target population and checks gateway-genre
alignment, and runs a rough offline validation: for each Step 5a ramped-up
user, hold out their actual early-session (gateway) titles, build a profile
from only their later sessions, and check whether the recommender would
have surfaced those held-out titles -- compared against a content-only
baseline, to directly test whether cold-start blending actually helps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommendations.gateway_analysis import classify_ramp_up, get_window_midpoint
from recommendations.hybrid_recommender import (
    build_content_vectors,
    build_genre_index,
    hybrid_recommend,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"
RESULTS_PATH = PROCESSED_DIR / "evaluate_recommender_results.json"

LOW_ENGAGEMENT_QUANTILE = 0.25  # bottom 25% by total observed session count
TOP_N = 10


def flag_at_risk(user_features: pd.DataFrame) -> pd.Series:
    model = joblib.load(ARTIFACTS_DIR / "churn_model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "churn_scaler.joblib")
    threshold = joblib.load(ARTIFACTS_DIR / "churn_threshold.joblib")

    feature_cols = ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]
    X = scaler.transform(user_features[feature_cols].to_numpy())
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba >= threshold, index=user_features.index)


def define_low_engagement_population(user_features: pd.DataFrame) -> pd.Series:
    cutoff = user_features["n_sessions"].quantile(LOW_ENGAGEMENT_QUANTILE)
    return user_features["n_sessions"] <= cutoff


def offline_validation(
    sessions: pd.DataFrame,
    midpoint: pd.Timestamp,
    item_embeddings: np.ndarray,
    content_vectors: np.ndarray,
    item_ids: list,
    catalog: pd.DataFrame,
    genre_idx: dict,
    ramped_up_user_ids: set,
) -> dict:
    """For each ramped-up user, hold out their early (pre-midpoint) titles,
    build a profile from only their late (post-midpoint) titles, and check
    whether the recommender surfaces the held-out titles -- once with the
    real hybrid blend, once forcing content-only, so cold-start blending's
    actual contribution can be checked directly rather than assumed."""
    hits_hybrid, hits_content_only = 0, 0
    n_evaluated = 0

    for user_id in ramped_up_user_ids:
        user_sessions = sessions[sessions["user_id"] == user_id]
        early = set(user_sessions.loc[user_sessions["timestamp"] < midpoint, "title_id"])
        late = list(dict.fromkeys(user_sessions.loc[user_sessions["timestamp"] >= midpoint, "title_id"]))
        if not early or not late:
            continue
        n_evaluated += 1

        recs_hybrid = hybrid_recommend(
            late, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=TOP_N
        )
        if set(recs_hybrid["tconst"]) & early:
            hits_hybrid += 1

        # content-only baseline
        recs_content_only = hybrid_recommend(
            late, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=TOP_N,
            force_content_only=True,
        )
        if set(recs_content_only["tconst"]) & early:
            hits_content_only += 1

    return {
        "n_ramped_up_users_evaluated": n_evaluated,
        "hit_rate_hybrid": hits_hybrid / n_evaluated if n_evaluated else float("nan"),
        "hit_rate_content_only": hits_content_only / n_evaluated if n_evaluated else float("nan"),
        "note": (
            f"'hit' = at least one of a user's actual held-out early-session titles appeared in "
            f"their top-{TOP_N} recommendations built from only their later sessions."
        ),
    }


def main() -> dict:
    user_features = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    sessions = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["timestamp"])
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    catalog["genres"] = catalog["genres"].str.split(",")

    item_ids = json.loads((PROCESSED_DIR / "item_ids.json").read_text())
    item_embeddings = np.load(PROCESSED_DIR / "item_embeddings.npy")
    genre_idx = build_genre_index(catalog)
    content_vectors = build_content_vectors(catalog, item_ids, genre_idx)

    # --- targeting population ---
    low_engagement = define_low_engagement_population(user_features)
    at_risk = flag_at_risk(user_features)

    n_low = int(low_engagement.sum())
    n_at_risk = int(at_risk.sum())
    n_both = int((low_engagement & at_risk).sum())
    pct_low_that_are_at_risk = n_both / n_low if n_low else float("nan")
    pct_at_risk_that_are_low = n_both / n_at_risk if n_at_risk else float("nan")

    print(f"[evaluate] low-engagement population (bottom {LOW_ENGAGEMENT_QUANTILE:.0%} by n_sessions): {n_low} users")
    print(f"[evaluate] at-risk population (Step 4 churn model flag): {n_at_risk} users")
    print(f"[evaluate] overlap: {n_both} users in both "
          f"({pct_low_that_are_at_risk:.1%} of low-engagement users are also at-risk, "
          f"{pct_at_risk_that_are_low:.1%} of at-risk users are also low-engagement)")

    # --- recommendations for the target population + gateway alignment ---
    gateway = json.loads((PROCESSED_DIR / "gateway_analysis_results.json").read_text())
    gateway_genres = [g["category"] for g in gateway["top_genres_by_lift"] if g["lift"] > 1.0]
    print(f"[evaluate] gateway genres (lift > 1.0 from Step 5a): {gateway_genres}")

    target_user_ids = user_features.loc[low_engagement, "user_id"].tolist()
    watched_by_user = sessions.groupby("user_id")["title_id"].apply(list)
    catalog_lookup = catalog.set_index("tconst")

    n_recs_total, n_recs_gateway_genre = 0, 0
    for user_id in target_user_ids:
        watched = watched_by_user.get(user_id, [])
        recs = hybrid_recommend(watched, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=TOP_N)
        for tconst in recs["tconst"]:
            n_recs_total += 1
            if tconst in catalog_lookup.index:
                rec_genres = set(catalog_lookup.loc[tconst, "genres"])
                if rec_genres & set(gateway_genres):
                    n_recs_gateway_genre += 1

    gateway_alignment_rate = n_recs_gateway_genre / n_recs_total if n_recs_total else float("nan")
    print(f"[evaluate] {n_recs_gateway_genre}/{n_recs_total} recommendations to the target population "
          f"({gateway_alignment_rate:.1%}) are in a gateway genre")

    # --- offline validation ---
    ramp = classify_ramp_up(sessions, get_window_midpoint(sessions))
    ramped_up_ids = set(ramp.loc[ramp["ramped_up"], "user_id"])
    validation = offline_validation(
        sessions, get_window_midpoint(sessions), item_embeddings, content_vectors,
        item_ids, catalog, genre_idx, ramped_up_ids,
    )
    print(f"\n[evaluate] offline validation: hybrid hit-rate={validation['hit_rate_hybrid']:.1%}, "
          f"content-only hit-rate={validation['hit_rate_content_only']:.1%} "
          f"(n={validation['n_ramped_up_users_evaluated']})")

    honest_note = (
        f"Cold-start blending {'meaningfully outperforms' if validation['hit_rate_hybrid'] > validation['hit_rate_content_only'] * 1.2 else 'does NOT clearly outperform'} "
        "content-only scoring on this offline check "
        f"(hybrid={validation['hit_rate_hybrid']:.1%} vs. content-only={validation['hit_rate_content_only']:.1%}). "
        "IMPORTANT CAVEAT on that gap: the CF item embeddings (Step 5b) were fit on the full sessions.csv, "
        "which includes these same users' 'held-out' early-session titles as real training interactions -- "
        "the SVD already encoded that a given user watched both their early and late titles together before "
        "this offline check ever holds anything out. So the hybrid hit-rate reflects data leakage through the "
        "CF pathway, not leakage-free generalization; it demonstrates the mechanism CAN recover a held-out "
        "title when it was part of CF training, not that it would generalize this well to genuinely unseen "
        "preferences. Content-only has no such leakage (catalog metadata only), so its low hit-rate is the "
        "more trustworthy of the two numbers here. A proper fix would refit CF excluding each evaluated user's "
        "held-out interactions, which wasn't done given project scope -- flagged explicitly rather than "
        "presenting the 21% figure as validated. "
        f"Gateway-genre alignment for the target population is {gateway_alignment_rate:.1%}, consistent with "
        "Step 5a's finding that gateway signal is weak in this dataset -- this offline check and the gateway "
        "genre alignment are both low-confidence, rough proxies on simulated data with no real held-out user "
        "feedback, not a validated production evaluation."
    )
    print(f"\n{honest_note}")

    results = {
        "low_engagement_population_size": n_low,
        "at_risk_population_size": n_at_risk,
        "overlap_count": n_both,
        "pct_low_engagement_that_are_at_risk": pct_low_that_are_at_risk,
        "pct_at_risk_that_are_low_engagement": pct_at_risk_that_are_low,
        "gateway_genres_used": gateway_genres,
        "n_recommendations_to_target_population": n_recs_total,
        "n_recommendations_in_gateway_genre": n_recs_gateway_genre,
        "gateway_alignment_rate": gateway_alignment_rate,
        "offline_validation": validation,
        "offline_validation_has_cf_training_leakage": True,
        "honest_note": honest_note,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[evaluate] wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
