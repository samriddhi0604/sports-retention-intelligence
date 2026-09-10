"""
Step 5c -- Hybrid recommender with cold-start blending.

Pure collaborative filtering (Step 5b) fails exactly for the population
this project targets: a low-activity user has too little interaction
history for their CF profile vector (the mean of their watched items'
embeddings) to be reliable -- 1-2 points barely constrain a `k`-dimensional
average. This blends the CF similarity score with a content-based score
built from real IMDb genre/rating metadata, shifting the blend toward
content-based as a user's distinct-item count drops.

Weighting function (as specified in CLAUDE.md): for a user with
`interaction_count` distinct watched items,
    weight_cf = min(1, interaction_count / 10)
    weight_content = 1 - weight_cf
So a user with 10+ distinct watched items gets a pure CF-based score (their
embedding profile is reasonably stable by then); a user with 2 gets 80% of
the score from content similarity and only 20% from their thin CF profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

COLD_START_FULL_CF_THRESHOLD = 10  # distinct items at which weight_cf reaches 1.0


def build_genre_index(catalog: pd.DataFrame) -> dict:
    all_genres = sorted({g for genres in catalog["genres"] for g in genres})
    return {g: i for i, g in enumerate(all_genres)}


def content_vector_for_row(row: pd.Series, genre_idx: dict) -> np.ndarray:
    """Multi-hot genre vector + normalized rating for one catalog row --
    built entirely from real IMDb metadata, computable for ANY title
    (including ones too rarely watched to have a CF embedding)."""
    vec = np.zeros(len(genre_idx) + 1)
    for g in row["genres"]:
        if g in genre_idx:
            vec[genre_idx[g]] = 1.0
    vec[-1] = row["averageRating"] / 10.0
    return vec


def build_content_vectors(catalog: pd.DataFrame, item_ids: list, genre_idx: dict) -> np.ndarray:
    """One row per item_id, for the fixed recommendable-candidate universe."""
    catalog_lookup = catalog.set_index("tconst")
    vectors = np.zeros((len(item_ids), len(genre_idx) + 1))
    for row_i, tconst in enumerate(item_ids):
        if tconst not in catalog_lookup.index:
            continue
        vectors[row_i] = content_vector_for_row(catalog_lookup.loc[tconst], genre_idx)
    return vectors


def compute_weights(interaction_count: int, threshold: int = COLD_START_FULL_CF_THRESHOLD) -> tuple[float, float]:
    weight_cf = min(1.0, interaction_count / threshold)
    return weight_cf, 1.0 - weight_cf


def hybrid_recommend(
    watched_item_ids: list,
    item_embeddings: np.ndarray,
    content_vectors: np.ndarray,
    item_ids: list,
    catalog: pd.DataFrame,
    genre_idx: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """Candidates are drawn from `item_ids` (the CF-eligible, >=2-session
    universe -- rare items aren't good recommendation candidates anyway).
    But the user's PROFILE is built from all of their distinct watched
    items, including ones too rare to have a CF embedding -- a singleton
    watched title still has real genre/rating metadata and should still
    inform the content-based half of the score, not be silently dropped."""
    item_idx = {t: i for i, t in enumerate(item_ids)}
    watched_item_ids = list(dict.fromkeys(watched_item_ids))  # distinct, order-preserving
    interaction_count = len(watched_item_ids)
    weight_cf, weight_content = compute_weights(interaction_count)

    cf_watched_idx = [item_idx[t] for t in watched_item_ids if t in item_idx]
    if cf_watched_idx:
        cf_profile = item_embeddings[cf_watched_idx].mean(axis=0, keepdims=True)
        cf_sims = cosine_similarity(cf_profile, item_embeddings)[0]
    else:
        # none of this user's watched items are CF-eligible -- there's no
        # embedding signal to lean on regardless of what the formula's
        # weight_cf says, so the CF term contributes nothing here.
        cf_sims = np.zeros(len(item_ids))
        weight_cf, weight_content = 0.0, 1.0

    catalog_lookup = catalog.set_index("tconst")
    watched_content_vecs = [
        content_vector_for_row(catalog_lookup.loc[t], genre_idx)
        for t in watched_item_ids
        if t in catalog_lookup.index
    ]
    if watched_content_vecs:
        content_profile = np.mean(watched_content_vecs, axis=0, keepdims=True)
        content_sims = cosine_similarity(content_profile, content_vectors)[0]
    else:
        content_sims = np.zeros(len(item_ids))

    hybrid_scores = weight_cf * cf_sims + weight_content * content_sims
    for idx in cf_watched_idx:
        hybrid_scores[idx] = -np.inf

    top_idx = np.argsort(-hybrid_scores)[:top_n]
    catalog_lookup = catalog.set_index("tconst")
    return pd.DataFrame(
        {
            "tconst": [item_ids[i] for i in top_idx],
            "title": catalog_lookup.reindex([item_ids[i] for i in top_idx])["primaryTitle"].to_numpy(),
            "hybrid_score": hybrid_scores[top_idx],
            "weight_cf": weight_cf,
            "weight_content": weight_content,
            "interaction_count": interaction_count,
        }
    )


def main() -> None:
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    catalog["genres"] = catalog["genres"].str.split(",")
    sessions = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["timestamp"])

    item_ids = json.loads((PROCESSED_DIR / "item_ids.json").read_text())
    item_embeddings = np.load(PROCESSED_DIR / "item_embeddings.npy")
    genre_idx = build_genre_index(catalog)
    content_vectors = build_content_vectors(catalog, item_ids, genre_idx)
    print(f"[hybrid] loaded {len(item_ids)} items, content vectors shape {content_vectors.shape}")

    watched_by_user = sessions.groupby("user_id")["title_id"].apply(list)
    session_counts = sessions.groupby("user_id").size().sort_values()

    # demonstrate the weighting shift concretely: one very low-activity user
    # (near cold-start) and one high-activity user
    example_users = [
        session_counts.index[0],  # lowest activity
        session_counts.index[len(session_counts) // 2],  # median activity
        session_counts.index[-1],  # highest activity
    ]

    print("\n=== Hybrid recommendations for example users at different activity levels ===")
    for user_id in example_users:
        watched = list(dict.fromkeys(watched_by_user.get(user_id, [])))  # distinct, order-preserving
        recs = hybrid_recommend(watched, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=5)
        print(f"\n  user_id={user_id}, distinct items watched={len(watched)}, "
              f"weight_cf={recs['weight_cf'].iloc[0]:.2f}, weight_content={recs['weight_content'].iloc[0]:.2f}")
        for _, r in recs.iterrows():
            print(f"    {r['title']} (score={r['hybrid_score']:.3f})")


if __name__ == "__main__":
    main()
