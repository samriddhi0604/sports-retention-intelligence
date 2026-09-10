"""
Step 7 -- tests that the hybrid recommender's cold-start weighting shifts
toward content-based scoring as interaction count drops, on a small
hand-constructed example, before trusting it on the full simulated catalog
(already exercised for real in recommendations/hybrid_recommender.py).
"""

import numpy as np
import pandas as pd

from recommendations.hybrid_recommender import build_content_vectors, build_genre_index, compute_weights, hybrid_recommend


def test_compute_weights_formula():
    assert compute_weights(0) == (0.0, 1.0)
    assert compute_weights(5) == (0.5, 0.5)
    assert compute_weights(10) == (1.0, 0.0)
    assert compute_weights(20) == (1.0, 0.0)  # capped at 1.0, not 2.0


def _make_small_catalog():
    # t1/t3 share a genre (Drama), t2/t4 share a different genre (Comedy) --
    # so content similarity pairs t1<->t3 and t2<->t4
    return pd.DataFrame(
        {
            "tconst": ["t1", "t2", "t3", "t4"],
            "primaryTitle": ["T1", "T2", "T3", "T4"],
            "genres": [["Drama"], ["Comedy"], ["Drama"], ["Comedy"]],
            "averageRating": [8.0, 8.0, 8.0, 8.0],
        }
    )


def test_cold_start_user_gets_content_based_recommendation():
    catalog = _make_small_catalog()
    item_ids = ["t1", "t2", "t3", "t4"]
    genre_idx = build_genre_index(catalog)
    content_vectors = build_content_vectors(catalog, item_ids, genre_idx)

    # CF embeddings pair t1<->t2 and t3<->t4 -- the OPPOSITE of the content
    # pairing, so the test can distinguish which signal actually won
    item_embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # a user who has watched exactly 1 item -> weight_content should
    # dominate (0.9), so the top recommendation should be t3 (content
    # match), not t2 (CF match)
    recs = hybrid_recommend(["t1"], item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=3)

    assert recs["weight_content"].iloc[0] == 0.9
    assert recs.iloc[0]["tconst"] == "t3"


def test_warm_user_gets_cf_based_recommendation():
    catalog = _make_small_catalog()
    item_ids = ["t1", "t2", "t3", "t4"]
    genre_idx = build_genre_index(catalog)
    content_vectors = build_content_vectors(catalog, item_ids, genre_idx)
    item_embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # pad with 9 filler ids (not real items) to reach interaction_count=10,
    # so weight_cf hits its 1.0 ceiling -- only t1 actually contributes to
    # either profile since the fillers aren't in the catalog/item universe
    watched = ["t1"] + [f"filler{i}" for i in range(9)]
    recs = hybrid_recommend(watched, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=3)

    assert recs["weight_cf"].iloc[0] == 1.0
    assert recs.iloc[0]["tconst"] == "t2"  # CF match, not t3 (the content match)


def test_force_content_only_ignores_cf_regardless_of_interaction_count():
    catalog = _make_small_catalog()
    item_ids = ["t1", "t2", "t3", "t4"]
    genre_idx = build_genre_index(catalog)
    content_vectors = build_content_vectors(catalog, item_ids, genre_idx)
    item_embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    watched = ["t1"] + [f"filler{i}" for i in range(9)]  # would normally force weight_cf=1.0
    recs = hybrid_recommend(
        watched, item_embeddings, content_vectors, item_ids, catalog, genre_idx, top_n=3,
        force_content_only=True,
    )

    assert recs["weight_cf"].iloc[0] == 0.0
    assert recs.iloc[0]["tconst"] == "t3"  # content match, even though interaction_count=10
