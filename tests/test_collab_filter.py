"""
Step 7 -- test that the matrix factorization approach (TruncatedSVD, as
used in recommendations/collab_filter.py) reconstructs a small known
interaction matrix reasonably well, before trusting it on the full
280k-session real interaction matrix.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD


def test_svd_reconstructs_exact_low_rank_matrix():
    # construct a matrix that is EXACTLY rank 2 by design: outer products of
    # two known factor pairs -- a rank-2 SVD should reconstruct it almost
    # perfectly (this is the known-answer case: rank matches components)
    rng = np.random.default_rng(0)
    true_users = rng.normal(size=(20, 2))
    true_items = rng.normal(size=(15, 2))
    matrix = true_users @ true_items.T

    svd = TruncatedSVD(n_components=2, random_state=0)
    user_emb = svd.fit_transform(matrix)
    item_emb = svd.components_.T
    reconstruction = user_emb @ item_emb.T

    rmse = np.sqrt(np.mean((matrix - reconstruction) ** 2))
    assert rmse < 1e-8  # exact rank match -- should be near machine precision


def test_svd_reconstruction_improves_with_more_components():
    # a rank-5 matrix: reconstruction error should be strictly lower with
    # k=5 components than with k=2 (a sane monotonicity check on the
    # reconstruction-quality logic collab_filter.py relies on for choosing k)
    rng = np.random.default_rng(1)
    true_users = rng.normal(size=(30, 5))
    true_items = rng.normal(size=(25, 5))
    matrix = true_users @ true_items.T

    errors = {}
    for k in [2, 5]:
        svd = TruncatedSVD(n_components=k, random_state=0)
        user_emb = svd.fit_transform(matrix)
        item_emb = svd.components_.T
        reconstruction = user_emb @ item_emb.T
        errors[k] = np.sqrt(np.mean((matrix - reconstruction) ** 2))

    assert errors[5] < errors[2]
    assert errors[5] < 1e-8  # k matches true rank -- should be near-exact
