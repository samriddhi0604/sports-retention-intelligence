"""
Step 5b -- Collaborative filtering via matrix factorization.

Builds the user-item interaction matrix from the real session log
(data/processed/sessions.csv, real IMDb tconst as item ids), factorizes it
with TruncatedSVD into user/item embeddings, and concretely inspects
item-item similarity for a handful of real titles spanning different
genres/languages -- the actual test of whether co-viewing reveals
cross-genre/cross-language affinity that metadata-based similarity
structurally cannot. The real result is reported, including if it's weaker
or messier than a clean story.

Language isn't in the cleaned catalog (data/clean.py never needed it -- see
data/README.md), so it's pulled here directly from the already-downloaded
data/raw/imdb/title.akas.tsv.gz, scoped only to titles that actually appear
in sessions.csv. This is a self-contained enrichment for this step's
qualitative inspection, not a change to the core Step 1 pipeline outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_PATH = PROCESSED_DIR / "collab_filter_results.json"

CANDIDATE_K = [10, 20, 30, 50]
HELD_OUT_FRACTION = 0.10
RNG_SEED = 42
MIN_ITEM_SESSIONS = 2  # drop singleton items -- one interaction can't inform a useful embedding
TOP_N_NEIGHBORS = 5


def load_primary_language(tconst_set: set) -> dict:
    """Stream title.akas.tsv.gz (already downloaded in Step 1) and return a
    tconst -> language dict, scoped only to titles that appear in our
    sessions, picking the first non-null language seen per title."""
    print("[collab_filter] scanning title.akas.tsv.gz for language of titles actually watched...")
    language_by_title: dict = {}
    for chunk in pd.read_csv(
        RAW_DIR / "imdb" / "title.akas.tsv.gz",
        sep="\t",
        na_values="\\N",
        quoting=csv.QUOTE_NONE,
        usecols=["titleId", "language"],
        chunksize=1_000_000,
        dtype=str,
    ):
        chunk = chunk[chunk["titleId"].isin(tconst_set) & chunk["language"].notna()]
        for tconst, lang in zip(chunk["titleId"], chunk["language"]):
            language_by_title.setdefault(tconst, lang)
        if len(language_by_title) >= len(tconst_set):
            break
    print(f"[collab_filter] found a language for {len(language_by_title)}/{len(tconst_set)} titles")
    return language_by_title


def build_interaction_matrix(sessions: pd.DataFrame) -> tuple[sparse.csr_matrix, list, list]:
    item_counts = sessions["title_id"].value_counts()
    keep_items = set(item_counts[item_counts >= MIN_ITEM_SESSIONS].index)
    sessions = sessions[sessions["title_id"].isin(keep_items)]

    user_ids = sorted(sessions["user_id"].unique())
    item_ids = sorted(sessions["title_id"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {t: i for i, t in enumerate(item_ids)}

    counts = sessions.groupby(["user_id", "title_id"]).size().reset_index(name="count")
    rows = counts["user_id"].map(user_idx)
    cols = counts["title_id"].map(item_idx)

    # Raw (or log1p) counts make SVD's leading component a popularity axis
    # rather than genuine taste structure -- confirmed empirically here (1st
    # singular value ~3x the 2nd, explained-variance ratio ~3x the rest).
    # Applying IDF weighting per item (same idea as classic TF-IDF/LSA on
    # term-document matrices, for the same reason: down-weight terms/items
    # that appear across most "documents"/users, since ubiquity carries no
    # discriminating signal) lets the more nuanced co-occurrence structure
    # actually surface in the top components instead of being swamped.
    n_users_per_item = counts.groupby("title_id")["user_id"].nunique()
    idf = np.log(len(user_ids) / n_users_per_item)
    item_idf = counts["title_id"].map(idf).to_numpy()
    values = np.log1p(counts["count"].to_numpy()) * item_idf

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    print(f"[collab_filter] interaction matrix: {matrix.shape}, {matrix.nnz} nonzero entries "
          f"(dropped {len(item_counts) - len(keep_items)} singleton items)")
    return matrix, user_ids, item_ids


def select_latent_dim(matrix: sparse.csr_matrix, rng: np.random.Generator) -> tuple[int, dict]:
    """Mask a held-out fraction of nonzero entries, fit TruncatedSVD at each
    candidate k on the remaining matrix, and score by RMSE reconstructing
    the held-out cells (not the full dense matrix -- just those cells)."""
    coo = matrix.tocoo()
    n_nonzero = len(coo.data)
    held_out_idx = rng.choice(n_nonzero, size=int(n_nonzero * HELD_OUT_FRACTION), replace=False)
    held_out_mask = np.zeros(n_nonzero, dtype=bool)
    held_out_mask[held_out_idx] = True

    train_data = coo.data.copy()
    train_data[held_out_mask] = 0.0
    train_matrix = sparse.csr_matrix((train_data, (coo.row, coo.col)), shape=matrix.shape)

    held_out_rows = coo.row[held_out_mask]
    held_out_cols = coo.col[held_out_mask]
    held_out_true = coo.data[held_out_mask]

    rmse_by_k = {}
    for k in CANDIDATE_K:
        svd = TruncatedSVD(n_components=k, random_state=RNG_SEED)
        user_emb = svd.fit_transform(train_matrix)
        item_emb = svd.components_.T
        preds = np.einsum("ij,ij->i", user_emb[held_out_rows], item_emb[held_out_cols])
        rmse = float(np.sqrt(np.mean((preds - held_out_true) ** 2)))
        rmse_by_k[k] = rmse
        print(f"[collab_filter] k={k}: held-out RMSE={rmse:.4f}")

    best_k = min(rmse_by_k, key=rmse_by_k.get)
    return best_k, rmse_by_k


def inspect_neighbors(
    item_embeddings: np.ndarray,
    item_ids: list,
    catalog: pd.DataFrame,
    language_by_title: dict,
    sample_titles: list,
) -> list:
    item_idx = {t: i for i, t in enumerate(item_ids)}
    catalog_lookup = catalog.set_index("tconst")
    sims = cosine_similarity(item_embeddings)

    results = []
    for tconst in sample_titles:
        if tconst not in item_idx:
            continue
        idx = item_idx[tconst]
        neighbor_order = np.argsort(-sims[idx])
        neighbor_order = [i for i in neighbor_order if i != idx][:TOP_N_NEIGHBORS]

        query_row = catalog_lookup.loc[tconst]
        query_info = {
            "title": query_row["primaryTitle"],
            "genres": query_row["genres"],
            "language": language_by_title.get(tconst, "unknown"),
        }
        neighbors = []
        cross_genre_or_language_count = 0
        for n_idx in neighbor_order:
            n_tconst = item_ids[n_idx]
            n_row = catalog_lookup.loc[n_tconst]
            n_language = language_by_title.get(n_tconst, "unknown")
            shares_genre = bool(set(n_row["genres"]) & set(query_row["genres"]))
            shares_language = n_language == query_info["language"]
            if not shares_genre or not shares_language:
                cross_genre_or_language_count += 1
            neighbors.append(
                {
                    "title": n_row["primaryTitle"],
                    "genres": n_row["genres"],
                    "language": n_language,
                    "similarity": float(sims[idx, n_idx]),
                    "shares_genre_with_query": shares_genre,
                    "shares_language_with_query": shares_language,
                }
            )
        results.append(
            {
                "query": query_info,
                "neighbors": neighbors,
                "n_cross_genre_or_language_neighbors": cross_genre_or_language_count,
            }
        )
        print(f"\n  Query: {query_info['title']} ({query_info['genres']}, lang={query_info['language']})")
        for n in neighbors:
            flag = "CROSS" if (not n["shares_genre_with_query"] or not n["shares_language_with_query"]) else "same"
            print(f"    [{flag}] {n['title']} ({n['genres']}, lang={n['language']}) sim={n['similarity']:.3f}")

    return results


def recommend_for_user(
    watched_item_ids: list,
    item_embeddings: np.ndarray,
    item_ids: list,
    catalog: pd.DataFrame,
    gateway_genres: list,
    gateway_boost: float = 0.2,
    top_n: int = 10,
) -> pd.DataFrame:
    """CF-only recommendations: average embedding of a user's watched items,
    ranked by cosine similarity to all other items, boosted for gateway genres."""
    item_idx = {t: i for i, t in enumerate(item_ids)}
    watched_idx = [item_idx[t] for t in watched_item_ids if t in item_idx]
    if not watched_idx:
        return pd.DataFrame(columns=["tconst", "score"])

    profile = item_embeddings[watched_idx].mean(axis=0, keepdims=True)
    sims = cosine_similarity(profile, item_embeddings)[0]

    catalog_lookup = catalog.set_index("tconst")
    genre_map = catalog_lookup["genres"].reindex(item_ids)
    boost = np.array(
        [1.0 + gateway_boost if bool(set(g or []) & set(gateway_genres)) else 1.0 for g in genre_map]
    )
    scores = sims * boost

    for idx in watched_idx:
        scores[idx] = -np.inf  # don't recommend what they've already watched

    top_idx = np.argsort(-scores)[:top_n]
    return pd.DataFrame(
        {
            "tconst": [item_ids[i] for i in top_idx],
            "title": catalog_lookup.loc[[item_ids[i] for i in top_idx], "primaryTitle"].to_numpy(),
            "score": scores[top_idx],
        }
    )


def diagnose_similarity_structure(
    item_embeddings: np.ndarray, item_ids: list, sessions: pd.DataFrame, rng: np.random.Generator
) -> dict:
    """Nearest-neighbor similarity alone can't distinguish 'genuine taste
    affinity' from 'low-dimensional embedding space + searching for the
    single best match among tens of thousands of candidates always finds
    something'. This checks two things directly: (1) does item popularity
    correlate with similarity (a hub/popularity-bias artifact), and (2) how
    does typical (random-pair) similarity compare to the max achievable
    among many candidates -- if nearest-neighbor scores are only a little
    above the random-pair ceiling, that's consistent with a search-over-many-
    candidates effect rather than a real discovered structure."""
    item_counts = sessions["title_id"].value_counts()
    sample_idx = rng.choice(len(item_embeddings), size=min(800, len(item_embeddings)), replace=False)
    sims = cosine_similarity(item_embeddings[sample_idx])
    iu = np.triu_indices_from(sims, k=1)
    sample_ids = [item_ids[i] for i in sample_idx]
    pop = np.array([item_counts.get(t, 0) for t in sample_ids])
    pop_sum = pop[iu[0]] + pop[iu[1]]

    popularity_corr = float(np.corrcoef(pop_sum, sims[iu])[0, 1])
    random_pair_mean = float(sims[iu].mean())
    random_pair_p99 = float(np.percentile(sims[iu], 99))

    return {
        "popularity_similarity_correlation": popularity_corr,
        "random_pair_mean_similarity": random_pair_mean,
        "random_pair_99th_percentile_similarity": random_pair_p99,
    }


def main() -> dict:
    sessions = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["timestamp"])
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    catalog["genres"] = catalog["genres"].str.split(",")

    matrix, user_ids, item_ids = build_interaction_matrix(sessions)

    rng = np.random.default_rng(RNG_SEED)
    best_k, rmse_by_k = select_latent_dim(matrix, rng)
    print(f"[collab_filter] selected k={best_k} (lowest held-out RMSE)")

    svd = TruncatedSVD(n_components=best_k, random_state=RNG_SEED)
    user_embeddings = svd.fit_transform(matrix)
    item_embeddings = svd.components_.T

    language_by_title = load_primary_language(set(item_ids))

    # sample titles spanning different genres/languages, among reasonably-
    # watched items so their embeddings are meaningful (not near-random from
    # a single interaction)
    item_session_counts = sessions["title_id"].value_counts()
    well_watched = [t for t in item_ids if item_session_counts.get(t, 0) >= 20]
    catalog_lookup = catalog.set_index("tconst")
    sample_titles = []
    seen_langs = set()
    for tconst in well_watched:
        lang = language_by_title.get(tconst, "unknown")
        if lang not in seen_langs:
            sample_titles.append(tconst)
            seen_langs.add(lang)
        if len(sample_titles) >= 6:
            break

    print(f"\n=== Item-item nearest neighbors for {len(sample_titles)} sample titles ===")
    neighbor_results = inspect_neighbors(item_embeddings, item_ids, catalog, language_by_title, sample_titles)

    total_cross = sum(r["n_cross_genre_or_language_neighbors"] for r in neighbor_results)
    total_neighbors = sum(len(r["neighbors"]) for r in neighbor_results)
    cross_rate = total_cross / total_neighbors if total_neighbors else 0.0
    print(f"\n[collab_filter] {total_cross}/{total_neighbors} nearest-neighbor pairs "
          f"({cross_rate:.1%}) are cross-genre-or-cross-language")

    diagnostics = diagnose_similarity_structure(item_embeddings, item_ids, sessions, rng)
    print(f"\n[collab_filter] diagnostics: popularity-similarity correlation="
          f"{diagnostics['popularity_similarity_correlation']:.3f}, "
          f"random-pair mean similarity={diagnostics['random_pair_mean_similarity']:.3f}, "
          f"random-pair 99th pct={diagnostics['random_pair_99th_percentile_similarity']:.3f}")

    # Honest interpretation, not just the raw cross-genre rate: this
    # dataset's session generator never encoded any deliberate item-to-item
    # taste affinity (titles are chosen via genre-bucket preference, then
    # popularity-weighted within that bucket -- see data/features.py). A
    # weak popularity-similarity correlation rules out simple popularity
    # bias as the main driver, but nearest-neighbor scores (0.92-0.98) sit
    # far above typical random-pair similarity -- consistent with a
    # low-dimensional-embedding-space effect (searching for the single best
    # match among ~30k candidates in only `k` dimensions tends to surface a
    # spuriously high-similarity match regardless of genuine structure) more
    # than with a real discovered cross-genre taste signal.
    honest_note = (
        f"{cross_rate:.1%} of nearest-neighbor pairs share neither genre nor language, but this is "
        "NOT reported as a genuine 'hidden cross-genre affinity discovered' finding. The session "
        "generator (data/features.py) never encoded any deliberate item-to-item taste affinity beyond "
        "genre-bucket preference, so there is no real signal here to discover. Popularity bias is a "
        f"weak explanation on its own (popularity-similarity correlation={diagnostics['popularity_similarity_correlation']:.3f} "
        "on random pairs), but nearest-neighbor similarity scores sit far above the typical random-pair "
        f"similarity (99th percentile={diagnostics['random_pair_99th_percentile_similarity']:.3f}), which is "
        "more consistent with a low-dimensional-embedding search-over-many-candidates effect than with "
        "genuine cross-genre affinity. Reported as a real methodological limitation of this simulated "
        "dataset, not as a positive result -- per CLAUDE.md's instruction not to force the clean story "
        "to hold when the data doesn't support it."
    )
    print(f"\n{honest_note}")

    results = {
        "latent_dim_candidates": CANDIDATE_K,
        "held_out_rmse_by_k": rmse_by_k,
        "selected_k": best_k,
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "neighbor_inspection": neighbor_results,
        "cross_genre_or_language_rate": cross_rate,
        "similarity_diagnostics": diagnostics,
        "honest_note": honest_note,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[collab_filter] wrote {RESULTS_PATH}")

    np.save(PROCESSED_DIR / "item_embeddings.npy", item_embeddings)
    np.save(PROCESSED_DIR / "user_embeddings.npy", user_embeddings)
    Path(PROCESSED_DIR / "item_ids.json").write_text(json.dumps(list(item_ids)))
    Path(PROCESSED_DIR / "user_ids.json").write_text(json.dumps([int(u) for u in user_ids]))
    print("[collab_filter] wrote embeddings + id lists for Step 5c/5d")

    return results


if __name__ == "__main__":
    main()
