"""
Step 1d -- Feature engineering.

There is no public dataset of individual streaming-platform users' sessions
or churn outcomes (same reason data/README.md gives for viewership data, at
an even more sensitive per-user level). This module simulates users and
sessions, but grounds every feature in real data rather than arbitrary
synthetic parameters:

  - Match-day session timing is boosted on real Cricsheet match days
    (clean_cricsheet_matches.csv), scaled per-user by a synthetic
    sports_affinity trait. Note: the real Wikipedia pageview curve is
    deliberately NOT used here to modulate per-user session volume --
    IPL 2024 has matches on ~91% of days within the season itself (71
    matches across a 66-day window, doubleheaders included), so the wiki
    article's traffic stays elevated for nearly the whole season rather
    than spiking on individual match days. Using that curve (even
    smoothed) as a general activity multiplier would make almost every
    session simulated during the season land on a match day regardless of
    a user's actual sports interest -- a modeling mismatch, not a feature.
    Wikipedia's real-data role stays where it's actually valid: the
    aggregate-level viewership proxy for Step 3's spike detection.
  - Every session's title is a real title sampled from the real IMDb
    catalog (clean_imdb_catalog.csv) -- real genres, real rating.
  - sports_spike_engagement, genre_diversity, and avg_completion_rate are
    all computed from these real per-session records, not assigned directly.

The only genuinely synthetic pieces are (a) each user's latent traits
(sports affinity, genre-exploration concentration, base activity rate) and
(b) the logistic function used to sample a churn outcome from a user's
full-window trait realization. That churn relationship is an intentional,
documented, moderate-strength hypothesis being baked in for Step 2 to
detect and Step 4 to recover -- not a claim that real churn was observed.
See data/README.md and the root README's Honesty/Limitations section.

Two-phase simulation avoids circularity between "features" and "churn":
  1. Simulate each user's full analysis-window session history from their
     latent traits. Compute "true" full-window features from it.
  2. Sample churn from a logistic function of those true features + noise.
     Churned users get a churn day; their OBSERVED session history (and the
     features computed from it, which is what downstream steps actually
     use) is truncated to before that day -- exactly like a real churn
     dataset, where you only ever see behavior up to the point someone left.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

RNG_SEED = 42
N_USERS = 5000

# Churn-generating process (Step 2/4's headline relationship). Coefficients
# apply to standardized (z-scored) full-window features. Negative means
# "more of this reduces churn risk". NOISE_SIGMA is unexplained heterogeneity
# (price sensitivity, moving cities, competitor promos, etc. -- anything not
# captured by these 3 features) and is what keeps the effect real but not
# deterministic, per the "moderate effect strength" choice made with the user.
CHURN_INTERCEPT = -1.15  # tuned for an overall churn rate around 25-30%
BETA_SPORTS = -0.9
BETA_DIVERSITY = -0.35
BETA_COMPLETION = -0.5
NOISE_SIGMA = 2.4

MATCH_DAY_BOOST = 1.3  # extra session-rate multiplier on real match days, scaled by a user's sports_affinity
MIN_CHURN_DAY_OFFSET = 14  # a churned user is observed for at least 2 weeks before leaving


def load_real_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(PROCESSED_DIR / "clean_cricsheet_matches.csv", parse_dates=["date"])
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    catalog["genres"] = catalog["genres"].str.split(",")
    wiki = pd.read_csv(PROCESSED_DIR / "clean_wikipedia_pageviews.csv", parse_dates=["date"])
    return matches, catalog, wiki


# ---------------------------------------------------------------------------
# Pure, unit-testable feature functions (see tests/test_data_features.py)
# ---------------------------------------------------------------------------


def compute_sports_spike_engagement(session_dates: pd.Series, match_days: set) -> float:
    """Fraction of a user's sessions that fall on a real match day."""
    if len(session_dates) == 0:
        return np.nan
    on_match_day = session_dates.dt.normalize().isin(match_days)
    return float(on_match_day.mean())


def compute_genre_diversity(genre_lists: list[list[str]], all_genres: list[str]) -> float:
    """Shannon entropy of a user's genre-tag frequency, normalized to [0, 1]
    by the max possible entropy given the full catalog's genre vocabulary."""
    if len(genre_lists) == 0:
        return np.nan
    counts: dict[str, int] = {}
    for genres in genre_lists:
        for g in genres:
            counts[g] = counts.get(g, 0) + 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(all_genres))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_avg_completion_rate(ratings: pd.Series) -> float:
    """Proxy for completion rate: mean IMDb rating (0-10) of watched titles,
    normalized to [0, 1]. Not a true completion-rate field -- IMDb doesn't
    have one -- documented explicitly as a proxy (avg rating of what a user
    chose to watch is a reasonable stand-in for engagement/satisfaction)."""
    if len(ratings) == 0:
        return np.nan
    return float(ratings.mean() / 10.0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_user_traits(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "sports_affinity": rng.beta(2, 2, size=n_users),
            "genre_concentration": rng.uniform(0.5, 5.0, size=n_users),
            "base_activity_rate": rng.gamma(shape=2.0, scale=0.15, size=n_users),
        }
    )


def _build_genre_pools(catalog: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    catalog = catalog.copy()
    catalog["primary_genre"] = catalog["genres"].apply(lambda g: g[0])
    all_genres = sorted(catalog["primary_genre"].unique())
    pools = {}
    for genre in all_genres:
        sub = catalog[catalog["primary_genre"] == genre]
        weights = np.log1p(sub["numVotes"].to_numpy())
        weights = weights / weights.sum()
        pools[genre] = sub.assign(_weight=weights).reset_index(drop=True)
    return all_genres, pools


def simulate_sessions(
    users: pd.DataFrame,
    matches: pd.DataFrame,
    catalog: pd.DataFrame,
    wiki: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate each user's full-window session history. Returns one row per
    session: user_id, date, tconst, genres (list), averageRating."""
    # wiki is used only for its date column (the daily index spanning the
    # analysis window) -- see the module docstring for why its traffic curve
    # is not used to modulate per-user session volume.
    window_days = wiki["date"].sort_values().reset_index(drop=True)
    match_day_set = set(matches["date"].dt.normalize())
    is_match_day = window_days.dt.normalize().isin(match_day_set).to_numpy()

    all_genres, genre_pools = _build_genre_pools(catalog)
    n_days = len(window_days)

    session_records = []
    for row in users.itertuples(index=False):
        # expected sessions per day for this user: a roughly constant base
        # rate, boosted on real match days in proportion to this user's
        # individual sports_affinity (the deliberately synthetic, hypothesis-
        # carrying trait -- see module docstring).
        match_multiplier = np.where(is_match_day, 1.0 + row.sports_affinity * MATCH_DAY_BOOST, 1.0)
        expected_per_day = row.base_activity_rate * match_multiplier
        n_sessions_per_day = rng.poisson(expected_per_day)

        total_sessions = int(n_sessions_per_day.sum())
        if total_sessions == 0:
            continue

        # which day each session happened on
        day_indices = np.repeat(np.arange(n_days), n_sessions_per_day)
        session_dates = window_days.to_numpy()[day_indices]

        # which primary genre each session draws from, per this user's taste
        genre_probs = rng.dirichlet(np.full(len(all_genres), row.genre_concentration))
        genre_choices = rng.choice(all_genres, size=total_sessions, p=genre_probs)

        for genre, count in zip(*np.unique(genre_choices, return_counts=True)):
            pool = genre_pools[genre]
            picks = rng.choice(len(pool), size=int(count), p=pool["_weight"].to_numpy())
            picked_dates = session_dates[genre_choices == genre]
            for date, idx in zip(picked_dates, picks):
                title_row = pool.iloc[idx]
                session_records.append(
                    {
                        "user_id": row.user_id,
                        "date": date,
                        "tconst": title_row["tconst"],
                        "genres": title_row["genres"],
                        "averageRating": title_row["averageRating"],
                    }
                )

    return pd.DataFrame(session_records)


def sample_churn(
    users: pd.DataFrame,
    full_window_features: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample churn labels from a logistic function of standardized
    full-window features, plus noise. Returns user_id, churned, churn_day_offset."""
    merged = users.merge(full_window_features, on="user_id", how="left")
    # users with zero full-window sessions have no engagement signal at all --
    # treat them as maximally likely to churn (they never engaged) rather than
    # dropping them here; validate.py will report/flag them explicitly.
    for col in ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]:
        merged[col] = merged[col].fillna(0.0)

    z = lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
    logit = (
        CHURN_INTERCEPT
        + BETA_SPORTS * z(merged["sports_spike_engagement"])
        + BETA_DIVERSITY * z(merged["genre_diversity"])
        + BETA_COMPLETION * z(merged["avg_completion_rate"])
        + rng.normal(0, NOISE_SIGMA, size=len(merged))
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, prob)

    return pd.DataFrame({"user_id": merged["user_id"], "churned": churned})


def build_user_feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    matches, catalog, wiki = load_real_sources()
    all_genres = sorted({g for genres in catalog["genres"] for g in genres})

    print(f"[features] simulating {N_USERS} users, {len(catalog)}-title real catalog")
    users = simulate_user_traits(N_USERS, rng)
    sessions = simulate_sessions(users, matches, catalog, wiki, rng)
    print(f"[features] simulated {len(sessions)} full-window sessions")

    match_day_set = set(matches["date"].dt.normalize())

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "sports_spike_engagement": compute_sports_spike_engagement(
                    group["date"], match_day_set
                ),
                "genre_diversity": compute_genre_diversity(group["genres"].tolist(), all_genres),
                "avg_completion_rate": compute_avg_completion_rate(group["averageRating"]),
                "n_sessions": len(group),
            }
        )

    full_window_features = (
        sessions.groupby("user_id").apply(summarize, include_groups=False).reset_index()
    )
    print("[features] computed full-window ('true') features for churn sampling")

    churn = sample_churn(users, full_window_features, rng)
    users = users.merge(churn, on="user_id", how="left")
    users["churned"] = users["churned"].fillna(1).astype(int)  # never-active users treated as churned
    print(f"[features] sampled churn: {users['churned'].mean():.1%} overall churn rate")

    # churn day, only meaningful for churned users
    window_start, window_end = wiki["date"].min(), wiki["date"].max()
    window_len_days = (window_end - window_start).days
    churn_offsets = rng.integers(MIN_CHURN_DAY_OFFSET, window_len_days + 1, size=len(users))
    users["churn_day"] = np.where(
        users["churned"] == 1,
        window_start + pd.to_timedelta(churn_offsets, unit="D"),
        pd.NaT,
    )

    # truncate each churned user's sessions to before their churn day --
    # this is the OBSERVED history downstream steps actually use.
    sessions_with_churn = sessions.merge(users[["user_id", "churned", "churn_day"]], on="user_id")
    observed_mask = (sessions_with_churn["churned"] == 0) | (
        sessions_with_churn["date"] < sessions_with_churn["churn_day"]
    )
    observed_sessions = sessions_with_churn[observed_mask]
    n_truncated = len(sessions_with_churn) - len(observed_sessions)
    print(f"[features] truncated {n_truncated} post-churn sessions from observed history")

    observed_features = (
        observed_sessions.groupby("user_id").apply(summarize, include_groups=False).reset_index()
    )

    user_table = users[["user_id", "churned"]].merge(observed_features, on="user_id", how="left")
    n_zero_session = user_table["n_sessions"].isna().sum()
    print(f"[features] {n_zero_session} users have zero observed sessions (no pre-churn activity)")

    out = PROCESSED_DIR / "user_features_raw.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    user_table.to_csv(out, index=False)
    print(f"[features] wrote {out}")
    return user_table


if __name__ == "__main__":
    build_user_feature_table()
