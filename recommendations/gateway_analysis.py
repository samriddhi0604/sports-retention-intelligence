"""
Step 5a -- Gateway content analysis.

Identifies users whose engagement ramped up over the analysis window (many
more sessions in the second half than the first), then checks what those
users watched EARLY -- before they ramped up -- looking for titles/genres
that show up disproportionately often compared to users who never ramped up.

This is a real, checkable analysis on the simulated session log, not a
hardcoded assumption that sports content is a gateway. If cricket-adjacent
viewing doesn't actually show up as a strong signal, that's reported as-is
(see the Honesty Note in the root README).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_PATH = PROCESSED_DIR / "gateway_analysis_results.json"

MIN_FIRST_HALF_SESSIONS = 3  # need a meaningful "before" period to compare at all
RAMP_UP_MULTIPLIER = 2.0  # second-half sessions must exceed this x first-half sessions
MIN_GROUP_OCCURRENCES = 5  # a title needs to appear for >=5 distinct ramped-up users to be reported
LAPLACE_SMOOTHING = 1.0


def get_window_midpoint(sessions: pd.DataFrame) -> pd.Timestamp:
    wiki = pd.read_csv(PROCESSED_DIR / "clean_wikipedia_pageviews.csv", parse_dates=["date"])
    start, end = wiki["date"].min(), wiki["date"].max()
    return start + (end - start) / 2


def classify_ramp_up(
    sessions: pd.DataFrame,
    midpoint: pd.Timestamp,
    min_first_half: int = MIN_FIRST_HALF_SESSIONS,
    multiplier: float = RAMP_UP_MULTIPLIER,
) -> pd.DataFrame:
    """Returns one row per user with first_half_count, second_half_count,
    eligible (had enough first-half activity to compare), and ramped_up."""
    sessions = sessions.copy()
    sessions["half"] = np.where(sessions["timestamp"] < midpoint, "first", "second")
    counts = (
        sessions.groupby(["user_id", "half"]).size().unstack(fill_value=0).reindex(
            columns=["first", "second"], fill_value=0
        )
    )
    counts.columns = ["first_half_count", "second_half_count"]
    counts = counts.reset_index()

    counts["eligible"] = counts["first_half_count"] >= min_first_half
    counts["ramped_up"] = counts["eligible"] & (
        counts["second_half_count"] > multiplier * counts["first_half_count"]
    )
    return counts


def compute_lift(
    early_sessions_ramped: pd.Series,
    early_sessions_never: pd.Series,
    min_occurrences: int = MIN_GROUP_OCCURRENCES,
    smoothing: float = LAPLACE_SMOOTHING,
) -> pd.DataFrame:
    """Given two Series of category values (genre or title_id) from each
    group's early sessions, return a DataFrame of lift = share-in-ramped /
    share-in-never, for categories occurring at least min_occurrences times
    in the ramped-up group."""
    ramped_counts = early_sessions_ramped.value_counts()
    never_counts = early_sessions_never.value_counts()

    all_categories = ramped_counts.index.union(never_counts.index)
    ramped_share = (ramped_counts.reindex(all_categories, fill_value=0) + smoothing) / (
        len(early_sessions_ramped) + smoothing * len(all_categories)
    )
    never_share = (never_counts.reindex(all_categories, fill_value=0) + smoothing) / (
        len(early_sessions_never) + smoothing * len(all_categories)
    )

    result = pd.DataFrame(
        {
            "category": all_categories,
            "count_in_ramped": ramped_counts.reindex(all_categories, fill_value=0).to_numpy(),
            "count_in_never": never_counts.reindex(all_categories, fill_value=0).to_numpy(),
            "lift": (ramped_share / never_share).to_numpy(),
        }
    )
    result = result[result["count_in_ramped"] >= min_occurrences]
    return result.sort_values("lift", ascending=False).reset_index(drop=True)


def main() -> dict:
    sessions = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["timestamp"])
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    catalog["genres"] = catalog["genres"].str.split(",")

    midpoint = get_window_midpoint(sessions)
    print(f"[gateway] window midpoint: {midpoint.date()}")

    ramp = classify_ramp_up(sessions, midpoint)
    n_eligible = int(ramp["eligible"].sum())
    n_ramped = int(ramp["ramped_up"].sum())
    print(
        f"[gateway] {n_eligible} users had >= {MIN_FIRST_HALF_SESSIONS} first-half sessions "
        f"(eligible for comparison); {n_ramped} of those ramped up "
        f"(second half > {RAMP_UP_MULTIPLIER}x first half)"
    )

    ramped_ids = set(ramp.loc[ramp["ramped_up"], "user_id"])
    never_ids = set(ramp.loc[ramp["eligible"] & ~ramp["ramped_up"], "user_id"])

    early_sessions = sessions[sessions["timestamp"] < midpoint]
    early_ramped = early_sessions[early_sessions["user_id"].isin(ramped_ids)]
    early_never = early_sessions[early_sessions["user_id"].isin(never_ids)]
    print(f"[gateway] {len(early_ramped)} early sessions from ramped-up users, "
          f"{len(early_never)} from never-ramped users")

    # title-level lift
    title_lift = compute_lift(early_ramped["title_id"], early_never["title_id"])
    top_titles = title_lift.head(15).merge(
        catalog[["tconst", "primaryTitle"]], left_on="category", right_on="tconst", how="left"
    )

    # genre-level lift (a session's title can have multiple genres, so this
    # explodes to one row per (session, genre) before counting)
    early_ramped_genres = early_ramped.merge(
        catalog[["tconst", "genres"]], left_on="title_id", right_on="tconst"
    ).explode("genres")["genres"]
    early_never_genres = early_never.merge(
        catalog[["tconst", "genres"]], left_on="title_id", right_on="tconst"
    ).explode("genres")["genres"]
    genre_lift = compute_lift(early_ramped_genres, early_never_genres, min_occurrences=10)

    # direct test: does watching during a real match window, early on,
    # predict later ramp-up? (session_type is the only "sports-adjacent"
    # signal in this dataset -- the catalog itself is movies, not cricket
    # broadcasts, so this is the honest way to check the sports-gateway
    # hypothesis specifically, separate from generic genre/title lift.)
    sports_share_ramped = float((early_ramped["session_type"] == "sports_window").mean())
    sports_share_never = float((early_never["session_type"] == "sports_window").mean())

    print("\n=== Top genres by early-session lift (ramped-up vs. never-ramped) ===")
    print(genre_lift.head(10).to_string(index=False))
    print("\n=== Top titles by early-session lift ===")
    print(top_titles[["primaryTitle", "count_in_ramped", "count_in_never", "lift"]].to_string(index=False))
    print(f"\n=== Sports-window share of early sessions: ramped-up={sports_share_ramped:.3f}, "
          f"never-ramped={sports_share_never:.3f} ===")

    top_genre = genre_lift.iloc[0]["category"] if len(genre_lift) > 0 else None
    sports_is_top_signal = (
        top_genre is not None and "sport" in str(top_genre).lower()
    ) or sports_share_ramped > sports_share_never * 1.2

    # honest caveat: with only ~2.8k early sessions from ramped-up users
    # spread across a 109k-title catalog, few individual titles clear the
    # min-occurrence bar at all, and none of the ones that do exceed
    # lift 1.0 -- title-level signal is too sparse here to show genuine
    # overrepresentation; genre-level lift (aggregated across many titles
    # per genre) is the more reliable read.
    title_level_note = (
        "title-level lift is weak/noisy at this session volume (max observed lift "
        f"{title_lift['lift'].max():.2f} among titles clearing the {MIN_GROUP_OCCURRENCES}-occurrence "
        "bar, i.e. even the least-underrepresented titles aren't actually overrepresented) -- "
        "genre-level lift above is the more trustworthy signal"
        if len(title_lift) > 0 and title_lift["lift"].max() < 1.0
        else "at least one individual title shows genuine (>1.0) overrepresentation"
    )
    print(f"\n[gateway] {title_level_note}")

    results = {
        "window_midpoint": str(midpoint.date()),
        "n_eligible_users": n_eligible,
        "n_ramped_up_users": n_ramped,
        "n_never_ramped_users": len(never_ids),
        "top_genres_by_lift": genre_lift.head(10).to_dict(orient="records"),
        "top_titles_by_lift": top_titles[
            ["primaryTitle", "count_in_ramped", "count_in_never", "lift"]
        ].to_dict(orient="records"),
        "title_level_note": title_level_note,
        "sports_window_share_early_ramped": sports_share_ramped,
        "sports_window_share_early_never_ramped": sports_share_never,
        "sports_content_is_a_gateway_signal": sports_is_top_signal,
        "honest_note": (
            "Sports-window viewing does appear as a gateway signal in this run."
            if sports_is_top_signal
            else "Sports-window viewing does NOT stand out as a gateway signal in this run "
            "-- the top lift signal is genre/title-driven, not match-timing-driven. "
            "Reported honestly rather than forcing the sports narrative."
        ),
    }

    print(f"\n{results['honest_note']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[gateway] wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
