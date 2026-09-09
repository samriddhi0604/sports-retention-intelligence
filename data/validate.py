"""
Step 1e -- Validation.

Reads data/processed/user_features_raw.csv and data/processed/sessions_raw.csv
(both written by features.py), sanity-checks each independently of the code
that produced it, and fails loudly (raises) rather than silently continuing
if a check fails. Saves the final validated tables to
data/processed/user_features.csv and data/processed/sessions.csv -- the only
files analysis/training/recommendation steps (2-5) should read from -- plus
data/schema.md documenting both.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

MIN_PLAUSIBLE_CHURN_RATE = 0.05
MAX_PLAUSIBLE_CHURN_RATE = 0.60


class ValidationError(Exception):
    pass


def validate_user_features(df: pd.DataFrame) -> None:
    if not df["user_id"].is_unique:
        raise ValidationError("user_id is not unique")

    if not df["churned"].isin([0, 1]).all():
        raise ValidationError("churned contains values other than 0/1")

    churn_rate = df["churned"].mean()
    if not (MIN_PLAUSIBLE_CHURN_RATE <= churn_rate <= MAX_PLAUSIBLE_CHURN_RATE):
        raise ValidationError(
            f"churn rate {churn_rate:.1%} outside plausible range "
            f"[{MIN_PLAUSIBLE_CHURN_RATE:.0%}, {MAX_PLAUSIBLE_CHURN_RATE:.0%}]"
        )

    fraction_cols = ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]
    for col in fraction_cols:
        observed = df[col].dropna()
        if (observed < 0).any() or (observed > 1).any():
            raise ValidationError(f"{col} has values outside [0, 1]")

    if (df["n_sessions"].dropna() < 0).any():
        raise ValidationError("n_sessions contains negative counts")

    # A user with zero observed (pre-churn) sessions has no basis for any of
    # the three engagement features -- they come through as NaN. There's no
    # sensible value to impute (a 0 would falsely claim "definitely no sports
    # engagement" rather than "unknown"), so they're dropped from the
    # modeling table rather than silently kept or fabricated a value.
    print(f"[validate] {len(df)} rows before dropping zero-session users")


def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=["sports_spike_engagement", "genre_diversity", "avg_completion_rate"])
    n_dropped = n_before - len(df)
    print(f"[validate] dropped {n_dropped} users with zero observed sessions (no engagement signal)")

    df["n_sessions"] = df["n_sessions"].astype(int)

    print(f"[validate] final table: {len(df)} users, churn rate {df['churned'].mean():.1%}")
    print(df[["sports_spike_engagement", "genre_diversity", "avg_completion_rate", "n_sessions"]].describe())

    out = PROCESSED_DIR / "user_features.csv"
    df.to_csv(out, index=False)
    print(f"[validate] wrote {out}")
    return df


def validate_and_save_sessions(user_ids: pd.Series) -> pd.DataFrame:
    sessions = pd.read_csv(
        PROCESSED_DIR / "sessions_raw.csv", parse_dates=["timestamp"]
    )
    catalog = pd.read_csv(PROCESSED_DIR / "clean_imdb_catalog.csv")
    wiki = pd.read_csv(PROCESSED_DIR / "clean_wikipedia_pageviews.csv", parse_dates=["date"])

    n_before = len(sessions)
    # referential integrity: every session's user must survive in the final
    # (post zero-session-drop) user table, and every title must be a real
    # catalog title
    sessions = sessions[sessions["user_id"].isin(user_ids)]
    bad_titles = ~sessions["title_id"].isin(catalog["tconst"])
    if bad_titles.any():
        raise ValidationError(f"{bad_titles.sum()} sessions reference a title_id not in the IMDb catalog")

    window_start, window_end = wiki["date"].min(), wiki["date"].max()
    out_of_range = (sessions["timestamp"] < window_start) | (sessions["timestamp"] > window_end)
    if out_of_range.any():
        raise ValidationError(f"{out_of_range.sum()} sessions fall outside the analysis window")

    if not sessions["session_type"].isin(["movie", "sports_window"]).all():
        raise ValidationError("session_type contains an unexpected value")

    print(f"[validate] sessions: {n_before} -> {len(sessions)} rows after filtering to valid users")

    out = PROCESSED_DIR / "sessions.csv"
    sessions.to_csv(out, index=False)
    print(f"[validate] wrote {out}")
    return sessions


def write_schema_doc(user_features: pd.DataFrame, sessions: pd.DataFrame) -> None:
    schema_path = Path(__file__).resolve().parent / "schema.md"
    lines = [
        "# Processed data schema",
        "",
        "Both tables are written by data/validate.py from data/features.py's raw output.",
        "Steps 2-5 should read only these two files, not the `_raw` intermediates.",
        "",
        "## data/processed/user_features.csv",
        "",
        "One row per simulated subscriber who had at least one observed (pre-churn) session.",
        "",
        "| column | dtype | meaning |",
        "|---|---|---|",
        "| user_id | int | unique subscriber id |",
        "| churned | int (0/1) | 1 if the user churned during the analysis window |",
        "| sports_spike_engagement | float [0,1] | fraction of the user's observed sessions on a real IPL 2024 match day |",
        "| genre_diversity | float [0,1] | normalized Shannon entropy of genre tags across the user's watched titles |",
        "| avg_completion_rate | float [0,1] | mean IMDb rating/10 of the user's watched titles (completion-rate proxy) |",
        "| n_sessions | int | number of observed sessions the above are computed from |",
        "",
        f"Row count: {len(user_features)}. Dtypes:",
        "```",
        str(user_features.dtypes),
        "```",
        "",
        "## data/processed/sessions.csv",
        "",
        "One row per simulated viewing session (the raw interaction log Step 5 needs for",
        "gateway analysis and collaborative filtering). Same OBSERVED (pre-churn-truncated)",
        "session set that user_features.csv is aggregated from -- not the full-window",
        "hypothetical history used internally for churn sampling.",
        "",
        "| column | dtype | meaning |",
        "|---|---|---|",
        "| user_id | int | matches user_features.csv user_id |",
        "| title_id | str | real IMDb tconst -- canonical item id, use this (not title strings) for joins |",
        "| timestamp | date | session date (day granularity -- Cricsheet has no match kickoff times) |",
        "| session_type | str | 'sports_window' if timestamp is a real IPL 2024 match day, else 'movie' |",
        "",
        f"Row count: {len(sessions)}. Dtypes:",
        "```",
        str(sessions.dtypes),
        "```",
    ]
    schema_path.write_text("\n".join(lines))
    print(f"[validate] wrote {schema_path}")


def main() -> None:
    raw_path = PROCESSED_DIR / "user_features_raw.csv"
    df = pd.read_csv(raw_path)
    print(f"[validate] loaded {raw_path}: {df.shape}")

    validate_user_features(df)
    print("[validate] all pre-drop sanity checks passed")

    df = clean_and_save(df)

    # Re-validate the final table too -- catches any issue introduced by the
    # drop/cast step itself, not just the raw input.
    validate_user_features(df)
    print("[validate] final table re-validated successfully")

    sessions = validate_and_save_sessions(df["user_id"])
    write_schema_doc(df, sessions)


if __name__ == "__main__":
    main()
