"""
Step 1e -- Validation.

Reads data/processed/user_features_raw.csv (written by features.py),
sanity-checks it independently of the code that produced it, and fails
loudly (raises) rather than silently continuing if a check fails. Saves the
final validated table to data/processed/user_features.csv, which is the
only file analysis/training/recommendation steps (2-5) should read from.
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


if __name__ == "__main__":
    main()
