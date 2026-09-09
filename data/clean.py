"""
Step 1c -- Cleaning.

Reads data/raw/*.csv, handles missing values explicitly, de-duplicates on
each table's natural key (asserting uniqueness afterward rather than just
eyeballing it), aligns timestamps, and filters to the documented analysis
window. Writes cleaned tables to data/processed/ so features.py can be run
independently of this stage.

Analysis window: 2024-01-01 to 2024-06-30. Chosen to bracket the real IPL
2024 season (2024-03-22 to 2024-05-26, from Cricsheet) with ~3 months of
pre-season and ~1 month of post-season baseline days on either side -- the
spike detector in analysis/spike_detection.py needs non-event days to
establish a seasonal baseline, so the window can't be just the season itself.

Timestamp alignment: Cricsheet match dates are calendar dates in India
(IST, UTC+5:30); Wikimedia's daily pageview buckets are UTC calendar days.
IPL matches are always played in the afternoon/evening IST (never within
5.5 hours of midnight IST), so an IST match date and its UTC calendar date
are always the same day -- no shift is needed. Both are treated here as
plain (timezone-naive) calendar dates for that reason, which is stated
explicitly rather than assumed silently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

WINDOW_START = pd.Timestamp("2024-01-01")
WINDOW_END = pd.Timestamp("2024-06-30")
CATALOG_MAX_YEAR = 2024  # drop titles not yet released as of the analysis window


def clean_cricsheet_matches() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "cricsheet_ipl_2024_matches.csv", parse_dates=["date"])

    before = len(df)
    df = df.drop_duplicates(subset=["match_id"])
    assert df["match_id"].is_unique, "match_id must be unique after dedup"
    print(f"[clean] cricsheet: {before} -> {len(df)} rows after de-dup on match_id")

    # match_number is missing for 4 rows -- these are playoff fixtures named
    # "Qualifier 1/2", "Eliminator", "Final" rather than numbered league
    # matches. Kept as-is (documented, not dropped): the calendar only needs
    # `date`, and playoff matches are real, high-engagement event days we
    # want in the spike calendar, not noise to remove.
    missing_match_number = df["match_number"].isna().sum()
    print(f"[clean] cricsheet: {missing_match_number} playoff rows have no match_number (kept)")

    df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].reset_index(drop=True)
    print(f"[clean] cricsheet: {len(df)} rows within analysis window {WINDOW_START.date()}..{WINDOW_END.date()}")

    out = PROCESSED_DIR / "clean_cricsheet_matches.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[clean] wrote {out}")
    return df


def clean_imdb_catalog() -> pd.DataFrame:
    titles = pd.read_csv(RAW_DIR / "imdb_indian_titles.csv")
    ratings = pd.read_csv(RAW_DIR / "imdb_indian_ratings.csv")

    before = len(titles)
    titles = titles.drop_duplicates(subset=["tconst"])
    assert titles["tconst"].is_unique, "tconst must be unique after dedup (titles)"
    ratings = ratings.drop_duplicates(subset=["tconst"])
    assert ratings["tconst"].is_unique, "tconst must be unique after dedup (ratings)"
    print(f"[clean] imdb titles: {before} -> {len(titles)} rows after de-dup on tconst")

    # Missing primaryTitle (2 rows): can't meaningfully use an unnamed title
    # in a catalog a user "watches" -- drop.
    n_missing_title = titles["primaryTitle"].isna().sum()
    titles = titles.dropna(subset=["primaryTitle"])
    print(f"[clean] imdb titles: dropped {n_missing_title} rows with missing primaryTitle")

    # Missing genres (3,985 rows): genre_diversity is computed directly from
    # this field, so a title with no genre can't contribute to it -- drop
    # rather than impute a fake genre.
    n_missing_genre = titles["genres"].isna().sum()
    titles = titles.dropna(subset=["genres"])
    print(f"[clean] imdb titles: dropped {n_missing_genre} rows with missing genres")

    # startYear > 2024 (18,009 rows): announced/upcoming titles that did not
    # exist for a user to have watched during the 2024 analysis window --
    # drop to keep the simulated watch-history catalog realistic.
    n_future = (titles["startYear"] > CATALOG_MAX_YEAR).sum()
    titles = titles[titles["startYear"] <= CATALOG_MAX_YEAR]
    print(f"[clean] imdb titles: dropped {n_future} rows with startYear > {CATALOG_MAX_YEAR}")

    # Inner join with ratings: avg_completion_rate is proxied from the
    # rating/vote-count fields, so a title with no rating can't be used for
    # that feature. Rather than imputing a fake rating, the usable catalog
    # is restricted to titles that have both real genres and a real rating.
    catalog = titles.merge(ratings, on="tconst", how="inner")
    print(
        f"[clean] imdb catalog: {len(titles)} genre-tagged titles + ratings inner join "
        f"-> {len(catalog)} usable titles (dropped {len(titles) - len(catalog)} with no rating)"
    )
    assert catalog["tconst"].is_unique, "tconst must be unique in the final catalog"

    catalog["genres"] = catalog["genres"].str.split(",")

    out = PROCESSED_DIR / "clean_imdb_catalog.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out, index=False)
    print(f"[clean] wrote {out}")
    return catalog


def clean_wikipedia_pageviews() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "wikipedia_ipl2024_pageviews.csv", parse_dates=["date"])

    before = len(df)
    df = df.drop_duplicates(subset=["date"])
    assert df["date"].is_unique, "date must be unique after dedup"
    print(f"[clean] wikipedia: {before} -> {len(df)} rows after de-dup on date")

    df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].reset_index(drop=True)

    full_range = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    missing_days = full_range.difference(df["date"])
    if len(missing_days) > 0:
        print(f"[clean] wikipedia: {len(missing_days)} missing day(s) in window: {list(missing_days.date)}")
        df = (
            df.set_index("date")
            .reindex(full_range)
            .rename_axis("date")
            .reset_index()
        )
        df["views"] = df["views"].interpolate(method="linear")
        print("[clean] wikipedia: gaps filled via linear interpolation (flagged above, not silent)")
    else:
        print("[clean] wikipedia: no missing days in window")

    out = PROCESSED_DIR / "clean_wikipedia_pageviews.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[clean] wrote {out}")
    return df


def main() -> None:
    print("=== Cleaning Cricsheet matches ===")
    clean_cricsheet_matches()
    print("\n=== Cleaning IMDb catalog ===")
    clean_imdb_catalog()
    print("\n=== Cleaning Wikipedia pageviews ===")
    clean_wikipedia_pageviews()
    print("\n[clean] all sources cleaned into data/processed/")


if __name__ == "__main__":
    main()
