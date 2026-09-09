"""
Step 1b -- Ingestion.

Downloads three real data sources into data/raw/ and does the minimal parsing
needed to get each one into a flat table. No cleaning/joining/feature work
happens here -- that's clean.py and features.py. Every load logs its shape
and schema so problems are caught immediately.

Sources:
  - Cricsheet (cricsheet.org): ball-by-ball/match-level IPL data. We use it
    for the match calendar (dates), filtered to the IPL 2024 season
    (2024-03-22 .. 2024-05-26). No API key.
  - IMDb non-commercial datasets (datasets.imdbws.com): title.basics,
    title.ratings, title.akas. Used to build a real catalog of Indian-language
    films (genres + rating/vote engagement proxy). No API key.
  - Wikimedia pageviews REST API: daily pageviews for the
    "2024_Indian_Premier_League" English Wikipedia article, 2024-01-01 to
    2024-06-30 (covers pre-season baseline, the season itself, and a
    post-season tail). Stands in structurally for concurrent-viewer traffic
    -- see data/README.md for why. No API key.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
USER_AGENT = "sports-retention-intelligence-project (contact: rahul101000@gmail.com)"

IPL_SEASON = "2024"
WIKI_ARTICLE = "2024_Indian_Premier_League"
WIKI_START = "20240101"
WIKI_END = "20240630"
INDIAN_LANGUAGES = {"hi", "ta", "te", "kn", "ml", "bn", "mr", "pa", "gu", "ur"}
MIN_TITLE_YEAR = 2000


def _log_frame(name: str, df: pd.DataFrame) -> None:
    print(f"[ingest] {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[ingest] {name} dtypes:\n{df.dtypes}")
    for col in df.columns:
        if "date" in col.lower() or "year" in col.lower():
            print(f"[ingest] {name}.{col} range: {df[col].min()} .. {df[col].max()}")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[ingest] already downloaded, skipping: {dest}")
        return dest
    print(f"[ingest] downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    print(f"[ingest] done: {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def ingest_cricsheet() -> pd.DataFrame:
    """Download all-time IPL match data, filter to the 2024 season, return match-level rows."""
    zip_path = _download(
        "https://cricsheet.org/downloads/ipl_json.zip", RAW_DIR / "cricsheet" / "ipl_json.zip"
    )

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as f:
                match = json.load(f)
            info = match.get("info", {})
            if str(info.get("season")) != IPL_SEASON:
                continue
            dates = info.get("dates", [])
            outcome = info.get("outcome", {})
            rows.append(
                {
                    "match_id": name.replace(".json", ""),
                    "date": dates[0] if dates else None,
                    "season": info.get("season"),
                    "team1": info.get("teams", [None, None])[0],
                    "team2": info.get("teams", [None, None])[1]
                    if len(info.get("teams", [])) > 1
                    else None,
                    "venue": info.get("venue"),
                    "city": info.get("city"),
                    "event_name": info.get("event", {}).get("name"),
                    "match_number": info.get("event", {}).get("match_number"),
                    "winner": outcome.get("winner"),
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    _log_frame("cricsheet_ipl_2024_matches", df)

    out_path = RAW_DIR / "cricsheet_ipl_2024_matches.csv"
    df.to_csv(out_path, index=False)
    print(f"[ingest] wrote {out_path}")
    return df


def ingest_imdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download IMDb bulk datasets, filter to Indian-language movies, return (titles, ratings)."""
    akas_path = _download(
        "https://datasets.imdbws.com/title.akas.tsv.gz", RAW_DIR / "imdb" / "title.akas.tsv.gz"
    )
    basics_path = _download(
        "https://datasets.imdbws.com/title.basics.tsv.gz",
        RAW_DIR / "imdb" / "title.basics.tsv.gz",
    )
    ratings_path = _download(
        "https://datasets.imdbws.com/title.ratings.tsv.gz",
        RAW_DIR / "imdb" / "title.ratings.tsv.gz",
    )

    # title.basics has no region/language field -- that lives in title.akas.
    # Stream it in chunks (it's ~500MB uncompressed) and keep only the set of
    # title ids that have an Indian region or an Indian-language alternate title.
    print("[ingest] scanning title.akas.tsv.gz for Indian-region/language title ids...")
    indian_ids: set[str] = set()
    for chunk in pd.read_csv(
        akas_path,
        sep="\t",
        na_values="\\N",
        quoting=csv.QUOTE_NONE,
        usecols=["titleId", "region", "language"],
        chunksize=500_000,
        dtype=str,
    ):
        mask = (chunk["region"] == "IN") | (chunk["language"].isin(INDIAN_LANGUAGES))
        indian_ids.update(chunk.loc[mask, "titleId"].tolist())
    print(f"[ingest] found {len(indian_ids):,} title ids with an Indian region/language")

    print("[ingest] scanning title.basics.tsv.gz for Indian movies...")
    basics_chunks = []
    for chunk in pd.read_csv(
        basics_path,
        sep="\t",
        na_values="\\N",
        quoting=csv.QUOTE_NONE,
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
        chunksize=500_000,
        dtype=str,
    ):
        chunk = chunk[chunk["tconst"].isin(indian_ids) & (chunk["titleType"] == "movie")]
        basics_chunks.append(chunk)
    titles = pd.concat(basics_chunks, ignore_index=True)
    titles["startYear"] = pd.to_numeric(titles["startYear"], errors="coerce")
    titles = titles[titles["startYear"] >= MIN_TITLE_YEAR].reset_index(drop=True)
    titles = titles.drop(columns=["titleType"])
    _log_frame("imdb_indian_titles", titles)

    print("[ingest] scanning title.ratings.tsv.gz...")
    title_id_set = set(titles["tconst"])
    ratings_chunks = []
    for chunk in pd.read_csv(
        ratings_path,
        sep="\t",
        na_values="\\N",
        quoting=csv.QUOTE_NONE,
        dtype={"tconst": str, "averageRating": float, "numVotes": "Int64"},
        chunksize=500_000,
    ):
        ratings_chunks.append(chunk[chunk["tconst"].isin(title_id_set)])
    ratings = pd.concat(ratings_chunks, ignore_index=True)
    _log_frame("imdb_indian_ratings", ratings)

    titles_out = RAW_DIR / "imdb_indian_titles.csv"
    ratings_out = RAW_DIR / "imdb_indian_ratings.csv"
    titles.to_csv(titles_out, index=False)
    ratings.to_csv(ratings_out, index=False)
    print(f"[ingest] wrote {titles_out}")
    print(f"[ingest] wrote {ratings_out}")
    return titles, ratings


def ingest_wikipedia_pageviews() -> pd.DataFrame:
    """Download daily pageviews for the IPL 2024 Wikipedia article as a viewership proxy."""
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{WIKI_ARTICLE}/daily/{WIKI_START}/{WIKI_END}"
    )
    print(f"[ingest] fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.load(response)

    df = pd.DataFrame(payload["items"])
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df = df[["date", "views"]].sort_values("date").reset_index(drop=True)
    _log_frame("wikipedia_ipl2024_pageviews", df)

    out_path = RAW_DIR / "wikipedia_ipl2024_pageviews.csv"
    df.to_csv(out_path, index=False)
    print(f"[ingest] wrote {out_path}")
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Ingesting Cricsheet IPL 2024 match calendar ===")
    ingest_cricsheet()
    print("\n=== Ingesting IMDb Indian-language movie catalog ===")
    ingest_imdb()
    print("\n=== Ingesting Wikipedia pageviews (viewership proxy) ===")
    ingest_wikipedia_pageviews()
    print("\n[ingest] all sources ingested into data/raw/")


if __name__ == "__main__":
    main()
