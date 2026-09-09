# Data sources

`data/raw/` is gitignored (raw downloads, ~750MB total). Run `python data/ingest.py`
from the project root (with the `.venv` active) to re-download everything below.
Each source is fetched fresh from its origin -- no API key required for any of them.

## 1. Cricsheet -- IPL 2024 match calendar

- Source: https://cricsheet.org/downloads/ipl_json.zip (all-time IPL ball-by-ball
  data as JSON, ~5MB)
- Used for: the real event calendar. Filtered to `season == "2024"`, giving
  **71 matches** from 2024-03-22 to 2024-05-26 (this is Cricsheet's actual
  recorded count for the season, not a filtering artifact -- verified directly
  against the raw zip).
- Output: `data/raw/cricsheet_ipl_2024_matches.csv`
- Note: Cricsheet records match *dates* but not kickoff times, so downstream
  spike/engagement windows are defined at day granularity, not hour-of-day.

## 2. IMDb non-commercial datasets -- Indian film catalog

- Source: https://datasets.imdbws.com/ (`title.akas.tsv.gz`, `title.basics.tsv.gz`,
  `title.ratings.tsv.gz`; ~750MB combined, official free bulk exports)
- Used for: real genre metadata and a real rating/vote-count engagement proxy.
  `title.basics` has no region/language field itself, so `title.akas` is scanned
  first to find title ids with region `IN` or an Indian language code
  (hi/ta/te/kn/ml/bn/mr/pa/gu/ur); those ids are then used to filter
  `title.basics` (titleType == "movie", startYear >= 2000) and `title.ratings`.
- Result: 173,772 Indian-language movies (2000-2031, includes announced/upcoming
  titles), 121,361 of which have a rating.
- Output: `data/raw/imdb_indian_titles.csv`, `data/raw/imdb_indian_ratings.csv`
- TMDb was intentionally not used (would require an API key sign-up); IMDb's
  bulk datasets cover the same genre/rating need with no key.

## 3. Wikimedia pageviews REST API -- viewership proxy

- Source: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/...
  (free, no key), article `2024_Indian_Premier_League` on en.wikipedia,
  daily granularity, 2024-01-01 to 2024-06-30.
- Used for: **a structural stand-in for concurrent-viewer traffic.** No real
  streaming-platform viewership data is public anywhere (commercially
  sensitive). Wikipedia pageviews around real IPL match dates give a genuine
  baseline+event-spike pattern that can be honestly correlated against the
  real Cricsheet calendar -- but they are article views, not streaming
  viewership, and this is not claimed to be literal viewership data anywhere
  in this project.
- Note: the REST API only supports daily/monthly granularity per article
  (true hourly data only exists in raw all-Wikipedia dumps, which would mean
  scanning terabytes of all-article data to extract one article's hourly
  counts -- not practical here). All spike-detection work in this project is
  therefore done at daily granularity with a day-of-week seasonal baseline,
  not hour-of-day.
- Output: `data/raw/wikipedia_ipl2024_pageviews.csv`

## What's real vs. simulated

There is no public dataset of individual streaming-platform users' session
timestamps or churn outcomes (same sensitivity reason as #3 above, at an even
more sensitive per-user level). Per-user sessions, subscriptions, and churn
labels in this project are therefore simulated -- but every feature computed
from them is anchored to the real data above, not arbitrary parameters:

- `sports_spike_engagement` is computed by generating synthetic session
  timestamps and checking them against the **real** Cricsheet match-day
  calendar above.
- `genre_diversity` and `avg_completion_rate` are computed by having synthetic
  users "watch" titles sampled from the **real** IMDb Indian-film catalog
  above (real genre tags, and a completion-rate proxy anchored to the real
  ratings/votes distribution), not from made-up genre labels or a synthetic
  distribution unrelated to any real data.

See the root `README.md` (Step 8) for the full honesty/limitations statement.
