# Sports-Driven Retention Intelligence

Churn prediction and win-back recommendations for a streaming platform, built to test one
question end-to-end rather than as three disconnected demos.

## The Core Question

Does engagement with live sports moments reduce subscriber churn on a streaming platform, and
can that insight drive personalized win-back recommendations? Every component below acts on
that single thread: a statistical test establishes the finding, a classifier operationalizes it,
and a recommendation layer acts on it.

Built for a data/analytics internship application (JioStar) whose JD calls out sports, customer
engagement, monetisation, recommendations/personalisation, and statistical experimentation.

## Data: what's real and what's simulated

**Real data, used directly:**

- **[Cricsheet](https://cricsheet.org)** — the IPL 2024 match calendar. 71 real matches,
  2024-03-22 to 2024-05-26 (Cricsheet's own recorded count for the season, verified directly
  against the raw data, not a filtering artifact). Cricsheet records match *dates* but not
  kickoff times, so every "match window" in this project is day-granularity, not hourly.
- **[IMDb non-commercial datasets](https://datasets.imdbws.com)** — a real catalog of
  109,854 Indian-language movies (2000–2024, filtered via `title.akas` region/language,
  since `title.basics` itself has no language field) with real genre tags and real
  rating/vote-count data.
- **[Wikimedia pageviews REST API](https://wikimedia.org/api/rest_v1/)** — daily pageviews for
  the "2024 Indian Premier League" English Wikipedia article, 2024-01-01 to 2024-06-30, as a
  **structural stand-in for concurrent-viewer traffic**. No real streaming-platform viewership
  data is public anywhere (commercially sensitive) — this is article traffic, not literal
  viewership, and is never presented as anything else.

**Simulated, and stated plainly as such:** there is no public dataset of individual streaming
users' sessions or churn outcomes (same sensitivity reason as above, at an even more sensitive
per-user level). 5,000 synthetic subscribers were generated — but every feature computed from
them is grounded in the real data above, not arbitrary parameters:

- A user's `sports_spike_engagement` is computed from synthetic session *timestamps* checked
  against the **real** Cricsheet match calendar.
- Every simulated session picks a **real** title from the real IMDb catalog (popularity-weighted,
  like real streaming behavior), so `genre_diversity` and `avg_completion_rate` (a rating-based
  proxy — IMDb has no completion-rate field) come from real genre tags and real ratings, not
  invented values.
- Churn itself is sampled from a documented logistic function of these features plus substantial
  random noise (see `data/features.py`) — a real, controllable, but **intentionally not
  deterministic** relationship, calibrated to a believable ~0.80 AUC rather than a suspiciously
  clean one.

This is the honest framing if asked in an interview: *"I simulated a subscriber population with
a hypothesized relationship, grounded every feature in real match/content data, and built a
pipeline to test and act on it."* That is a legitimate and different claim from presenting
simulated findings as real-world results. See `data/README.md` for exact source URLs and
re-download instructions, and `data/schema.md` for the two processed table schemas.

## How to run everything

```bash
python -m venv .venv
pip install -r requirements.txt   # or .venv/Scripts/pip.exe install -r requirements.txt on Windows

python data/ingest.py             # downloads ~750MB of real data into data/raw/
python data/clean.py
python data/features.py           # simulates 5,000 users + sessions (~30-60s)
python data/validate.py           # -> data/processed/user_features.csv, sessions.csv

python analysis/hypothesis_test.py
python analysis/spike_detection.py
python analysis/visualize.py      # -> figures/*.png

python training/train_churn_model.py   # -> model_artifacts/

python recommendations/gateway_analysis.py
python recommendations/collab_filter.py     # re-scans title.akas.tsv.gz for language (~1-2 min)
python recommendations/hybrid_recommender.py
python recommendations/evaluate_recommender.py
python recommendations/win_back.py

pytest tests/ -v                  # 23 tests
```

On Windows, replace `python` with `.venv\Scripts\python.exe` (or activate the venv first with
`.venv\Scripts\activate`) for each command above; on Linux/Mac, activate with
`source .venv/bin/activate` or replace `python` with `.venv/bin/python`.

## Methodology & Results

### 1. Data pipeline (`data/`)
Ingestion → cleaning (missing-value handling documented per field, dedup asserted, IST/UTC
timestamp alignment reasoned through explicitly) → feature engineering (session simulation
grounded in real data, see above) → validation (independent sanity checks, fails loudly on
violation). **4,979 users** survive (21 dropped for zero observed sessions — no engagement
signal to impute), **34.6% churn rate**, **280,585 simulated sessions**.

### 2. Hypothesis test (`analysis/hypothesis_test.py`) — the headline finding
Both groups failed Shapiro-Wilk normality, so a **Mann-Whitney U test** was used instead of
assuming a t-test was valid.

| | Retained (n=3,257) | Churned (n=1,722) |
|---|---|---|
| Mean sports-spike engagement | **46.5%** | **23.0%** |

**p ≈ 1.18 × 10⁻²³²**. Retained users engage with sports moments roughly twice as much as
churned users.

### 3. Viewership spike detection (`analysis/spike_detection.py`)
Day-of-week median/MAD baseline (not a rolling window, which would be dominated by the spikes
themselves — IPL 2024 has matches on ~91% of days *within* its season) — evaluated at day
granularity against the real match calendar (the Wikimedia API only exposes daily granularity
per article, so this isn't hour-level as originally scoped).

| Metric | Value |
|---|---|
| Recall (event days caught) | **59.0%** (36/61) |
| Precision (flagged days that were real) | **97.3%** |
| F1 | **0.73** |

### 4. Churn classification model (`training/train_churn_model.py`)
Logistic Regression, stratified 80/20 split, threshold tuned to maximize F1 via 5-fold
cross-validated predictions **on the training set only** (test set untouched until final scoring).

| Metric | Value |
|---|---|
| Precision | 0.899 |
| Recall | 0.570 |
| F1 | 0.698 |
| ROC-AUC | 0.768 |

Standardized coefficients: `sports_spike_engagement` **-1.47** (strongest), `genre_diversity`
-0.68, `avg_completion_rate` -0.28 — all negative, correctly matching the direction and relative
ordering of the ground-truth generating process.

### 5. Recommendation engine (`recommendations/`)
An ML-based hybrid recommender (not a generic "watch more" rule), addressing two specific
production challenges: engagement is a general trait rather than category-specific, and
cross-genre/cross-language affinity requires behavioral (not metadata-based) similarity to
discover. A simple rule-based baseline (`win_back.py`) is kept alongside it for comparison.

**5a. Gateway analysis** — users whose session count ramped up mid-window (242 of 4,821
eligible) vs. never-ramped users, comparing what they watched *before* ramping up.
**Honest null result: sports-window viewing is NOT a gateway signal** (18.5% vs. 15.6% of early
sessions — not a meaningful difference). Genre-level lift is weak (War leads at 1.18x);
title-level lift is too sparse/noisy at this session volume to show genuine overrepresentation
(max observed lift 0.44, i.e. below 1.0). Reported as-is rather than forcing the sports
narrative to hold.

**5b. Collaborative filtering** (`collab_filter.py`) — user-item matrix (4,974 × 29,930,
IDF-weighted to correct an empirically-confirmed popularity-dominance artifact in raw counts),
factorized with `TruncatedSVD` (k=10, chosen via held-out-cell RMSE). 96.7% of inspected
nearest-neighbor pairs are cross-genre-or-cross-language — but this is **explicitly not reported
as genuine discovered taste affinity**: the session generator never encoded deliberate
item-to-item affinity beyond genre preference, popularity bias is empirically weak
(r=0.05 on random pairs), and the pattern is more consistent with a low-dimensional-embedding
search-over-30k-candidates effect than real structure. A real per-title language lookup (pulled
from `title.akas.tsv.gz`, not in the cleaned catalog) made this check genuine rather than assumed.

**5c. Hybrid recommender** (`hybrid_recommender.py`) — blends CF similarity with a
content-based score (real IMDb genre/rating metadata) via `weight_cf = min(1, distinct_items/10)`.
A cold-start-handling bug was caught and fixed during development: a user whose only watched
title was too rare to have a CF embedding initially got all-zero scores, because content vectors
were mistakenly scoped to the same CF-restricted item list.

**5d. Targeting & evaluation** (`evaluate_recommender.py`) — target population is
**low-overall-engagement users** (bottom 25% by session count, 1,248 users), not
low-sports-specifically, per 5a's framing. This overlaps with but isn't identical to Step 4's
at-risk flag (1,132 users): 64.7%/71.4% overlap in each direction, reported rather than assumed.
Offline validation (hold out each 5a ramped-up user's early titles, check if the recommender
surfaces them from a later-sessions-only profile): **hybrid 21.1% vs. content-only 0.8%** hit
rate — but the CF embeddings were trained on the full session log including these same
"held-out" titles, so **this gap reflects training leakage, not clean generalization**; the
content-only number is the more trustworthy of the two. Flagged explicitly rather than presented
as a validated win.

**Rule-based baseline** (`win_back.py`) — of the 1,132 at-risk users, 1,120 (98.9%) fall below
the retained-user average sports engagement and get a sports-content recommendation. This is an
expected, nearly mechanical consequence of `sports_spike_engagement` being the churn model's
dominant coefficient, not a new discovery.

### 6. Visualizations (`figures/`)
`engagement_distribution.png`, `spike_timeline.png` (visually shows recall degrading as the
season's baseline itself rises), `precision_recall_curve.png`.

### 7. Tests (`tests/`)
23 tests across 6 files, each against small hand-constructed cases with independently-verified
expected answers — feature computation, the hypothesis test's test-selection logic, spike
detection on an injected synthetic spike, SVD reconstruction on an exactly-low-rank matrix, and
an end-to-end example proving the hybrid recommender's cold-start weighting actually flips the
top recommendation as watched-item count crosses the threshold (not just that a formula returns
the right number).

## Known Limitations

- **Simulated subscriber population.** No public dataset of individual streaming users' sessions
  or churn exists. Every feature is grounded in real match/content data, but the users
  themselves, their session timestamps, and their churn outcomes are simulated with a
  documented, deliberately-not-deterministic hypothesized relationship — see `data/features.py`.
- **Day-granularity, not hour-granularity, spike detection and match windows.** The real API
  (Wikimedia) and real data source (Cricsheet) don't expose hourly data, so this is a scope
  adjustment from the original hourly design, not a choice made for convenience.
- **The gateway-content finding is a null result.** Sports-adjacent viewing did not emerge as a
  meaningful gateway signal in this simulated dataset. Reported honestly rather than forced.
- **The collaborative-filtering cross-genre/cross-language finding is not a validated
  discovery.** The high cross-genre neighbor rate is more consistent with a
  low-dimensional-embedding artifact than genuine taste affinity, given the session generator
  never encoded any such affinity to begin with. See `recommendations/collab_filter.py`'s
  diagnostics.
- **The offline recommender evaluation has data leakage** through the CF training pathway (see
  Step 5d above) and is a rough proxy on simulated data with no real held-out user feedback —
  not a validated production evaluation.
- **The recommender is an offline-evaluated prototype**, not a production system: no online A/B
  testing, no real user feedback loop, no real held-out interactions.
- **The win-back rule-based baseline's "recommend sports" finding is close to tautological** —
  it follows almost mechanically from the churn model's own strongest coefficient, not an
  independent discovery.

## Honesty Note (if asked in an interview)

- Every p-value, model metric, and recall/precision number above comes from an actual run in
  this repo, copied from saved output — never estimated.
- "I simulated data with a hypothesized relationship and built a pipeline to test and act on it"
  is the accurate description of Steps 1, 2, and 4. It is a different and more defensible claim
  than presenting simulated findings as real-world results.
- The recommendation engine is a real hybrid model (SVD-based collaborative filtering blended
  with content-based similarity for cold start) — but its two most interesting findings
  (cross-genre affinity, and cold-start blending's offline lift) both turned out, under scrutiny,
  to be artifacts rather than validated discoveries, and are reported that way rather than
  polished into a cleaner story.
- The "gateway content" analysis is correlational on simulated data, not causal, and specifically
  did not find sports content to be a meaningful gateway in this run.
