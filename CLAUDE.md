# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Sports-Driven Retention Intelligence: Churn Prediction & Win-Back Recommendations.**
A streaming-platform analytics project testing whether live-sports engagement predicts
subscriber retention, then acting on that finding with a churn classifier and a win-back
recommendation layer. Built for a data/analytics internship application (JioStar) whose JD
specifically calls out content/show performance, sports, customer engagement, monetisation,
recommendations/search/personalisation, and statistical experimentation -- this project is
designed to hit all of those directly rather than being a generic churn-prediction demo.

**Nothing exists yet.** Build the entire project from scratch, following the steps below in
order. Each step should run and produce real output before moving to the next.

## The Core Question

Does engagement with live sports moments reduce subscriber churn on a streaming platform, and
can that insight drive personalized win-back recommendations? This is the single thread tying
together every component below -- don't let it drift into three disconnected mini-projects.

## Build Order

### Step 0 — Scaffolding
- Initialize the directory structure below.
- `requirements.txt`: pandas, numpy, scipy, scikit-learn, joblib, matplotlib, pytest.
- Initialize git (`git init`) and create a `.gitignore` (exclude `.venv/`, `__pycache__/`,
  `*.joblib` if artifacts get large, `.env`).
- **Commit:** "Initial project scaffolding and requirements."

### Step 1 — Real data processing pipeline

Prefer real data over synthetic wherever possible. This step is the most important one to get
right, since every downstream result (hypothesis test, churn model, spike detection) is only as
credible as the data feeding it.

**1a. Data sources to use (India/JioStar-specific, in order of preference):**
- **Sports event calendar (cricket):** [Cricsheet.org](https://cricsheet.org) — free, official
  ball-by-ball data for IPL, international matches, and most major tournaments, downloadable as
  YAML/CSV/JSON with no API key required. Use match dates/times as the real event calendar for
  spike-correlation, and optionally use ball-by-ball detail (wickets, boundaries) to identify
  peak-drama moments within a match, not just "a match happened."
- **User engagement / content data (Indian cinema):**
  - [IMDb non-commercial datasets](https://datasets.imdbws.com) -- free official bulk downloads.
    `title.basics.tsv.gz` has region/language fields (filter to Hindi/Tamil/Telugu/etc. for
    Indian film coverage); `title.ratings.tsv.gz` gives real rating/vote-count data usable as an
    engagement proxy.
  - [TMDb API](https://www.themoviedb.org/documentation/api) -- free API key, solid Bollywood/
    regional Indian film coverage with genres, popularity scores, and release dates; use this for
    real genre-diversity features instead of synthetic genre tags.
  - Alternative with zero API setup: search Kaggle for a pre-packaged "Bollywood movies dataset"
    or "Indian movies dataset" CSV.
- **Concurrent-viewer / traffic time series:** no real streaming-platform viewership data is
  public anywhere (commercially sensitive, India or otherwise). Use a structural analog with the
  same baseline+event-spike pattern -- e.g. Wikipedia pageview spikes (via the Wikimedia REST
  API, free, no key) around real Cricsheet match dates, which can be genuinely correlated against
  the real match calendar. State clearly in the README that this stands in structurally for
  viewership, not literally.
- Only fall back to fully synthetic data (`data/generate_synthetic_data.py`) if none of the
  above are accessible in the build environment (e.g. no internet access).

**1b. Ingestion (`data/ingest.py`):**
- Download/load each raw source into `data/raw/` (do not commit large raw files to git --
  add `data/raw/` to `.gitignore` and instead commit a small `data/README.md` explaining how to
  re-download each source).
- Log basic shape/schema on load (row counts, column dtypes, date ranges) so data issues are
  caught immediately, not discovered three steps later.

**1c. Cleaning (`data/clean.py`):**
- Handle missing values explicitly -- decide and document (don't silently drop) whether to
  impute, drop, or flag missing engagement/rating values.
- De-duplicate on the natural key (user_id + timestamp, or event_id) and assert uniqueness
  afterward with a test, not just an eyeballed check.
- Align timezones/timestamp formats across sources before any join -- a sports-schedule dataset
  and a viewership dataset in different timezones will silently corrupt the spike-detection
  results in Step 3 if not caught here.
- Filter to a defensible, documented time window (e.g. one full season, or a fixed N-month
  range) rather than an arbitrary subset -- state the window and why it was chosen.

**1d. Feature engineering (`data/features.py`):**
- Build `sports_spike_engagement` per user as a real measured quantity: for each real Cricsheet
  match timestamp, define a viewing window (e.g. +/- 2 hours) and calculate what fraction of a
  user's sessions fall inside vs. outside those windows -- this replaces the synthetic version
  with something actually computed from real interaction timestamps against a real match calendar.
- Build `genre_diversity` from real genre metadata (IMDb `title.basics.tsv.gz` genres, or TMDb's
  genre fields, joined against whichever titles a simulated/real user history includes),
  `avg_completion_rate` from whatever real completion/rating proxy is available (note explicitly
  if a true completion-rate field isn't available and a proxy, like rating given or vote count,
  is used instead).
- Write unit tests confirming the feature calculations against a small hand-constructed example
  where the expected answer is known, before trusting them on the full dataset.

**1e. Validation (`data/validate.py`):**
- Sanity-check distributions after feature engineering (no user with >100% sports engagement,
  no negative session counts, reasonable date ranges) and fail loudly (raise an error) rather
  than silently continuing if a check fails.
- Save a small `data/processed/` output (the cleaned, feature-engineered table) that Steps 2-5
  read from -- keep raw ingestion, cleaning, and feature engineering as separate, independently
  re-runnable stages rather than one monolithic script.

**Commit for this step:** break it into the sub-steps above rather than one commit -- e.g.
"Add data ingestion for Cricsheet and IMDb/TMDb sources", then "Add cleaning and timezone
alignment", then "Add engagement feature engineering from real interaction data", then "Add data
validation checks". This keeps the history honest and reviewable, and matches the incremental
commit practice described later in this file.

**If real data genuinely isn't accessible in the build environment:** fall back to
`data/generate_synthetic_data.py` as originally planned, but the README (Step 8) must state
plainly that synthetic data was used and why, so this is never presented as a real-data result
by mistake.

### Step 2 — Hypothesis test (the core finding)
- Write `analysis/hypothesis_test.py`: compare sports-spike engagement between churned and
  retained users using an appropriate test (Welch's t-test if variances differ, or a
  non-parametric alternative if the distribution isn't roughly normal -- check first, don't
  default blindly).
- Report the test statistic, p-value, and group means. This result is the headline finding the
  rest of the project builds on -- don't bury it in a notebook, surface it in the README too.
- **Commit:** "Add hypothesis test for sports engagement vs. churn."

### Step 3 — Viewership spike detection
- Write `analysis/spike_detection.py`: detect anomalous concurrent-viewer spikes in the hourly
  time series. Use a seasonal-baseline approach (e.g. median/MAD per hour-of-day and
  day-of-week combination) rather than a naive rolling window, since a naive window gets
  distorted by the spike itself if the spike falls inside the averaging window -- this was a
  real bug encountered building the prototype version of this project, worth being aware of.
- Evaluate against the known injected event hours: report recall (event hours caught) and
  precision (fraction of flagged hours that were real events).
- **Commit:** "Add live-event viewership spike detection with seasonal baseline."

### Step 4 — Churn classification model
- Write `training/train_churn_model.py`: train a classifier (start with Logistic Regression --
  it's a good fit if the underlying relationship is roughly linear/logistic, and is more
  interpretable than a tree ensemble for explaining *why* a user is at risk, which matters for
  the win-back logic in Step 5).
- Use stratified train/test split; tune the decision threshold via the precision-recall curve
  (maximize F1) rather than the default 0.5 cutoff, especially given class imbalance in churn
  data.
- Report precision, recall, F1, ROC-AUC, and the standardized feature coefficients (direction
  and relative magnitude) -- the coefficients are what justify the win-back targeting logic
  in Step 5, so make sure they're actually inspected, not just the aggregate metrics.
- Save the model, scaler, and threshold as artifacts under `model_artifacts/`.
- **Commit:** "Add churn classification model with threshold tuning."

### Step 5 — Win-back recommendation logic
- Write `recommendations/win_back.py`: for users flagged at-risk by the Step 4 model, apply a
  rule that recommends sports content specifically to the segment whose historical sports
  engagement is below the retained-user average -- i.e. act on the Step 2 finding, not just
  produce a generic "watch more" recommendation.
- Report how many at-risk users fall into this segment vs. would get a different
  (non-sports-weighted) recommendation.
- This is intentionally a rule-based layer, not a full collaborative-filtering recommender --
  be upfront about that scope choice in the README (see Step 8) rather than overstating it as
  a production recommendation system.
- **Commit:** "Add win-back recommendation targeting based on churn model coefficients."

### Step 6 — Visualization
- Add a few plots (matplotlib, saved as PNGs under `figures/`): the sports-engagement
  distribution split by churned/retained, the viewership time series with detected spikes
  overlaid, and the churn model's precision-recall curve.
- These make the project demoable in an interview without needing to run code live.
- **Commit:** "Add visualizations for engagement distributions, spikes, and model evaluation."

### Step 7 — Tests
- Write `pytest` tests: at minimum, a test that the hypothesis test function returns a sane
  p-value on a known synthetic case with an obvious effect, a test for the spike detector on a
  synthetic series with a known injected spike, and a test for the win-back segmentation logic
  on a small hand-constructed DataFrame.
- **Commit:** "Add tests for hypothesis test, spike detection, and win-back logic."

### Step 8 — Documentation
- Write `README.md`: the core question, data source (real or synthetic -- state clearly which),
  each component's method and result, how to run everything, and a "Known Limitations" section
  (e.g. synthetic data if used, rule-based rather than ML-based recommender, spike detection
  tuned on injected synthetic events rather than validated against a real sports calendar).
- A clear-eyed limitations section is a positive signal in an interview, not something to hide.
- **Commit:** "Add README with methodology, results, and limitations."

## Target Project Structure

```
sports-retention-intelligence/
  data/
    raw/                          # downloaded MovieLens / sports-schedule / viewership sources (gitignored)
    processed/                    # cleaned, feature-engineered output that downstream steps read from
    ingest.py
    clean.py
    features.py
    validate.py
    generate_synthetic_data.py    # fallback only, clearly labeled if used
    README.md                     # how to re-download each real source
  analysis/
    hypothesis_test.py
    spike_detection.py
  training/
    train_churn_model.py
  recommendations/
    win_back.py
  model_artifacts/
    churn_model.joblib
    churn_scaler.joblib
    churn_threshold.joblib
  figures/
    engagement_distribution.png
    spike_timeline.png
    precision_recall_curve.png
  tests/
    test_data_features.py
    test_hypothesis.py
    test_spike_detection.py
    test_winback.py
  requirements.txt
  README.md
  .gitignore
```

## Git Commit Practice

Commit after each step above, in order, with a clear message describing what that step added --
this is standard incremental development practice and naturally produces a real, honest commit
history since each commit corresponds to an actual working increment of the project. Use `git
add` scoped to the files relevant to that step rather than `git add .` for everything at once,
so each commit's diff matches its message.

Do **not** backdate commits, manipulate `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE`, or otherwise
artificially spread commits across fake historical dates to simulate a longer development
timeline than what actually happened. If the whole project is built in one sitting, the commit
history should honestly reflect that -- multiple real commits made close together in real time
is completely normal and not something to disguise. A recruiter or interviewer who checks commit
timestamps and finds them artificially spread out (or reads code comments/methodology and asks
about a specific "week 3" decision that never happened) is a credibility risk far worse than an
honest same-day commit history.

## Honesty Note for Resume/Interview Use

- The hypothesis-test p-value, churn model metrics, and spike-detection recall/precision must
  all come from an actual run in this repo -- copy them from real output, never estimate.
- If synthetic data is used, say so plainly if asked in an interview: "I simulated data with a
  hypothesized relationship and built a pipeline to test and act on it" is a legitimate,
  understandable thing to say, and is a very different (and more defensible) claim than
  presenting synthetic findings as real-world results.
- The win-back "recommendation engine" is a rule-based targeting layer, not a trained
  recommendation model -- describe it accurately as that if asked to go deeper.