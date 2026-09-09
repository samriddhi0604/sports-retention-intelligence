# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Sports-Driven Retention Intelligence: Churn Prediction & Win-Back Recommendations.**
A streaming-platform analytics project testing whether live-sports engagement predicts
subscriber retention, then acting on that finding with a churn classifier and an ML-based
hybrid recommendation engine (collaborative filtering + content-based cold-start handling).
Built for a data/analytics internship application (JioStar) whose JD
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
- `requirements.txt`: pandas, numpy, scipy, scikit-learn, joblib, matplotlib, pytest
  (scikit-learn's `TruncatedSVD` covers the matrix factorization step; add `implicit` only if
  ALS on implicit feedback ends up being a better fit than SVD for the actual interaction data).
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
- **First, generate the session-level simulation explicitly** (this underlies everything else
  in this sub-step and is required as its own persisted artifact for Step 5, not just an
  intermediate variable): for each simulated user, generate a sequence of viewing sessions
  across the chosen time window, where each session picks a real title (by `tconst`) from the
  cleaned IMDb catalog, weighted by real popularity (e.g. IMDb `numVotes`) so popular titles get
  picked more often, similar to real streaming behavior. Each session record needs
  `user_id`, `tconst`, `timestamp`. This is the same session-generation logic already used to
  compute the aggregate features below -- make sure it is written once as its own function/
  script and both the aggregation step and Step 5 call it (or read its persisted output),
  rather than two different re-implementations of "how a user picks something to watch."
- Build `sports_spike_engagement` per user as a real measured quantity: for each real Cricsheet
  match timestamp, define a viewing window (e.g. +/- 2 hours) and calculate what fraction of a
  user's sessions (from the session log just generated) fall inside vs. outside those windows --
  this replaces the synthetic version with something actually computed from real interaction
  timestamps against a real match calendar.
- Build `genre_diversity` from real genre metadata (IMDb `title.basics.tsv.gz` genres, or TMDb's
  genre fields) attached to each session's `tconst`, aggregated per user, and
  `avg_completion_rate` from whatever real completion/rating proxy is available (note explicitly
  if a true completion-rate field isn't available and a proxy, like rating given or vote count,
  is used instead).
- Write unit tests confirming the feature calculations against a small hand-constructed example
  where the expected answer is known, before trusting them on the full dataset.

**1e. Validation (`data/validate.py`):**
- Sanity-check distributions after feature engineering (no user with >100% sports engagement,
  no negative session counts, reasonable date ranges) and fail loudly (raise an error) rather
  than silently continuing if a check fails.
- Save **two** outputs to `data/processed/`, not one:
  1. `user_features.csv` -- the aggregated per-user feature table (sports_spike_engagement,
     genre_diversity, avg_completion_rate, churned) that Steps 2 and 4 read from.
  2. `sessions.csv` -- the **raw, per-session interaction log**: one row per simulated viewing
     event, with `user_id`, `title_id` (the real IMDb `tconst`, not the title string -- use
     `tconst` as the canonical item ID everywhere downstream so joins stay unambiguous even
     if two titles share a name), `timestamp`, and `session_type` (e.g. movie vs. sports-window
     viewing). **This raw log is required by Step 5** (gateway analysis needs per-user session
     ordering over time; collaborative filtering needs the full user-item interaction matrix) --
     it is easy to skip this and only keep the aggregated table, since Steps 2-4 don't need it,
     but Step 5 will not work without it. Generate and persist it now even though it isn't used
     until Step 5, rather than needing to regenerate it later.
- Write a small `data/schema.md` documenting both output tables' columns and dtypes, since two
  different downstream steps (aggregate-feature consumers in Steps 2/4, and raw-session
  consumers in Step 5) both depend on this data staying consistent.
- Keep raw ingestion, cleaning, and feature engineering as separate, independently re-runnable
  stages rather than one monolithic script.

**Commit for this step:** break it into the sub-steps above rather than one commit -- e.g.
"Add data ingestion for Cricsheet and IMDb/TMDb sources", then "Add cleaning and timezone
alignment", then "Add session-level simulation and engagement feature engineering", then "Add
data validation and persist processed user_features/sessions tables". This keeps the history
honest and reviewable, and matches the incremental
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

### Step 5 — Engagement-uplift recommendation engine (ML-based, not rule-based)

This step replaces a simple rule ("recommend sports to low-sports users") with a proper
ML-based recommender, motivated by two real complexities a production streaming platform
actually faces (source: JioStar's own internal framing, relayed by the user from a company
presentation):

1. **Engagement is a general trait, not category-specific.** Heavy watchers of one category
   (e.g. IPL cricket) tend to be heavy watchers of unrelated categories too (e.g. reality TV).
   This means the right target population isn't "users with low sports engagement" -- it's
   **users with low overall engagement**, regardless of which category they're under-watching.
   The actionable question becomes: what content historically precedes a low-engagement user
   becoming a high-engagement one, and can that pathway be recommended to today's
   low-engagement users?
2. **Cross-genre/cross-language affinity can't be found via metadata.** A Hindi family drama
   and an English fantasy epic share no genre/language tags, yet the same users might watch
   both. Genre-tag-based recommendation structurally cannot discover this. Collaborative
   filtering on behavior (not metadata) can, since it finds latent similarity from co-viewing
   patterns rather than manual labels.

**5a. Identify "gateway" content (`recommendations/gateway_analysis.py`):**
- Among simulated users, identify those whose engagement ramps up over the simulated timeline
  (early sessions low, later sessions high -- define "ramp-up" with a clear, documented
  threshold, e.g. session count in the second half of the window is >2x the first half).
- For these "became a power user" users, look at what they watched in their *early*, low-
  engagement sessions (before the ramp-up) -- these are candidate gateway titles.
- Report the titles/genres that appear disproportionately often in early sessions of users who
  later ramped up, versus early sessions of users who never ramped up. This is a real,
  checkable analysis, not a hardcoded assumption -- if cricket/sports content doesn't actually
  show up as a strong gateway signal in the simulated data, report that honestly rather than
  forcing the sports narrative to hold.

**5b. Collaborative filtering via matrix factorization (`recommendations/collab_filter.py`):**
- Build the user-item interaction matrix from the simulated viewing sessions (real IMDb titles,
  as already implemented in `data/features.py`).
- Factorize it using SVD (`scipy.sparse.linalg.svds` or `sklearn.decomposition.TruncatedSVD` is
  sufficient for a project of this scale; ALS via an implicit-feedback library like `implicit`
  is a reasonable alternative if the interaction matrix is sparse/implicit rather than
  explicit-rating-based) into user embeddings and item embeddings, choosing a latent dimension
  (e.g. 20-50) via a quick reconstruction-error or held-out-interaction check, not an arbitrary
  guess.
- Compute item-item cosine similarity on the item embeddings. **Concretely inspect the result**
  for at least a handful of items: pull the top-5 nearest neighbors for a few titles spanning
  different genres/languages, and report whether any genuinely cross-genre/cross-language
  pairings emerge (this is the actual test of whether the hypothesis -- co-viewing reveals
  hidden affinity that metadata can't -- holds in this dataset). Report the real result,
  including if it's weaker or messier than the clean "Saas-Bahu meets Game of Thrones" story.
- For a given low-engagement user, generate recommendations by finding items with high
  embedding similarity to their few watched titles, weighted toward the gateway titles/genres
  identified in 5a.

**5c. Cold-start handling (`recommendations/hybrid_recommender.py`):**
- Pure collaborative filtering fails exactly for the population this project targets --
  low-activity users have too little interaction history for a reliable embedding. Address
  this directly rather than ignoring it:
- Build a hybrid score: blend the collaborative-filtering similarity score with a
  content-based similarity score (using real IMDb genre/rating metadata already ingested in
  Step 1) via a weighting that shifts toward content-based as a user's interaction count drops
  below a defined threshold (e.g. weight collaborative filtering at
  `min(1, interaction_count / 10)` and content-based at the remainder -- document whatever
  specific weighting function is actually used and why).
- This directly solves the practical problem: a user with only 2 watched titles still gets a
  sensible recommendation, grounded in metadata, rather than an unreliable embedding-based one.

**5d. Targeting and evaluation (`recommendations/evaluate_recommender.py`):**
- Define the target population as low-overall-engagement users (not low-sports-specifically),
  per the 5a framing. Note this overlaps with, but isn't identical to, the Step 4 churn model's
  "at-risk" flag -- Step 4's features (sports-spike engagement, genre diversity, completion
  rate) are themselves engagement proxies, so the two populations should correlate strongly, but
  report the actual overlap (e.g. "X% of low-engagement users are also flagged at-risk by the
  Step 4 model") rather than assuming they're the same group. Use low-overall-engagement (this
  step's definition) as the primary targeting criterion, since it's the more direct measure of
  the thing being acted on, and mention the churn-model overlap as a secondary cross-check.
- For this population, generate top-N recommendations via the hybrid recommender and report:
  how many recommended titles are "gateway" titles/genres identified in 5a, and (if feasible
  with the simulated data) a simple offline validation -- e.g. holding out each ramped-up
  user's actual early-session titles and checking whether the recommender would have surfaced
  them for a similar low-engagement user, as a rough proxy for "would this have worked."
- Report results honestly, including any weaknesses (e.g. if cold-start blending doesn't
  meaningfully outperform content-only for very sparse users, say so).

**Commit for this step:** break into sub-steps as above -- e.g. "Add gateway-content analysis
for engagement uplift", "Add collaborative filtering via matrix factorization", "Add hybrid
recommender with cold-start blending", "Add recommender evaluation and targeting logic".

**Framing for the README/resume (Step 8):** this step should be described as an ML-based
hybrid recommender addressing two specific, named production challenges (engagement is a
general trait rather than category-specific; cross-genre/cross-language affinity requires
behavioral rather than metadata-based similarity) -- not as "a recommendation system," which
undersells the actual design reasoning here.

### Step 6 — Visualization
- Add a few plots (matplotlib, saved as PNGs under `figures/`): the sports-engagement
  distribution split by churned/retained, the viewership time series with detected spikes
  overlaid, and the churn model's precision-recall curve.
- These make the project demoable in an interview without needing to run code live.
- **Commit:** "Add visualizations for engagement distributions, spikes, and model evaluation."

### Step 7 — Tests
- Write `pytest` tests: at minimum, a test that the hypothesis test function returns a sane
  p-value on a known synthetic case with an obvious effect, a test for the spike detector on a
  synthetic series with a known injected spike, a test that the matrix factorization reconstructs
  a small known interaction matrix reasonably well, and a test that the hybrid recommender's
  cold-start weighting shifts toward content-based scoring as interaction count drops, on a
  small hand-constructed example.
- **Commit:** "Add tests for hypothesis test, spike detection, and recommendation engine."

### Step 8 — Documentation
- Write `README.md`: the core question, data source (real or synthetic -- state clearly which),
  each component's method and result, how to run everything, and a "Known Limitations" section
  (e.g. simulated subscriber population, daily-granularity spike detection due to Wikipedia API
  limits, offline-only recommender evaluation with no real held-out user feedback, spike detection
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
    gateway_analysis.py
    collab_filter.py
    hybrid_recommender.py
    evaluate_recommender.py
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
    test_gateway_analysis.py
    test_collab_filter.py
    test_hybrid_recommender.py
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
- The recommendation engine is a real hybrid model: matrix-factorization-based collaborative
  filtering blended with content-based similarity for cold-start users. It is not a
  production-grade system (no online A/B testing, no real user feedback loop, offline
  evaluation only) -- describe it accurately as an offline-evaluated prototype if asked to go
  deeper, not as something validated with real user outcomes.
- The "gateway content" analysis (Step 5a) is a correlational finding from simulated data, not
  a causal claim -- if asked, be clear that it identifies content associated with later
  engagement ramp-up in the simulation, not proven to cause it.