# Build Hurdles and How They Were Resolved

A chronological record of the real problems hit while building this project — data
discrepancies, bugs, and methodological issues — and how each was actually fixed. Design
*decisions* (day-granularity windows, sample size, effect strength, etc.) are covered in the
README and aren't repeated here; this file is specifically about things that were **wrong** and
had to be caught.

## Step 1 — Data pipeline

### IMDb's `title.basics` doesn't actually have a language field
My original plan was to filter Indian films using `title.basics.tsv.gz`'s region/language
fields. Those fields don't actually exist there — they live in `title.akas.tsv.gz`, a separate
513MB file. Fixed by streaming `title.akas` first (chunked, 1M rows at a time) to build a set of
Indian-region/language title IDs, then using that set to filter `title.basics`.

### Genre column didn't round-trip through CSV
The cleaning step converted the `genres` field to a Python list before saving to CSV. Pandas
serializes a list column as its Python `repr()` string (`"['Action', 'Drama']"`), which is
fragile to parse back. Fixed by keeping `genres` as a raw comma-separated string in the saved CSV
and splitting it downstream wherever it's actually consumed.

### Sports-engagement values saturated near 90% — a real modeling bug, not just bad luck
The first full simulation run produced sports-spike engagement of ~87-91% for retained users,
when only 39% of days in the analysis window are real match days. Dug into why and found the
root cause: IPL 2024 has matches on **~91% of days within its own ~66-day season** (71 matches,
doubleheaders included), so the Wikipedia "2024 Indian Premier League" article's traffic stays
elevated for nearly the entire season, not just individual match days. I'd been using that curve
as a general per-user daily-activity multiplier, which meant almost every session simulated
during the season landed on a match day regardless of a user's actual simulated sports interest —
a modeling mismatch, not a real signal.

Fixed by dropping the Wikipedia curve from per-user session-volume modeling entirely. Match-day
session timing is now driven directly off the real Cricsheet calendar, scaled per user by an
individual affinity trait; Wikipedia's role moved to where it's actually valid — the
aggregate-level viewership proxy used later for spike detection. Re-ran: retained users dropped
to a believable 47% average, churned to 22%, both close to the real 39% base rate.

### Churn-effect calibration took several iterations
The first calibrated run hit 0.838 cross-validated AUC — too clean for a believable, real-world
churn signal. Iteratively increased the churn-generating noise and adjusted the intercept to land
on ~0.80 AUC and a ~30-35% churn rate, checking with a quick logistic-regression cross-validation
after each change rather than guessing at parameters.

## Step 3 — Spike detection

### `fillna` crash on a raw numpy array
`df["modified_z"].fillna(np.where(...))` raised a `TypeError` — `fillna` doesn't accept a bare
ndarray, only a scalar, dict, or Series. Fixed by wrapping the fallback array in a `pd.Series`
with a matching index before passing it in.

## Mid-build — the recommendation-engine scope changed

Partway through, the win-back logic changed from a simple rule ("recommend sports to low-sports
users") to a full ML-based hybrid recommender — gateway analysis, collaborative filtering via
matrix factorization, and cold-start blending — which in turn meant the data pipeline needed to
start persisting a full per-session log to disk, something that had only ever existed as an
in-memory variable before.

Handled this as a purely additive change: same random seed, same code paths, just also writing
to disk what was already being computed. Verified this held by re-running the full pipeline
afterward and confirming the churn rate, the zero-session-user count, and the truncated-session
count were all identical to the pre-change run — nothing already reported earlier got quietly
recomputed.

## Collaborative filtering

### Crash at the very end from unserializable numpy types
After all the expensive computation (interaction matrix, a 4-way latent-dimension search, a
re-scan of the language data) had already finished, `json.dumps()` crashed on numpy `int64`
values in the id lists. Fixed by casting to native Python `int`/`list` before serializing — cheap
fix, but it meant losing a full run's worth of computation to a one-line bug.

### SVD's leading component was a popularity axis, not taste structure
Diagnosed directly rather than assumed: the first singular value from raw-count SVD was ~114,
nearly 3x the second (~41), and its share of explained variance (4.4%) dwarfed every other
component (~0.6-1.5% each) — the textbook signature of a dominant popularity axis swamping
genuine structure. Fixed by applying IDF weighting per item (the standard TF-IDF-style
correction for exactly this problem) before factorizing.

### Even after that fix, nearest-neighbor similarity stayed suspiciously high across every genre
IDF weighting didn't fix the underlying pattern: nearest neighbors for well-watched items stayed
at 0.92-0.98 cosine similarity regardless of genre or language, and the cross-genre rate actually
went up slightly. Tried a full TF-IDF scheme (normalizing by each user's total activity too) as a
further fix — this made things measurably worse (similarity flattened to 0.98-1.0 across the
board, explained variance went nearly flat across every component), so that approach was
dropped.

Rather than keep tuning until something "looked right," I ran direct diagnostics: checked the
background (random-pair) similarity distribution (genuinely varied, mean 0.11, real spread from
-0.93 to +0.99 — not a degenerate space), the correlation between item popularity and similarity
(weak, r=0.05), and the random-pair 99th percentile (0.79, still below the ~0.92-0.98
nearest-neighbor scores). Conclusion: the high similarity for well-watched items is most
consistent with a low-dimensional-embedding effect — searching for the single best match among
~30,000 candidates in only 10 dimensions tends to surface a spuriously high-similarity match
regardless of real structure — especially since the session-generation process never encoded any
deliberate item-to-item taste affinity to begin with, so there was no real signal there to find.
Wrote the final report to say this plainly rather than claim a positive discovery the numbers
superficially seemed to support.

## Hybrid recommender

### Cold-start users got all-zero recommendation scores
A user whose only watched title happened to be rare (watched only once across the whole
simulation, and therefore excluded from the collaborative-filtering item set by a minimum-count
filter) got every recommendation score as exactly 0.0. Root cause: content vectors had been built
only over that same restricted item list, so the user's one watched item had no content vector
either — nothing to build a profile from in either direction.

Fixed by decoupling the two item universes: the recommendable candidate set stays restricted to
items with enough interactions to trust (rare items aren't great recommendations anyway), but a
user's profile is now built from all of their real watched titles regardless of rarity, since
content-based scoring only needs a title's real genre/rating, not repeat interactions.

## Recommender evaluation

### The "content-only baseline" wasn't actually content-only
The first attempt at isolating a content-only comparison passed zeroed-out embeddings into the
existing recommendation function, expecting that to neutralize the collaborative-filtering term.
It didn't: the blending weight is driven by interaction count, not by the embeddings passed in,
so for active users the collaborative-filtering weight still came out at 1.0 — and with zeroed
embeddings, every item's score became exactly zero, an arbitrary tie-order rather than a genuine
content ranking. That produced a suspiciously clean 0.0% hit rate that didn't survive a second
look. Fixed by adding an explicit override that bypasses the weighting formula entirely instead
of trying to fake it through the inputs.

### That fix surfaced a second bug: watched items could be recommended back to the user
With the override in place, the "exclude already-watched items" logic (which only looped over
the collaborative-filtering profile list) excluded nothing, since that list was now forced empty.
Fixed by separating "which items to exclude from recommendations" from "which items feed the
collaborative-filtering profile" — two different questions that had been using the same list.

### Even after both fixes, the comparison had real data leakage
With both bugs fixed, the hybrid approach still vastly outperformed content-only (21.1% vs.
0.8% hit rate). Rather than accept that as a clean win, I traced why the gap was so large and
found the real issue: the collaborative-filtering embeddings had been trained on the entire
session log, including the very same "held-out" early-session titles this evaluation was
supposed to be testing against — the model had already learned that a given user watched both
their early and late titles together, before the evaluation ever held anything out.

Documented this explicitly rather than presenting 21.1% as a validated result — in the report
text and as a machine-readable flag. The content-only number, which has no such leakage, is the
more trustworthy of the two. A proper fix would mean retraining collaborative filtering excluding
each evaluated user's held-out interactions, which wasn't done given the project's scope.

## Tests

### The hypothesis-test function wasn't independently testable
The core comparison logic was embedded inside a function that read directly from the processed
CSV, with no way to feed it a synthetic case. Extracted it into its own pure function and
verified — by re-running against the real data — that its output matched the pre-refactor version
exactly, before writing any tests against it, so the refactor itself didn't silently change an
already-reported result.

### Import error when running a recommendation script directly
One script imports from its sibling modules in the same package. Running it directly from the
command line failed with a module-not-found error, since Python only puts the script's own
directory on the import path by default, not the project root. Fixed with an explicit path
adjustment at the top of the file. Also cleaned up a genuinely unused import that had been left
in along the way.

## Documentation

### The documented setup command was syntactically wrong
The README's first draft told readers to run `.venv/Scripts/activate` without the `source`
prefix needed to actually activate a virtual environment in bash. Fixed to match what had
actually been used and verified throughout the build — calling the venv's Python directly — with
both forms given for Windows and Linux/Mac.

## The pattern worth noting

Several of the hurdles above — the Wikipedia-saturation bug, the popularity-axis artifact, the
cross-genre similarity investigation, and the training-leakage discovery — weren't found because
anything crashed. The code ran fine and produced clean-looking numbers. They were found by
treating a suspiciously good result as a reason to dig further, not as a reason to stop. That's
also why the README's limitations section is as long as it is: several of those limitations are
the direct output of the investigations described here, not boilerplate caveats bolted on at the
end.
