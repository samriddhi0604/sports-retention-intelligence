# Processed data schema

Both tables are written by data/validate.py from data/features.py's raw output.
Steps 2-5 should read only these two files, not the `_raw` intermediates.

## data/processed/user_features.csv

One row per simulated subscriber who had at least one observed (pre-churn) session.

| column | dtype | meaning |
|---|---|---|
| user_id | int | unique subscriber id |
| churned | int (0/1) | 1 if the user churned during the analysis window |
| sports_spike_engagement | float [0,1] | fraction of the user's observed sessions on a real IPL 2024 match day |
| genre_diversity | float [0,1] | normalized Shannon entropy of genre tags across the user's watched titles |
| avg_completion_rate | float [0,1] | mean IMDb rating/10 of the user's watched titles (completion-rate proxy) |
| n_sessions | int | number of observed sessions the above are computed from |

Row count: 4979. Dtypes:
```
user_id                      int64
churned                      int64
sports_spike_engagement    float64
genre_diversity            float64
avg_completion_rate        float64
n_sessions                   int64
dtype: object
```

## data/processed/sessions.csv

One row per simulated viewing session (the raw interaction log Step 5 needs for
gateway analysis and collaborative filtering). Same OBSERVED (pre-churn-truncated)
session set that user_features.csv is aggregated from -- not the full-window
hypothetical history used internally for churn sampling.

| column | dtype | meaning |
|---|---|---|
| user_id | int | matches user_features.csv user_id |
| title_id | str | real IMDb tconst -- canonical item id, use this (not title strings) for joins |
| timestamp | date | session date (day granularity -- Cricsheet has no match kickoff times) |
| session_type | str | 'sports_window' if timestamp is a real IPL 2024 match day, else 'movie' |

Row count: 280585. Dtypes:
```
user_id                  int64
title_id                   str
timestamp       datetime64[us]
session_type               str
dtype: object
```