"""
Step 4 -- Churn classification model.

Logistic Regression on the three engineered features (sports_spike_engagement,
genre_diversity, avg_completion_rate) -- a good fit given the underlying
relationship was built as roughly logistic (see data/features.py), and its
coefficients are directly interpretable, which Step 5's win-back targeting
logic needs.

Threshold selection: the default 0.5 cutoff is arbitrary under class
imbalance (~35% churn here). Instead, 5-fold cross-validated probability
predictions on the TRAINING set are used to build a precision-recall curve
and pick the threshold that maximizes F1 -- tuned without ever touching the
held-out test set, which is then scored once, at the end, with that fixed
threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"

FEATURE_COLS = ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]
TARGET_COL = "churned"
RNG_SEED = 42


def pick_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds (the last point
    # is precision=1, recall=0 with no corresponding threshold) -- drop it.
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def train_and_evaluate() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG_SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # threshold picked from cross-validated predictions on the training set only
    cv_proba = cross_val_predict(
        LogisticRegression(random_state=RNG_SEED),
        X_train_scaled,
        y_train,
        cv=5,
        method="predict_proba",
    )[:, 1]
    threshold = pick_threshold_by_f1(y_train, cv_proba)
    print(f"[train] F1-optimal threshold (from CV on train set): {threshold:.4f}")

    model = LogisticRegression(random_state=RNG_SEED).fit(X_train_scaled, y_train)

    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
    }

    # coefficients are already in standardized (scaled-feature) space since
    # the model was fit on StandardScaler output
    coefficients = {
        feature: float(coef) for feature, coef in zip(FEATURE_COLS, model.coef_[0])
    }
    metrics["standardized_coefficients"] = dict(
        sorted(coefficients.items(), key=lambda kv: -abs(kv[1]))
    )
    metrics["intercept"] = float(model.intercept_[0])

    print("=== Churn model evaluation (held-out test set) ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "churn_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "churn_scaler.joblib")
    joblib.dump(threshold, ARTIFACTS_DIR / "churn_threshold.joblib")
    print(f"\n[train] wrote model/scaler/threshold to {ARTIFACTS_DIR}")

    results_path = ARTIFACTS_DIR / "churn_model_metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {results_path}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
