"""
Step 6 -- Visualization.

Three plots, saved as PNGs under figures/, so the project is demoable in an
interview without running code live:
  - engagement_distribution.png: sports-spike engagement split by churned/retained
  - spike_timeline.png: the viewership time series with detected spikes overlaid
  - precision_recall_curve.png: the churn model's precision-recall curve

Each plot is built directly from the same processed data / saved artifacts
the earlier steps already produced -- nothing new is computed here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

RNG_SEED = 42  # must match training/train_churn_model.py's split for a faithful test-set PR curve


def plot_engagement_distribution() -> None:
    df = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for churned, label, color in [(0, "Retained", "#2a9d8f"), (1, "Churned", "#e76f51")]:
        subset = df.loc[df["churned"] == churned, "sports_spike_engagement"]
        ax.hist(subset, bins=30, alpha=0.6, label=f"{label} (n={len(subset)}, mean={subset.mean():.2f})",
                color=color, density=True)
    ax.set_xlabel("Sports-spike engagement (fraction of sessions on a real IPL 2024 match day)")
    ax.set_ylabel("Density")
    ax.set_title("Sports engagement distribution: churned vs. retained users")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "engagement_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_spike_timeline() -> None:
    scored = pd.read_csv(PROCESSED_DIR / "spike_detection_scored_days.csv", parse_dates=["date"])
    matches = pd.read_csv(PROCESSED_DIR / "clean_cricsheet_matches.csv", parse_dates=["date"])
    match_days = set(matches["date"].dt.normalize())
    scored["is_match_day"] = scored["date"].dt.normalize().isin(match_days)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(scored["date"], scored["views"], color="#264653", linewidth=1, label="Daily pageviews")
    ax.plot(scored["date"], scored["baseline"], color="#adb5bd", linewidth=1, linestyle="--",
            label="Day-of-week baseline (median)")

    match_only = scored[scored["is_match_day"] & ~scored["is_flagged"]]
    flagged_true = scored[scored["is_flagged"] & scored["is_match_day"]]
    flagged_false = scored[scored["is_flagged"] & ~scored["is_match_day"]]

    ax.scatter(match_only["date"], match_only["views"], color="#adb5bd", marker="o", s=25, zorder=3,
               label="Real match day (not flagged)")
    ax.scatter(flagged_true["date"], flagged_true["views"], color="#2a9d8f", marker="^", s=60, zorder=4,
               label="Flagged spike (real match day)")
    ax.scatter(flagged_false["date"], flagged_false["views"], color="#e76f51", marker="x", s=60, zorder=4,
               label="Flagged spike (false positive)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Wikipedia pageviews (\"2024 Indian Premier League\" article)")
    ax.set_title("Viewership proxy with detected spikes vs. real IPL 2024 match days")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = FIGURES_DIR / "spike_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_precision_recall_curve() -> None:
    df = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    feature_cols = ["sports_spike_engagement", "genre_diversity", "avg_completion_rate"]
    X = df[feature_cols].to_numpy()
    y = df["churned"].to_numpy()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG_SEED)

    model = joblib.load(ARTIFACTS_DIR / "churn_model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "churn_scaler.joblib")
    threshold = joblib.load(ARTIFACTS_DIR / "churn_threshold.joblib")

    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#264653", linewidth=2)
    ax.axhline(y_test.mean(), color="#adb5bd", linestyle="--", label=f"No-skill baseline ({y_test.mean():.2f})")

    chosen_pred = (proba >= threshold).astype(int)
    chosen_precision = ((chosen_pred == 1) & (y_test == 1)).sum() / max(chosen_pred.sum(), 1)
    chosen_recall = ((chosen_pred == 1) & (y_test == 1)).sum() / y_test.sum()
    ax.scatter([chosen_recall], [chosen_precision], color="#e76f51", zorder=5, s=80,
               label=f"Chosen threshold ({threshold:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Churn model precision-recall curve (held-out test set)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "precision_recall_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_engagement_distribution()
    plot_spike_timeline()
    plot_precision_recall_curve()


if __name__ == "__main__":
    main()
