"""Day 13 · Week 2 Day 6 · Fair six-model comparison on breast cancer data."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import ensure_output_dir


RANDOM_STATE = 42


@dataclass(frozen=True)
class CancerSplits:
    """Training data for CV and an isolated final test data partition."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def split_breast_cancer_data() -> CancerSplits:
    """Keep test data separate before fair model comparison begins."""
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, stratify=dataset.target, random_state=RANDOM_STATE)
    return CancerSplits(x_train, x_test, y_train, y_test)


def candidate_models() -> dict[str, object]:
    """Return six candidates, with scaling embedded only where it is required."""
    return {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "KNeighborsClassifier": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7))]),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE),
        "SVC": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE))]),
    }


def compare_models(x_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Evaluate every candidate on the same five stratified training folds."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0), "recall": make_scorer(recall_score, zero_division=0), "f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc"}
    records: list[dict[str, float | str]] = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, x_train, y_train, cv=folds, scoring=scoring, n_jobs=1, error_score="raise")
        record: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            record[f"cv_{metric}_mean"] = float(np.mean(values))
            record[f"cv_{metric}_std"] = float(np.std(values))
        records.append(record)
    return pd.DataFrame(records).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def _save_plots(comparison: pd.DataFrame) -> None:
    """Save an ROC-AUC ranking chart and an F1 stability chart from the CV results."""
    output_dir = ensure_output_dir("model_comparison")
    ordered = comparison.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(ordered["model"], ordered["cv_roc_auc_mean"]); ax.set(xlabel="Mean ROC-AUC", title="Fair model comparison (training CV only)", xlim=(0.0, 1.05)); fig.tight_layout(); fig.savefig(output_dir / "model_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.errorbar(comparison["model"], comparison["cv_f1_mean"], yerr=comparison["cv_f1_std"], fmt="o", capsize=4); ax.tick_params(axis="x", rotation=35); ax.set(ylabel="F1 mean ± std", title="Model stability across identical CV folds", ylim=(0.0, 1.05)); ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(output_dir / "model_stability.png", dpi=150); plt.close(fig)


def main() -> None:
    """Run CV comparison on training data and report the sorting metric explicitly."""
    splits = split_breast_cancer_data(); comparison = compare_models(splits.x_train, splits.y_train)
    output_dir = ensure_output_dir("model_comparison"); comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig"); _save_plots(comparison)
    print("=== Day 13 · Week 2 Day 6 · Fair model comparison ===")
    print("Ranking metric: mean ROC-AUC from identical 5-fold StratifiedKFold training folds.")
    print(comparison[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean", "cv_f1_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
