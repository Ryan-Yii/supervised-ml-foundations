"""Day 12 · Week 2 Day 5 · Stratified cross-validation and SVM GridSearchCV."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ensure_output_dir


RANDOM_STATE = 42


@dataclass(frozen=True)
class CancerSplits:
    """A training partition and an untouched final test partition."""

    x_train: object
    x_test: object
    y_train: object
    y_test: object


def split_breast_cancer_data() -> CancerSplits:
    """Hold out 20% before any cross-validation or hyperparameter selection."""
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, stratify=dataset.target, random_state=RANDOM_STATE)
    return CancerSplits(x_train, x_test, y_train, y_test)


def build_grid_search() -> GridSearchCV:
    """Create a moderate, pipeline-safe five-fold SVM parameter search."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC(random_state=RANDOM_STATE))])
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"svc__C": [0.1, 1.0, 10.0], "svc__kernel": ["linear", "rbf"], "svc__gamma": ["scale", 0.01]},
        scoring="roc_auc",
        cv=cross_validator,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )


def main() -> None:
    """Search training folds only and write the actual CV summary and candidate table."""
    splits = split_breast_cancer_data()
    search = build_grid_search()
    search.fit(splits.x_train, splits.y_train)
    results = pd.DataFrame(search.cv_results_)
    result_columns = ["params", "mean_test_score", "std_test_score", *[f"split{fold}_test_score" for fold in range(5)], "rank_test_score"]
    output_dir = ensure_output_dir("cross_validation")
    results.loc[:, result_columns].sort_values("rank_test_score").to_csv(output_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    best_index = int(search.best_index_)
    summary = {
        "random_state": RANDOM_STATE,
        "test_set_policy": "held out before GridSearchCV and not passed to fit",
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "scoring": "roc_auc",
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_fold_scores": [float(results.loc[best_index, f"split{fold}_test_score"]) for fold in range(5)],
        "best_fold_mean": float(results.loc[best_index, "mean_test_score"]),
        "best_fold_std": float(results.loc[best_index, "std_test_score"]),
        "final_test_roc_auc": float(roc_auc_score(splits.y_test, search.best_estimator_.decision_function(splits.x_test))),
    }
    (output_dir / "cross_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== Day 12 · Week 2 Day 5 · Cross-validation and GridSearchCV ===")
    print(f"Best parameters: {search.best_params_}")
    print(f"CV ROC-AUC: {search.best_score_:.4f} ± {summary['best_fold_std']:.4f}")
    print(f"Final test ROC-AUC (not used for selection): {summary['final_test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
