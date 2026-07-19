"""Moderate training-only tuning for two promising Day 14 AI4I candidates."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from model_comparison import RANDOM_STATE, build_model_pipeline

TUNED_MODEL_KEYS = ("decision_tree", "random_forest_balanced")


def tune_promising_models(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Tune the two strongest baseline candidates using only the training partition."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {
        "decision_tree": (build_model_pipeline("decision_tree", None), {"classifier__max_depth": [3, 5, 8, None], "classifier__min_samples_leaf": [1, 2, 5]}),
        "random_forest_balanced": (build_model_pipeline("random_forest", "balanced"), {"classifier__n_estimators": [150, 300], "classifier__max_depth": [6, 10], "classifier__min_samples_leaf": [1, 2]}),
    }
    records: list[dict[str, Any]] = []; best: dict[str, dict[str, Any]] = {}
    for key, (pipeline, parameters) in searches.items():
        search = GridSearchCV(pipeline, parameters, scoring="f1", cv=folds, n_jobs=1, refit=True, return_train_score=False)
        search.fit(x_train, y_train)
        records.append({"model_key": key, "model": key.replace("_", " "), "cv_f1_mean": float(search.best_score_), "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]), "best_params": json.dumps(search.best_params_, sort_keys=True)})
        best[key] = {"estimator": search.best_estimator_, "params": search.best_params_, "cv_f1_mean": float(search.best_score_)}
    return pd.DataFrame(records).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True), best
