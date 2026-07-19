"""Tests for Week 2 selection workflows and the Day 14 IoT upgrade."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
IOT_DIR = PROJECT_ROOT / "projects" / "iot_failure_prediction"
for directory in (SRC_DIR, IOT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def test_grid_search_uses_pipeline_parameter_names_and_training_data_only() -> None:
    """SVM grid search exposes only ``svc__`` parameters for training folds."""
    from day12_cross_validation import build_grid_search, split_breast_cancer_data

    splits = split_breast_cancer_data()
    search = build_grid_search()
    assert isinstance(search, GridSearchCV)
    assert {"svc__C", "svc__kernel", "svc__gamma"}.issubset(search.param_grid)
    search.fit(splits.x_train, splits.y_train)
    assert hasattr(search, "best_params_")
    assert len(splits.x_test) > 0


def test_fair_comparison_reports_mean_and_std_for_all_six_models() -> None:
    """All candidates share CV folds and report every requested score."""
    from day13_model_comparison import compare_models, split_breast_cancer_data

    splits = split_breast_cancer_data()
    comparison = compare_models(splits.x_train, splits.y_train)
    assert set(comparison["model"]) == {
        "DummyClassifier",
        "LogisticRegression",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "SVC",
    }
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert f"cv_{metric}_mean" in comparison
        assert f"cv_{metric}_std" in comparison


def test_iot_upgrade_keeps_leakage_columns_out_and_adds_svm_and_knn() -> None:
    """Day 14 candidates include KNN/SVM without expanding the feature boundary."""
    from model_comparison import build_model_pipeline, model_configurations
    from train import FEATURE_COLUMNS, LEAKAGE_COLUMNS

    names = {configuration.model_name for configuration in model_configurations()}
    assert {"dummy", "logistic", "knn", "decision_tree", "random_forest", "svm"}.issubset(names)
    assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS)
    assert list(build_model_pipeline("svm", None).named_steps) == ["preprocessor", "classifier"]


def test_iot_tuning_is_limited_to_the_two_promising_cv_candidates() -> None:
    """Day 14 tunes the leading tree and balanced-forest candidates, not every model."""
    from tune_models import TUNED_MODEL_KEYS

    assert TUNED_MODEL_KEYS == ("decision_tree", "random_forest_balanced")
