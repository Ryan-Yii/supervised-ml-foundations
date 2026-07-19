"""Day 14 candidate models and fair training-only cross-validation for AI4I."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TYPE_FEATURE = "Type"
NUMERIC_FEATURES = ("Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]")
FEATURE_COLUMNS = (TYPE_FEATURE, *NUMERIC_FEATURES)
LEAKAGE_COLUMNS = ("UID", "UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")


@dataclass(frozen=True)
class ModelConfiguration:
    """One baseline candidate with an optional class-balance strategy."""

    key: str
    display_name: str
    model_name: str
    class_weight: str | None


def model_configurations() -> tuple[ModelConfiguration, ...]:
    """Return baseline candidates; Dummy is retained as the required lower bound."""
    return (
        ModelConfiguration("dummy", "DummyClassifier", "dummy", None),
        ModelConfiguration("logistic", "LogisticRegression", "logistic", None),
        ModelConfiguration("logistic_balanced", "LogisticRegression (balanced)", "logistic", "balanced"),
        ModelConfiguration("knn", "KNeighborsClassifier", "knn", None),
        ModelConfiguration("decision_tree", "DecisionTreeClassifier", "decision_tree", None),
        ModelConfiguration("decision_tree_balanced", "DecisionTreeClassifier (balanced)", "decision_tree", "balanced"),
        ModelConfiguration("random_forest", "RandomForestClassifier", "random_forest", None),
        ModelConfiguration("random_forest_balanced", "RandomForestClassifier (balanced)", "random_forest", "balanced"),
        ModelConfiguration("svm", "SVC", "svm", None),
        ModelConfiguration("svm_balanced", "SVC (balanced)", "svm", "balanced"),
    )


def build_model_pipeline(model_name: str, class_weight: str | None) -> Pipeline:
    """Build a leakage-safe categorical/numeric pipeline for one candidate."""
    needs_scaling = model_name in {"logistic", "knn", "svm"}
    numeric_transformer: Any = StandardScaler() if needs_scaling else "passthrough"
    preprocessor = ColumnTransformer([("numeric", numeric_transformer, list(NUMERIC_FEATURES)), ("type", OneHotEncoder(handle_unknown="ignore"), [TYPE_FEATURE])], remainder="drop")
    if model_name == "dummy":
        classifier: Any = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        classifier = LogisticRegression(class_weight=class_weight, max_iter=1_000, random_state=RANDOM_STATE)
    elif model_name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=11, weights="distance")
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(class_weight=class_weight, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=200, class_weight=class_weight, max_depth=10, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
    elif model_name == "svm":
        classifier = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=class_weight, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def compare_models_cv(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Report mean and standard deviation on identical 5-fold training splits."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0), "recall": make_scorer(recall_score, zero_division=0), "f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    records: list[dict[str, Any]] = []
    for configuration in model_configurations():
        scores = cross_validate(build_model_pipeline(configuration.model_name, configuration.class_weight), x_train, y_train, cv=folds, scoring=scoring, n_jobs=1, error_score="raise")
        record: dict[str, Any] = {"model_key": configuration.key, "model": configuration.display_name, "class_weight": configuration.class_weight or "none"}
        for name in scoring:
            values = scores[f"test_{name}"]
            record[f"cv_{name}_mean"] = float(np.mean(values))
            record[f"cv_{name}_std"] = float(np.std(values))
        records.append(record)
    return pd.DataFrame(records).sort_values(["cv_f1_mean", "cv_recall_mean", "cv_pr_auc_mean"], ascending=False).reset_index(drop=True)
