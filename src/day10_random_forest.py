"""Day 10 · Week 2 Day 3 · Random Forest analysis on breast cancer data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

from common import ensure_output_dir


RANDOM_STATE = 42


def split_breast_cancer_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the same 70/15/15 stratified split policy used in Day 09."""
    data = load_breast_cancer()
    x_train, x_remaining, y_train, y_remaining = train_test_split(data.data, data.target, test_size=0.30, stratify=data.target, random_state=RANDOM_STATE)
    x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=RANDOM_STATE)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_random_forest(n_estimators: int, max_depth: int | None, min_samples_leaf: int, max_features: str | float | None) -> RandomForestClassifier:
    """Build an unscaled bagging ensemble with fixed randomness."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1, random_state=RANDOM_STATE)


def main() -> None:
    """Analyze four forest controls, select by validation accuracy, and test once."""
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_breast_cancer_data()
    configurations = [
        {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": feature}
        for n in (50, 150, 300) for depth in (None, 4, 8) for leaf in (1, 3) for feature in ("sqrt", 0.5)
    ]
    results: list[dict[str, object]] = []
    for config in configurations:
        model = build_random_forest(**config).fit(x_train, y_train)
        results.append({**config, "train_accuracy": float(model.score(x_train, y_train)), "validation_accuracy": float(model.score(x_validation, y_validation))})
    best = max(results, key=lambda row: (float(row["validation_accuracy"]), -int(row["n_estimators"])))
    final_model = build_random_forest(best["n_estimators"], best["max_depth"], best["min_samples_leaf"], best["max_features"]).fit(np.vstack([x_train, x_validation]), np.concatenate([y_train, y_validation]))
    predicted = final_model.predict(x_test); output_dir = ensure_output_dir("random_forest")
    test_accuracy = float(accuracy_score(y_test, predicted))
    (output_dir / "random_forest_metrics.json").write_text(json.dumps({"validation_selection": best, "final_test_accuracy": test_accuracy, "analysis_configurations": results}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5)); ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=load_breast_cancer().target_names, cmap="Greens", colorbar=False, ax=ax); ax.set_title("Random forest final test confusion matrix"); fig.tight_layout(); fig.savefig(output_dir / "random_forest_confusion_matrix.png", dpi=150); plt.close(fig)
    order = np.argsort(final_model.feature_importances_)[-12:]; names = load_breast_cancer().feature_names
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(names[order], final_model.feature_importances_[order]); ax.set_title("Random forest feature importance (prediction association)"); fig.tight_layout(); fig.savefig(output_dir / "random_forest_feature_importance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth in (None, 4, 8):
        rows = [row for row in results if row["max_depth"] == depth and row["min_samples_leaf"] == 1 and row["max_features"] == "sqrt"]
        ax.plot([row["n_estimators"] for row in rows], [row["validation_accuracy"] for row in rows], marker="o", label=f"max_depth={depth}")
    ax.set(xlabel="n_estimators", ylabel="Validation accuracy", title="Forest estimator analysis", ylim=(0.0, 1.05)); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(output_dir / "estimator_analysis.png", dpi=150); plt.close(fig)
    print("=== Day 10 · Week 2 Day 3 · Random Forest ==="); print(f"Best validation configuration: {best}"); print(f"Final test accuracy (evaluated once): {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
