"""Day 09 · Week 2 Day 2 · Decision Tree selection on breast cancer data."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import ensure_output_dir


RANDOM_STATE = 42


@dataclass(frozen=True)
class CancerSplits:
    """A reproducible 70/15/15 stratified binary-classification split."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def split_breast_cancer_data() -> CancerSplits:
    """Split the built-in breast-cancer data without leaking test samples."""
    dataset = load_breast_cancer()
    x_train, x_remaining, y_train, y_remaining = train_test_split(dataset.data, dataset.target, test_size=0.30, stratify=dataset.target, random_state=RANDOM_STATE)
    x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=RANDOM_STATE)
    return CancerSplits(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_decision_tree(max_depth: int) -> DecisionTreeClassifier:
    """Build an unscaled tree; split thresholds are invariant to feature scale."""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)


def depth_records(splits: CancerSplits) -> list[dict[str, float | int]]:
    """Measure depth 1..20 on training and validation sets."""
    records: list[dict[str, float | int]] = []
    for depth in range(1, 21):
        model = build_decision_tree(depth).fit(splits.x_train, splits.y_train)
        records.append({"max_depth": depth, "train_accuracy": float(model.score(splits.x_train, splits.y_train)), "validation_accuracy": float(model.score(splits.x_validation, splits.y_validation))})
    return records


def _save_plots(records: list[dict[str, float | int]], model: DecisionTreeClassifier, y_test: np.ndarray, predicted: np.ndarray) -> None:
    """Save required selection, structure, importance, and final-test visualizations."""
    dataset = load_breast_cancer()
    output_dir = ensure_output_dir("decision_tree")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row["max_depth"] for row in records], [row["train_accuracy"] for row in records], marker="o", label="Training")
    ax.plot([row["max_depth"] for row in records], [row["validation_accuracy"] for row in records], marker="o", label="Validation")
    ax.set(xlabel="max_depth", ylabel="Accuracy", title="Decision-tree depth selection", ylim=(0.0, 1.05))
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(output_dir / "depth_selection.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=dataset.target_names, cmap="Blues", colorbar=False, ax=ax); ax.set_title("Decision tree final test confusion matrix"); fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 8)); plot_tree(model, feature_names=dataset.feature_names, class_names=dataset.target_names, filled=True, max_depth=3, fontsize=6, ax=ax); ax.set_title("Decision tree (first four levels)"); fig.tight_layout(); fig.savefig(output_dir / "tree_structure.png", dpi=150); plt.close(fig)
    importance = np.argsort(model.feature_importances_)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(dataset.feature_names[importance], model.feature_importances_[importance]); ax.set_title("Decision tree feature importance (prediction association)"); ax.set_xlabel("Importance"); fig.tight_layout(); fig.savefig(output_dir / "feature_importance.png", dpi=150); plt.close(fig)


def main() -> None:
    """Select depth from validation accuracy and evaluate test data once."""
    splits = split_breast_cancer_data()
    records = depth_records(splits)
    best = max(records, key=lambda row: (float(row["validation_accuracy"]), -int(row["max_depth"])))
    final_model = build_decision_tree(int(best["max_depth"])).fit(np.vstack([splits.x_train, splits.x_validation]), np.concatenate([splits.y_train, splits.y_validation]))
    predicted = final_model.predict(splits.x_test)
    output_dir = ensure_output_dir("decision_tree")
    with (output_dir / "decision_tree_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["max_depth", "train_accuracy", "validation_accuracy"]); writer.writeheader(); writer.writerows(records)
    _save_plots(records, final_model, splits.y_test, predicted)
    print("=== Day 09 · Week 2 Day 2 · Decision Tree ===")
    print(f"Best validation max_depth: {best['max_depth']}")
    print(f"Final test accuracy (evaluated once): {accuracy_score(splits.y_test, predicted):.4f}")


if __name__ == "__main__":
    main()
