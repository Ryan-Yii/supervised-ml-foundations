"""Day 08 · Week 2 Day 1 · K-Nearest Neighbors on Iris."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import ensure_output_dir


RANDOM_STATE = 42


@dataclass(frozen=True)
class IrisSplits:
    """Non-overlapping stratified Iris training, validation, and test sets."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def split_iris_data() -> IrisSplits:
    """Return a reproducible 70/15/15 stratified Iris split."""
    dataset = load_iris()
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        dataset.data, dataset.target, test_size=0.30, stratify=dataset.target, random_state=RANDOM_STATE
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=RANDOM_STATE
    )
    return IrisSplits(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_knn_pipeline(neighbors: int, weights: str) -> Pipeline:
    """Build a scaled KNN classifier; distance comparisons require common units."""
    return Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=neighbors, weights=weights))]
    )


def evaluate_candidates(splits: IrisSplits) -> list[dict[str, float | int | str]]:
    """Fit K=1..30 on training data and score only the validation data."""
    records: list[dict[str, float | int | str]] = []
    for weights in ("uniform", "distance"):
        for neighbors in range(1, 31):
            model = build_knn_pipeline(neighbors, weights)
            model.fit(splits.x_train, splits.y_train)
            records.append(
                {
                    "k": neighbors,
                    "weights": weights,
                    "train_accuracy": float(model.score(splits.x_train, splits.y_train)),
                    "validation_accuracy": float(model.score(splits.x_validation, splits.y_validation)),
                }
            )
    return records


def select_best_candidate(records: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    """Choose by validation accuracy, then prefer the smaller K for a stable tie-break."""
    return max(records, key=lambda item: (float(item["validation_accuracy"]), -int(item["k"])))


def save_selection_plot(records: list[dict[str, float | int | str]]) -> None:
    """Save training and validation accuracy curves for both weight strategies."""
    output_dir = ensure_output_dir("knn")
    fig, ax = plt.subplots(figsize=(9, 5))
    for weights in ("uniform", "distance"):
        subset = [row for row in records if row["weights"] == weights]
        ax.plot([row["k"] for row in subset], [row["train_accuracy"] for row in subset], "--", label=f"{weights} train")
        ax.plot([row["k"] for row in subset], [row["validation_accuracy"] for row in subset], marker="o", label=f"{weights} validation")
    ax.set(xlabel="K neighbors", ylabel="Accuracy", title="KNN K selection on validation data", ylim=(0.0, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "knn_k_selection.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_predicted: np.ndarray) -> None:
    """Save the final test confusion matrix after validation-only selection."""
    output_dir = ensure_output_dir("knn")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, display_labels=load_iris().target_names, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("KNN final test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "knn_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run Day 08, save real artifacts, and use test data only for final evaluation."""
    splits = split_iris_data()
    records = evaluate_candidates(splits)
    best = select_best_candidate(records)
    output_dir = ensure_output_dir("knn")
    with (output_dir / "knn_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "weights", "train_accuracy", "validation_accuracy"])
        writer.writeheader()
        writer.writerows(records)
    save_selection_plot(records)
    x_final = np.vstack([splits.x_train, splits.x_validation])
    y_final = np.concatenate([splits.y_train, splits.y_validation])
    final_model = build_knn_pipeline(int(best["k"]), str(best["weights"]))
    final_model.fit(x_final, y_final)
    predicted = final_model.predict(splits.x_test)
    test_accuracy = accuracy_score(splits.y_test, predicted)
    save_confusion_matrix(splits.y_test, predicted)
    print("=== Day 08 · Week 2 Day 1 · KNN ===")
    print(f"Best validation candidate: K={best['k']}, weights={best['weights']}")
    print(f"Validation accuracy: {float(best['validation_accuracy']):.4f}")
    print(f"Final test accuracy (evaluated once): {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
