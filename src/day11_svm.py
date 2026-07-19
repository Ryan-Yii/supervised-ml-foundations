"""Day 11 · Week 2 Day 4 · Support Vector Machines on breast cancer data."""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ensure_output_dir


RANDOM_STATE = 42


@dataclass(frozen=True)
class CancerSplits:
    """Independent stratified data partitions for selection and final testing."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def split_breast_cancer_data() -> CancerSplits:
    """Return a fixed 70/15/15 breast-cancer split."""
    dataset = load_breast_cancer()
    x_train, x_remaining, y_train, y_remaining = train_test_split(dataset.data, dataset.target, test_size=0.30, stratify=dataset.target, random_state=RANDOM_STATE)
    x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=RANDOM_STATE)
    return CancerSplits(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_svm_pipeline(kernel: str, c_value: float, gamma: str | float) -> Pipeline:
    """Build a scaled SVC; decision scores support ROC-AUC without probabilities."""
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel=kernel, C=c_value, gamma=gamma, random_state=RANDOM_STATE))])


def main() -> None:
    """Compare linear/RBF settings by validation accuracy and test the winner once."""
    splits = split_breast_cancer_data()
    settings = [("linear", c_value, "scale") for c_value in (0.1, 1.0, 10.0)] + [("rbf", c_value, gamma) for c_value in (0.1, 1.0, 10.0) for gamma in ("scale", 0.01, 0.1)]
    records: list[dict[str, object]] = []
    for kernel, c_value, gamma in settings:
        model = build_svm_pipeline(kernel, c_value, gamma).fit(splits.x_train, splits.y_train)
        records.append({"kernel": kernel, "C": c_value, "gamma": gamma, "train_accuracy": float(model.score(splits.x_train, splits.y_train)), "validation_accuracy": float(model.score(splits.x_validation, splits.y_validation))})
    best = max(records, key=lambda row: (float(row["validation_accuracy"]), -float(row["C"])))
    final_model = build_svm_pipeline(str(best["kernel"]), float(best["C"]), best["gamma"]).fit(np.vstack([splits.x_train, splits.x_validation]), np.concatenate([splits.y_train, splits.y_validation]))
    predicted = final_model.predict(splits.x_test); scores = final_model.decision_function(splits.x_test)
    output_dir = ensure_output_dir("svm")
    metrics = {"validation_selection": best, "final_test_accuracy": float(accuracy_score(splits.y_test, predicted)), "final_test_roc_auc": float(roc_auc_score(splits.y_test, scores)), "parameter_analysis": records}
    (output_dir / "svm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5)); ConfusionMatrixDisplay.from_predictions(splits.y_test, predicted, display_labels=load_breast_cancer().target_names, cmap="Purples", colorbar=False, ax=ax); ax.set_title("SVM final test confusion matrix"); fig.tight_layout(); fig.savefig(output_dir / "svm_confusion_matrix.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(records))
    colors = ["tab:blue" if row["kernel"] == "linear" else "tab:orange" for row in records]
    ax.scatter(positions, [row["validation_accuracy"] for row in records], c=colors)
    ax.plot(positions, [row["validation_accuracy"] for row in records], color="gray", alpha=0.35)
    ax.set_xticks(positions, [f"{row['kernel']}\nC={row['C']}, γ={row['gamma']}" for row in records], rotation=45, ha="right")
    ax.scatter([], [], color="tab:blue", label="linear")
    ax.scatter([], [], color="tab:orange", label="rbf")
    ax.set_xlabel("Kernel / C / gamma configuration")
    ax.set(ylabel="Validation accuracy", title="SVM parameter analysis", ylim=(0.0, 1.05)); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "svm_parameter_analysis.png", dpi=150); plt.close(fig)
    print("=== Day 11 · Week 2 Day 4 · SVM ==="); print(f"Best validation setting: {best}"); print(f"Final test accuracy: {metrics['final_test_accuracy']:.4f}, ROC-AUC: {metrics['final_test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
