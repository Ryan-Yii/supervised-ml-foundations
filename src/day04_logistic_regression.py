"""Day 4：用逻辑回归完成 Iris 多分类。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from common import configure_chinese_plot_font, output_file

configure_chinese_plot_font()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_classifier() -> Pipeline:
    """创建标准化与逻辑回归组成的 Pipeline。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
        ]
    )


def save_confusion_matrix(y_true: np.ndarray, y_predicted: np.ndarray, labels: list[str]) -> None:
    """保存分类混淆矩阵图片。"""
    figure_path = output_file("classification", "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predicted,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Iris 逻辑回归混淆矩阵")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"混淆矩阵已保存：{figure_path}")


def main() -> None:
    """训练 Iris 分类器并输出真实分类报告和预测概率。"""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    pipeline = build_classifier()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test[:5])
    accuracy = accuracy_score(y_test, predictions)

    print("=== Day 4：逻辑回归 ===")
    print(f"Accuracy：{accuracy:.4f}")
    print("\nclassification_report：")
    print(classification_report(y_test, predictions, target_names=iris.target_names, zero_division=0))
    print("前 5 个测试样本的 predict_proba（类别顺序：setosa、versicolor、virginica）：")
    print(np.round(probabilities, 4))
    print("指标说明：Accuracy 是总体正确率；Precision 衡量预测为某类时的可信度；")
    print("Recall 衡量该类样本被找回的比例；F1 是 Precision 与 Recall 的调和平均。")
    save_confusion_matrix(y_test, predictions, iris.target_names.tolist())


if __name__ == "__main__":
    main()
