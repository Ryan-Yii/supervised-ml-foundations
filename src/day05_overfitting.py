"""Day 5：通过决策树深度曲线观察欠拟合和过拟合。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from common import configure_chinese_plot_font, output_file

configure_chinese_plot_font()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def split_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将乳腺癌数据分为 70% 训练、15% 验证和 15% 测试集。"""
    dataset = load_breast_cancer()
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def evaluate_depths(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[list[int], list[float], list[float]]:
    """在训练/验证集上测量 1 至 20 层决策树的准确率。"""
    depths = list(range(1, 21))
    train_scores: list[float] = []
    validation_scores: list[float] = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        train_scores.append(float(accuracy_score(y_train, model.predict(x_train))))
        validation_scores.append(float(accuracy_score(y_validation, model.predict(x_validation))))
    return depths, train_scores, validation_scores


def save_depth_curve(depths: list[int], train_scores: list[float], validation_scores: list[float]) -> None:
    """保存训练集与验证集准确率随深度变化的曲线。"""
    figure_path = output_file("overfitting", "depth_vs_accuracy.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_scores, marker="o", label="训练集 Accuracy")
    ax.plot(depths, validation_scores, marker="o", label="验证集 Accuracy")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("决策树深度与泛化表现")
    ax.set_xticks(depths)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"深度曲线已保存：{figure_path}")


def main() -> None:
    """根据验证集选择深度，最后只在测试集做一次评价。"""
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data()
    depths, train_scores, validation_scores = evaluate_depths(
        x_train, y_train, x_validation, y_validation
    )
    best_index = max(range(len(depths)), key=lambda index: (validation_scores[index], -depths[index]))
    best_depth = depths[best_index]
    save_depth_curve(depths, train_scores, validation_scores)

    x_train_final = np.vstack((x_train, x_validation))
    y_train_final = np.concatenate((y_train, y_validation))
    final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
    final_model.fit(x_train_final, y_train_final)
    test_accuracy = accuracy_score(y_test, final_model.predict(x_test))

    print("=== Day 5：过拟合实验 ===")
    print(f"训练集形状：{x_train.shape}；验证集形状：{x_validation.shape}；测试集形状：{x_test.shape}")
    print(f"验证集选择的最佳 max_depth：{best_depth}")
    print(f"对应验证集 Accuracy：{validation_scores[best_index]:.4f}")
    print(f"最终测试集 Accuracy（仅评价一次）：{test_accuracy:.4f}")
    print("欠拟合（Underfitting）是模型过于简单；过拟合（Overfitting）是模型记住训练细节却泛化较差。")


if __name__ == "__main__":
    main()
