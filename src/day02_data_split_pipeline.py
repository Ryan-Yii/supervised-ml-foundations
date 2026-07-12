"""Day 2：分层划分数据，并用 Pipeline 避免数据泄漏。"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def split_iris_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 70%/15%/15% 进行分层训练、验证和测试划分。"""
    iris = load_iris()
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        iris.data,
        iris.target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_pipeline() -> Pipeline:
    """返回包含标准化和逻辑回归的分类 Pipeline。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    """仅在训练集拟合 Pipeline，并报告验证集准确率。"""
    x_train, x_validation, x_test, y_train, y_validation, _ = split_iris_data()
    pipeline = build_pipeline()

    # 只对训练集调用 fit；若先对全量数据标准化，就会产生数据泄漏（Data Leakage）。
    pipeline.fit(x_train, y_train)
    validation_accuracy = pipeline.score(x_validation, y_validation)

    print("=== Day 2：数据划分和 Pipeline ===")
    print(f"训练集形状：X={x_train.shape}, y={y_train.shape}")
    print(f"验证集形状：X={x_validation.shape}, y={y_validation.shape}")
    print(f"测试集形状：X={x_test.shape}（本脚本不用于调参）")
    print(f"验证集 Accuracy：{validation_accuracy:.4f}")
    print("数据泄漏说明：StandardScaler 的均值和标准差只从训练集学习。")


if __name__ == "__main__":
    main()
