"""Day 1：只做 Iris 数据探索，不训练任何模型。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def load_iris_dataframe() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """加载 Iris 数据，并返回特征 DataFrame、标签向量和类别名称。"""
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = iris.target
    class_names = iris.target_names.tolist()
    return features, labels, class_names


def main() -> None:
    """打印 Iris 数据集的基础结构与质量检查。"""
    features, labels, class_names = load_iris_dataframe()

    print("=== Day 1：Iris 数据探索 ===")
    print(f"样本数：{features.shape[0]}")
    print(f"特征数：{features.shape[1]}")
    print(f"类别名称：{class_names}")
    print(f"X.shape={features.shape}：150 个样本，每个样本有 4 个特征。")
    print(f"y.shape={labels.shape}：150 个样本各有 1 个类别标签。")
    print("\n前五行特征：")
    print(features.head())
    print("\n各列缺失值数量：")
    print(features.isna().sum())
    print("\n类别分布：")
    print(pd.Series(labels, name="species").value_counts().sort_index())
    print("\n本脚本只探索数据，不训练模型。")


if __name__ == "__main__":
    main()
