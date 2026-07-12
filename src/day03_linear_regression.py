"""Day 3：使用 Diabetes 数据集练习线性回归。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from common import configure_chinese_plot_font, output_file

configure_chinese_plot_font()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def train_linear_regression() -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """训练线性回归并返回真实值、预测值和回归指标。"""
    diabetes = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.20, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return metrics, y_test, predictions


def save_prediction_plot(actual: np.ndarray, predicted: np.ndarray) -> None:
    """保存真实值与预测值的散点图。"""
    figure_path = output_file("regression", "actual_vs_predicted.png")
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.75, edgecolor="none")
    ax.plot([lower, upper], [lower, upper], "r--", label="理想预测线")
    ax.set_title("线性回归：真实值与预测值")
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"散点图已保存：{figure_path}")


def main() -> None:
    """运行回归实验并输出真实指标。"""
    metrics, actual, predicted = train_linear_regression()
    save_prediction_plot(actual, predicted)
    print("=== Day 3：线性回归 ===")
    print(f"MAE：{metrics['mae']:.4f}")
    print(f"MSE：{metrics['mse']:.4f}")
    print(f"RMSE：{metrics['rmse']:.4f}")
    print(f"R²：{metrics['r2']:.4f}")
    print("R² 不是分类准确率；当模型比用均值预测还差时，R² 可以为负数。")


if __name__ == "__main__":
    main()
