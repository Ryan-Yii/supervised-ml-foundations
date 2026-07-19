"""训练、比较并评估 AI4I 设备故障预测模型。

模型选择、阈值选择和最终测试评估严格分离：
训练集用于五折交叉验证，验证集用于阈值分析，测试集只在最后评价一次。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import ensure_output_dir
from error_analysis import save_error_analysis as save_day14_error_analysis
from model_comparison import (
    build_model_pipeline as build_day14_model_pipeline,
    compare_models_cv as compare_day14_models_cv,
    model_configurations as day14_model_configurations,
)
from tune_models import tune_promising_models


RANDOM_STATE = 42
TARGET_COLUMN = "Machine failure"
TYPE_FEATURE = "Type"
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FEATURE_COLUMNS = (TYPE_FEATURE, *NUMERIC_FEATURES)
UCI_COLUMN_ALIASES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}
# UCI 的实际标识列为 UDI；UID 也作为同义禁用名称写入检查与文档。
LEAKAGE_COLUMNS = (
    "UID",
    "UDI",
    "Product ID",
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)
MODEL_FILENAME = "final_model.joblib"
THRESHOLD_FILENAME = "decision_threshold.json"


@dataclass(frozen=True)
class ModelConfiguration:
    """一个待比较的模型配置。"""

    key: str
    display_name: str
    model_name: str
    class_weight: str | None


@dataclass(frozen=True)
class DatasetSplits:
    """互不重叠的训练、验证、测试数据。"""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


MODEL_CONFIGURATIONS = (
    ModelConfiguration("dummy", "DummyClassifier", "dummy", None),
    ModelConfiguration("logistic", "LogisticRegression", "logistic", None),
    ModelConfiguration("logistic_balanced", "LogisticRegression (balanced)", "logistic", "balanced"),
    ModelConfiguration("decision_tree", "DecisionTreeClassifier", "decision_tree", None),
    ModelConfiguration("decision_tree_balanced", "DecisionTreeClassifier (balanced)", "decision_tree", "balanced"),
    ModelConfiguration("random_forest", "RandomForestClassifier", "random_forest", None),
    ModelConfiguration("random_forest_balanced", "RandomForestClassifier (balanced)", "random_forest", "balanced"),
)


def prepare_feature_target(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """选择允许特征和目标，并在构建 ``X`` 前执行泄漏检查。"""
    rename_map = {
        raw_name: canonical_name
        for raw_name, canonical_name in UCI_COLUMN_ALIASES.items()
        if raw_name in features.columns and canonical_name not in features.columns
    }
    normalized_features = features.rename(columns=rename_map)
    missing_features = set(FEATURE_COLUMNS) - set(normalized_features.columns)
    if missing_features:
        raise ValueError(f"数据缺少预期特征列：{sorted(missing_features)}")
    if TARGET_COLUMN not in targets.columns:
        raise ValueError(f"数据缺少目标列：{TARGET_COLUMN}")
    forbidden_in_features = set(FEATURE_COLUMNS).intersection(LEAKAGE_COLUMNS)
    if forbidden_in_features:
        raise RuntimeError(f"特征配置包含泄漏列：{sorted(forbidden_in_features)}")

    x_data = normalized_features.loc[:, list(FEATURE_COLUMNS)].copy()
    y_data = targets[TARGET_COLUMN].astype(int).rename(TARGET_COLUMN)
    if y_data.nunique() != 2 or set(y_data.unique()) - {0, 1}:
        raise ValueError("Machine failure 必须是取值为 0 和 1 的二分类标签。")
    return x_data, y_data


def fetch_ai4i_data() -> tuple[pd.DataFrame, pd.Series]:
    """通过官方 ``ucimlrepo`` 下载 AI4I 数据，但不写入原始数据副本。"""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=601)
    return prepare_feature_target(dataset.data.features.copy(), dataset.data.targets.copy())


def split_dataset(x_data: pd.DataFrame, y_data: pd.Series) -> DatasetSplits:
    """固定随机种子，以 60%/20%/20% 进行分层划分。"""
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x_data,
        y_data,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y_data,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return DatasetSplits(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_model_pipeline(model_name: str, class_weight: str | None) -> Pipeline:
    """保留 Week 1 兼容入口，并复用 Day 14 的唯一预处理工厂。"""
    return build_day14_model_pipeline(model_name, class_weight)


def metric_values(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """在给定阈值下计算故障正类的六项真实评价指标。"""
    if not 0.0 < threshold < 1.0:
        raise ValueError("分类阈值必须位于 0 与 1 之间。")
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def positive_probabilities(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    """按模型的真实类别顺序取出 ``Machine failure=1`` 概率。"""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("模型类别中不存在故障标签 1。")
    return model.predict_proba(x_data)[:, classes.index(1)]


def compare_models_cv(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """仅在训练集上完成五折分层交叉验证模型对比。"""
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    records: list[dict[str, Any]] = []
    for configuration in MODEL_CONFIGURATIONS:
        pipeline = build_model_pipeline(configuration.model_name, configuration.class_weight)
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        record: dict[str, Any] = {
            "model_key": configuration.key,
            "model": configuration.display_name,
            "class_weight": configuration.class_weight or "none",
        }
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            record[f"cv_{metric_name}_mean"] = float(np.mean(values))
            record[f"cv_{metric_name}_std"] = float(np.std(values))
        records.append(record)
    return pd.DataFrame(records).sort_values(
        by=["cv_f1_mean", "cv_recall_mean", "cv_pr_auc_mean"], ascending=False
    ).reset_index(drop=True)


def select_model_configuration(comparison: pd.DataFrame) -> ModelConfiguration:
    """按训练集 CV 的 F1、Recall、PR-AUC 顺序选择非 Dummy 最终模型。"""
    candidates = comparison.loc[comparison["model_key"] != "dummy"].sort_values(
        by=["cv_f1_mean", "cv_recall_mean", "cv_pr_auc_mean"], ascending=False
    )
    if candidates.empty:
        raise RuntimeError("没有可用于最终模型的非 Dummy 候选模型。")
    selected_key = str(candidates.iloc[0]["model_key"])
    for configuration in MODEL_CONFIGURATIONS:
        if configuration.key == selected_key:
            return configuration
    raise RuntimeError(f"无法找到模型配置：{selected_key}")


def select_validation_threshold(y_validation: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """只使用验证集分析阈值，并以 F1、Recall、Precision 顺序选择阈值。"""
    _, _, pr_thresholds = precision_recall_curve(y_validation, probabilities)
    candidate_thresholds = np.unique(np.concatenate((np.linspace(0.01, 0.99, 99), pr_thresholds)))
    records: list[dict[str, float]] = []
    for threshold in candidate_thresholds:
        if 0.0 < threshold < 1.0:
            values = metric_values(y_validation, probabilities, float(threshold))
            records.append({"threshold": float(threshold), **values})
    threshold_table = pd.DataFrame(records)
    if threshold_table.empty:
        raise RuntimeError("验证集阈值候选为空，无法选择分类阈值。")
    ordered = threshold_table.sort_values(
        by=["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    )
    return float(ordered.iloc[0]["threshold"]), threshold_table.sort_values("threshold").reset_index(drop=True)


def save_model_artifacts(model: Pipeline, threshold: float, output_dir: Path) -> tuple[Path, Path]:
    """保存最终 Pipeline 和由验证集选出的分类阈值。"""
    if not 0.0 < threshold < 1.0:
        raise ValueError("无法保存位于 0 与 1 之外的分类阈值。")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    threshold_path = output_dir / THRESHOLD_FILENAME
    joblib.dump(model, model_path)
    threshold_path.write_text(
        json.dumps({"threshold": threshold, "selected_on": "validation_set"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_path, threshold_path


def save_model_comparison(comparison: pd.DataFrame, output_dir: Path) -> Path:
    """保存训练集五折交叉验证对比表。"""
    path = output_dir / "model_comparison.csv"
    comparison.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_confusion_matrix(y_true: pd.Series, probabilities: np.ndarray, threshold: float, output_dir: Path) -> Path:
    """保存最终测试集的阈值化混淆矩阵。"""
    path = output_dir / "confusion_matrix.png"
    predicted = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predicted,
        labels=[0, 1],
        display_labels=["Normal", "Failure"],
        cmap="Reds",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("AI4I final test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_precision_recall_curve(y_true: pd.Series, probabilities: np.ndarray, output_dir: Path) -> Path:
    """保存最终测试集 Precision-Recall 曲线与真实 PR-AUC。"""
    path = output_dir / "precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AI4I final test precision-recall curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_threshold_analysis(threshold_table: pd.DataFrame, selected_threshold: float, output_dir: Path) -> tuple[Path, Path]:
    """保存仅由验证集产生的阈值表和阈值指标曲线。"""
    csv_path = output_dir / "validation_threshold_analysis.csv"
    image_path = output_dir / "threshold_analysis.png"
    threshold_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric_name in ("precision", "recall", "f1"):
        ax.plot(threshold_table["threshold"], threshold_table[metric_name], label=metric_name.capitalize())
    ax.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected = {selected_threshold:.3f}")
    ax.set_xlabel("Classification threshold (validation set only)")
    ax.set_ylabel("Score")
    ax.set_title("Validation threshold analysis")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return csv_path, image_path


def feature_importance_frame(model: Pipeline) -> pd.DataFrame:
    """提取最终逻辑回归系数或树模型重要性，供图表与审查使用。"""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        importance = np.abs(np.asarray(classifier.coef_).reshape(-1))
    else:
        raise ValueError("最终模型不支持特征重要性；DummyClassifier 不能作为最终模型。")
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def save_feature_importance(model: Pipeline, output_dir: Path) -> tuple[Path, Path]:
    """保存真实的最终模型特征重要性 CSV 与图表。"""
    importance = feature_importance_frame(model)
    csv_path = output_dir / "feature_importance.csv"
    image_path = output_dir / "feature_importance.png"
    importance.to_csv(csv_path, index=False, encoding="utf-8-sig")
    display_data = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(display_data["feature"], display_data["importance"])
    ax.set_xlabel("Absolute coefficient / feature importance")
    ax.set_title("Final model feature importance")
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return csv_path, image_path


def save_error_analysis(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """保存误报/漏报样本，并由这些真实样本生成特征摘要。"""
    predicted = (probabilities >= threshold).astype(int)
    results = x_test.reset_index(drop=True).copy()
    results["actual_machine_failure"] = y_test.reset_index(drop=True).astype(int)
    results["predicted_machine_failure"] = predicted
    results["failure_probability"] = probabilities
    false_positives = results.loc[
        (results["actual_machine_failure"] == 0) & (results["predicted_machine_failure"] == 1)
    ].sort_values("failure_probability", ascending=False)
    false_negatives = results.loc[
        (results["actual_machine_failure"] == 1) & (results["predicted_machine_failure"] == 0)
    ].sort_values("failure_probability")
    false_positive_path = output_dir / "false_positives.csv"
    false_negative_path = output_dir / "false_negatives.csv"
    report_path = output_dir / "error_analysis.md"
    false_positives.to_csv(false_positive_path, index=False, encoding="utf-8-sig")
    false_negatives.to_csv(false_negative_path, index=False, encoding="utf-8-sig")

    def summary(name: str, rows: pd.DataFrame) -> list[str]:
        lines = [f"## {name}", f"样本数：{len(rows)}"]
        if rows.empty:
            return [*lines, "没有该类错误样本，因此没有可归纳的典型特征。"]
        numeric_means = rows.loc[:, list(NUMERIC_FEATURES)].mean(numeric_only=True)
        type_counts = rows[TYPE_FEATURE].value_counts().to_dict()
        lines.append("平均数值特征：")
        lines.extend(f"- {column}: {value:.4f}" for column, value in numeric_means.items())
        lines.append(f"设备类型分布：{type_counts}")
        return lines

    report_lines = [
        "# 最终测试集错误分析",
        "下列统计完全基于最终测试集一次预测；它们描述误报（False Positive）和漏报（False Negative）的实际特征，不构成因果结论。",
        "",
        *summary("误报（False Positives）", false_positives),
        "",
        *summary("漏报（False Negatives）", false_negatives),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return false_positive_path, false_negative_path, report_path


def main_week01_legacy() -> None:
    """运行完整且无测试集泄漏的 Day 6 训练流程。"""
    x_data, y_data = fetch_ai4i_data()
    splits = split_dataset(x_data, y_data)
    output_dir = ensure_output_dir("iot_failure_prediction")

    comparison = compare_models_cv(splits.x_train, splits.y_train)
    selected_configuration = select_model_configuration(comparison)
    selected_model = build_model_pipeline(
        selected_configuration.model_name, selected_configuration.class_weight
    )
    selected_model.fit(splits.x_train, splits.y_train)
    validation_probabilities = positive_probabilities(selected_model, splits.x_validation)
    validation_default_metrics = metric_values(splits.y_validation, validation_probabilities, threshold=0.50)
    threshold, threshold_table = select_validation_threshold(splits.y_validation, validation_probabilities)
    validation_selected_metrics = metric_values(splits.y_validation, validation_probabilities, threshold)

    # 模型和阈值配置已确定；现在合并训练/验证集重训，测试集此前从未参与任何选择。
    x_train_final = pd.concat([splits.x_train, splits.x_validation], ignore_index=True)
    y_train_final = pd.concat([splits.y_train, splits.y_validation], ignore_index=True)
    final_model = build_model_pipeline(selected_configuration.model_name, selected_configuration.class_weight)
    final_model.fit(x_train_final, y_train_final)
    test_probabilities = positive_probabilities(final_model, splits.x_test)
    test_metrics = metric_values(splits.y_test, test_probabilities, threshold)
    test_predictions = (test_probabilities >= threshold).astype(int)

    comparison_path = save_model_comparison(comparison, output_dir)
    model_path, threshold_path = save_model_artifacts(final_model, threshold, output_dir)
    matrix_path = save_confusion_matrix(splits.y_test, test_probabilities, threshold, output_dir)
    pr_curve_path = save_precision_recall_curve(splits.y_test, test_probabilities, output_dir)
    threshold_csv_path, threshold_image_path = save_threshold_analysis(threshold_table, threshold, output_dir)
    importance_csv_path, importance_image_path = save_feature_importance(final_model, output_dir)
    fp_path, fn_path, error_report_path = save_error_analysis(
        splits.x_test, splits.y_test, test_probabilities, threshold, output_dir
    )
    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset (id=601)",
        "random_state": RANDOM_STATE,
        "target": TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "excluded_leakage_columns": list(LEAKAGE_COLUMNS),
        "splits": {
            "train_rows": int(len(splits.x_train)),
            "validation_rows": int(len(splits.x_validation)),
            "test_rows": int(len(splits.x_test)),
            "strategy": "stratified 60/20/20",
        },
        "cross_validation": {"strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", "models": comparison.to_dict(orient="records")},
        "selection": {
            "model_key": selected_configuration.key,
            "model": selected_configuration.display_name,
            "class_weight": selected_configuration.class_weight,
            "selection_priority": ["cv_f1_mean", "cv_recall_mean", "cv_pr_auc_mean"],
            "threshold": threshold,
            "threshold_selected_on": "validation_set_only",
        },
        "validation": {"at_0_50": validation_default_metrics, "at_selected_threshold": validation_selected_metrics},
        "final_test_once": test_metrics,
        "artifacts": {
            "model": model_path.as_posix(),
            "threshold": threshold_path.as_posix(),
            "model_comparison": comparison_path.as_posix(),
            "confusion_matrix": matrix_path.as_posix(),
            "precision_recall_curve": pr_curve_path.as_posix(),
            "threshold_analysis": threshold_image_path.as_posix(),
            "feature_importance": importance_image_path.as_posix(),
            "false_positives": fp_path.as_posix(),
            "false_negatives": fn_path.as_posix(),
            "error_analysis": error_report_path.as_posix(),
            "feature_importance_csv": importance_csv_path.as_posix(),
            "threshold_analysis_csv": threshold_csv_path.as_posix(),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Day 6：AI4I 设备故障预测 ===")
    print(f"训练/验证/测试行数：{len(splits.x_train)}/{len(splits.x_validation)}/{len(splits.x_test)}")
    print(f"最终模型：{selected_configuration.display_name}")
    print(f"验证集选择阈值：{threshold:.4f}")
    print("最终测试集指标（仅本次评价）：")
    for name, value in test_metrics.items():
        print(f"{name.upper()}：{value:.4f}")
    print("\n最终测试集 classification_report：")
    print(classification_report(splits.y_test, test_predictions, digits=4, zero_division=0))
    print(f"真实产物目录：{output_dir}")
    print(f"指标文件：{metrics_path}")


def main() -> None:
    """Run Day 14 model selection without allowing test data into any selection step."""
    x_data, y_data = fetch_ai4i_data()
    splits = split_dataset(x_data, y_data)
    output_dir = ensure_output_dir("iot_failure_prediction")
    baseline = compare_day14_models_cv(splits.x_train, splits.y_train)
    baseline.to_csv(output_dir / "baseline_model_comparison.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    tuned, tuned_estimators = tune_promising_models(splits.x_train, splits.y_train)
    tuned.to_csv(output_dir / "tuned_model_comparison.csv", index=False, encoding="utf-8-sig")
    selected_key = str(tuned.iloc[0]["model_key"])
    selected_model = tuned_estimators[selected_key]["estimator"]
    validation_probabilities = positive_probabilities(selected_model, splits.x_validation)
    validation_default_metrics = metric_values(splits.y_validation, validation_probabilities, threshold=0.50)
    threshold, threshold_table = select_validation_threshold(splits.y_validation, validation_probabilities)
    validation_selected_metrics = metric_values(splits.y_validation, validation_probabilities, threshold)
    selected_configuration = next(item for item in day14_model_configurations() if item.key == selected_key)
    final_model = build_day14_model_pipeline(selected_configuration.model_name, selected_configuration.class_weight)
    final_model.set_params(**tuned_estimators[selected_key]["params"])
    final_model.fit(pd.concat([splits.x_train, splits.x_validation], ignore_index=True), pd.concat([splits.y_train, splits.y_validation], ignore_index=True))
    test_probabilities = positive_probabilities(final_model, splits.x_test)
    test_metrics = metric_values(splits.y_test, test_probabilities, threshold)
    model_path, threshold_path = save_model_artifacts(final_model, threshold, output_dir)
    matrix_path = save_confusion_matrix(splits.y_test, test_probabilities, threshold, output_dir)
    pr_curve_path = save_precision_recall_curve(splits.y_test, test_probabilities, output_dir)
    threshold_csv_path, threshold_image_path = save_threshold_analysis(threshold_table, threshold, output_dir)
    importance_csv_path, importance_image_path = save_feature_importance(final_model, output_dir)
    fp_path, fn_path, error_report_path = save_day14_error_analysis(splits.x_test, splits.y_test, test_probabilities, threshold, output_dir)
    fig, ax = plt.subplots(figsize=(8, 5)); baseline_sorted = baseline.sort_values("cv_f1_mean"); ax.barh(baseline_sorted["model"], baseline_sorted["cv_f1_mean"]); ax.set(xlabel="Mean F1", title="AI4I baseline comparison (training CV only)"); fig.tight_layout(); comparison_plot_path = output_dir / "model_comparison.png"; fig.savefig(comparison_plot_path, dpi=150); plt.close(fig)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(splits.y_test, test_probabilities)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.4f}"); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="AI4I final test ROC curve"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); roc_path = output_dir / "roc_curve.png"; fig.savefig(roc_path, dpi=150); plt.close(fig)
    metrics_payload = {"dataset": "UCI AI4I 2020 Predictive Maintenance Dataset (id=601)", "random_state": RANDOM_STATE, "target": TARGET_COLUMN, "feature_columns": list(FEATURE_COLUMNS), "excluded_leakage_columns": list(LEAKAGE_COLUMNS), "splits": {"train_rows": int(len(splits.x_train)), "validation_rows": int(len(splits.x_validation)), "test_rows": int(len(splits.x_test)), "strategy": "stratified 60/20/20"}, "baseline_cross_validation": baseline.to_dict(orient="records"), "tuned_cross_validation": tuned.to_dict(orient="records"), "selection": {"model_key": selected_key, "model": selected_configuration.display_name, "threshold": threshold, "threshold_selected_on": "validation_set_only"}, "validation": {"at_0_50": validation_default_metrics, "at_selected_threshold": validation_selected_metrics}, "final_test_once": test_metrics, "artifacts": {"model": str(model_path), "threshold": str(threshold_path), "baseline": "baseline_model_comparison.csv", "tuned": "tuned_model_comparison.csv", "comparison_plot": str(comparison_plot_path), "confusion_matrix": str(matrix_path), "roc_curve": str(roc_path), "precision_recall_curve": str(pr_curve_path), "threshold_analysis": str(threshold_image_path), "feature_importance": str(importance_image_path), "false_positives": str(fp_path), "false_negatives": str(fn_path), "error_analysis": str(error_report_path), "feature_importance_csv": str(importance_csv_path), "threshold_analysis_csv": str(threshold_csv_path)}}
    (output_dir / "final_test_metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Day 14 · Week 2 Day 7 · AI4I model selection ===")
    print(f"Selected tuned model: {selected_configuration.display_name}; validation threshold: {threshold:.4f}")
    for name, value in test_metrics.items(): print(f"{name.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
