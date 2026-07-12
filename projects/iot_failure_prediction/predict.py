"""命令行加载最终模型，并对一条 AI4I 设备记录进行阈值化预测。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train import FEATURE_COLUMNS, MODEL_FILENAME, THRESHOLD_FILENAME


DEFAULT_OUTPUT_DIR = Path("outputs") / "iot_failure_prediction"
VALID_TYPES = {"L", "M", "H"}


def example_device_state() -> dict[str, float | str]:
    """返回一条合法的示例设备记录，供 ``--example`` 使用。"""
    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1_551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 120.0,
    }


def validate_device_state(device_state: dict[str, Any]) -> dict[str, float | str]:
    """验证字段、机器类型和数值范围，并返回标准化后的记录。"""
    provided_columns = set(device_state)
    required_columns = set(FEATURE_COLUMNS)
    if provided_columns != required_columns:
        missing = sorted(required_columns - provided_columns)
        unexpected = sorted(provided_columns - required_columns)
        raise ValueError(f"输入字段不匹配；缺少字段={missing}，多余字段={unexpected}。")

    machine_type = str(device_state["Type"]).upper()
    if machine_type not in VALID_TYPES:
        raise ValueError(f"Type 必须是 {sorted(VALID_TYPES)} 之一，收到 {device_state['Type']!r}。")
    normalized: dict[str, float | str] = {"Type": machine_type}
    for column in FEATURE_COLUMNS[1:]:
        try:
            value = float(device_state[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} 必须是有限数值，收到 {device_state[column]!r}。") from error
        if not math.isfinite(value):
            raise ValueError(f"{column} 必须是有限数值，收到 {device_state[column]!r}。")
        if column in {"Air temperature [K]", "Process temperature [K]"} and not 200.0 <= value <= 400.0:
            raise ValueError(f"{column} 应位于 200 至 400 K，收到 {value}。")
        if column in {"Rotational speed [rpm]", "Torque [Nm]"} and value <= 0.0:
            raise ValueError(f"{column} 必须大于 0，收到 {value}。")
        if column == "Tool wear [min]" and value < 0.0:
            raise ValueError(f"{column} 不能小于 0，收到 {value}。")
        normalized[column] = value
    return normalized


def load_model_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Any, float]:
    """加载已保存的最终模型与验证集阈值。"""
    model_path = output_dir / MODEL_FILENAME
    threshold_path = output_dir / THRESHOLD_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到最终模型：{model_path}。请先运行 train.py。")
    if not threshold_path.is_file():
        raise FileNotFoundError(f"未找到阈值文件：{threshold_path}。请先运行 train.py。")
    model = joblib.load(model_path)
    try:
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(threshold_payload["threshold"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"阈值文件格式无效：{threshold_path}。") from error
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"阈值必须位于 0 与 1 之间，读取到 {threshold}。")
    return model, threshold


def predict_device_state(
    model: Any, threshold: float, device_state: dict[str, Any]
) -> tuple[int, float]:
    """输出依据保存阈值的类别和故障概率，而不是模型默认 0.5 标签。"""
    if not 0.0 < threshold < 1.0:
        raise ValueError("分类阈值必须位于 0 与 1 之间。")
    validated_state = validate_device_state(device_state)
    frame = pd.DataFrame([validated_state], columns=list(FEATURE_COLUMNS))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("模型类别中不存在 Machine failure=1。")
    failure_index = classes.index(1)
    failure_probability = float(model.predict_proba(frame)[0][failure_index])
    prediction = int(failure_probability >= threshold)
    return prediction, failure_probability


def build_parser() -> argparse.ArgumentParser:
    """创建接受一条设备记录的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="使用已保存 AI4I 模型预测单条设备状态。")
    parser.add_argument("--example", action="store_true", help="使用内置示例记录。")
    parser.add_argument("--type", dest="machine_type", help="机器类型：L、M 或 H。")
    parser.add_argument("--air-temperature", type=float, help="Air temperature，单位 K。")
    parser.add_argument("--process-temperature", type=float, help="Process temperature，单位 K。")
    parser.add_argument("--rotational-speed", type=float, help="Rotational speed，单位 rpm。")
    parser.add_argument("--torque", type=float, help="Torque，单位 Nm。")
    parser.add_argument("--tool-wear", type=float, help="Tool wear，单位 min。")
    return parser


def state_from_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    """从命令行参数组装一条完整记录，缺失项由验证函数清楚指出。"""
    if arguments.example:
        return example_device_state()
    return {
        "Type": arguments.machine_type,
        "Air temperature [K]": arguments.air_temperature,
        "Process temperature [K]": arguments.process_temperature,
        "Rotational speed [rpm]": arguments.rotational_speed,
        "Torque [Nm]": arguments.torque,
        "Tool wear [min]": arguments.tool_wear,
    }


def main() -> None:
    """加载最终产物并输出类别、故障概率和实际使用的分类阈值。"""
    arguments = build_parser().parse_args()
    device_state = state_from_arguments(arguments)
    model, threshold = load_model_artifacts()
    prediction, probability = predict_device_state(model, threshold, device_state)
    label = "故障" if prediction == 1 else "正常"
    print("=== AI4I 单设备预测 ===")
    print(f"输入状态：{validate_device_state(device_state)}")
    print(f"故障概率：{probability:.4f}")
    print(f"分类阈值（验证集选择）：{threshold:.4f}")
    print(f"预测类别：{label}（Machine failure={prediction}）")


if __name__ == "__main__":
    main()
