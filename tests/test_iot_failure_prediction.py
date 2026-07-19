"""Day 6 故障预测项目的离线自动化测试，不下载 UCI 数据。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IOT_PROJECT_DIR = PROJECT_ROOT / "projects" / "iot_failure_prediction"
SRC_DIR = PROJECT_ROOT / "src"
for directory in (IOT_PROJECT_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


class IoTFailurePredictionTests(unittest.TestCase):
    """验证数据边界、管道、持久化与单样本预测。"""

    @staticmethod
    def make_feature_frame(rows: int = 24) -> pd.DataFrame:
        """创建不含真实数据的最小合法设备特征集。"""
        index = np.arange(rows)
        return pd.DataFrame(
            {
                "UDI": 10_000 + index,
                "Product ID": [f"L{number:05d}" for number in index],
                "Type": np.resize(np.array(["L", "M", "H"]), rows),
                "Air temperature [K]": 298.0 + index * 0.01,
                "Process temperature [K]": 308.0 + index * 0.01,
                "Rotational speed [rpm]": 1_500.0 + index,
                "Torque [Nm]": 40.0 + index * 0.1,
                "Tool wear [min]": 100.0 + index,
            }
        )

    def test_feature_target_preparation_excludes_identifiers_and_failure_modes(self) -> None:
        """训练数据仅保留六个允许特征，故障模式不进入 X。"""
        from train import FEATURE_COLUMNS, prepare_feature_target

        features = self.make_feature_frame()
        targets = pd.DataFrame(
            {
                "Machine failure": np.resize(np.array([0, 1]), len(features)),
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )

        x_data, y_data = prepare_feature_target(features, targets)

        self.assertEqual(tuple(x_data.columns), FEATURE_COLUMNS)
        self.assertEqual(y_data.name, "Machine failure")
        self.assertFalse({"UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"}.intersection(x_data.columns))

    def test_feature_target_preparation_accepts_ucimlrepo_unitless_columns(self) -> None:
        """官方接口省略单位后缀时仍规范化为项目约定的六项特征。"""
        from train import FEATURE_COLUMNS, prepare_feature_target

        features = self.make_feature_frame().drop(columns=["UDI", "Product ID"]).rename(
            columns={
                "Air temperature [K]": "Air temperature",
                "Process temperature [K]": "Process temperature",
                "Rotational speed [rpm]": "Rotational speed",
                "Torque [Nm]": "Torque",
                "Tool wear [min]": "Tool wear",
            }
        )
        targets = pd.DataFrame({"Machine failure": np.resize(np.array([0, 1]), len(features))})

        x_data, _ = prepare_feature_target(features, targets)

        self.assertEqual(tuple(x_data.columns), FEATURE_COLUMNS)

    def test_fetch_ai4i_data_requests_official_dataset_and_applies_feature_boundary(self) -> None:
        """数据加载入口请求 UCI id=601，且返回无泄漏的 X 与 y。"""
        from train import FEATURE_COLUMNS, fetch_ai4i_data

        features = self.make_feature_frame().drop(columns=["UDI", "Product ID"])
        targets = pd.DataFrame(
            {
                "Machine failure": np.resize(np.array([0, 1]), len(features)),
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )
        dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))

        with patch("ucimlrepo.fetch_ucirepo", return_value=dataset) as fetcher:
            x_data, y_data = fetch_ai4i_data()

        fetcher.assert_called_once_with(id=601)
        self.assertEqual(tuple(x_data.columns), FEATURE_COLUMNS)
        self.assertEqual(y_data.name, "Machine failure")

    def test_pipeline_contains_column_transformer_and_can_fit(self) -> None:
        """逻辑回归 Pipeline 统一封装数值与类别预处理。"""
        from train import build_model_pipeline

        x_data = self.make_feature_frame().drop(columns=["UDI", "Product ID"])
        y_data = pd.Series(np.resize(np.array([0, 1]), len(x_data)))
        pipeline = build_model_pipeline("logistic", class_weight="balanced")
        pipeline.fit(x_data, y_data)

        self.assertIsInstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
        self.assertEqual(pipeline.predict_proba(x_data[:1]).shape, (1, 2))

    def test_legacy_pipeline_uses_model_specific_numeric_preprocessing(self) -> None:
        """The compatibility factory scales only linear and distance-based models."""
        from train import build_model_pipeline

        def numeric_transformer(model_name: str) -> object:
            pipeline = build_model_pipeline(model_name, class_weight=None)
            preprocessor = pipeline.named_steps["preprocessor"]
            return next(transformer for name, transformer, _ in preprocessor.transformers if name == "numeric")

        for model_name in ("logistic", "knn", "svm"):
            self.assertIsInstance(numeric_transformer(model_name), StandardScaler)
        for model_name in ("decision_tree", "random_forest"):
            self.assertEqual(numeric_transformer(model_name), "passthrough")

    def test_saved_model_and_threshold_can_be_loaded_for_prediction(self) -> None:
        """保存后的模型与阈值可被重新加载并用于单样本预测。"""
        from predict import load_model_artifacts, predict_device_state
        from train import build_model_pipeline, save_model_artifacts

        x_data = self.make_feature_frame().drop(columns=["UDI", "Product ID"])
        y_data = pd.Series(np.resize(np.array([0, 1]), len(x_data)))
        model = build_model_pipeline("dummy", class_weight=None)
        model.fit(x_data, y_data)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            save_model_artifacts(model, threshold=0.40, output_dir=output_dir)
            loaded_model, threshold = load_model_artifacts(output_dir)
            label, probability = predict_device_state(
                loaded_model,
                threshold,
                x_data.iloc[0].to_dict(),
            )

        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(threshold, 0.40)

    def test_day14_decision_tree_saved_inference_contract(self) -> None:
        """A saved Day 14-style tree uses FEATURE_COLUMNS and threshold 0.19 after reload."""
        from predict import load_model_artifacts, predict_device_state, validate_device_state
        from train import FEATURE_COLUMNS, build_model_pipeline, save_model_artifacts

        x_data = self.make_feature_frame(rows=30).drop(columns=["UDI", "Product ID"])
        y_data = pd.Series(np.array([0] * 15 + [1] * 15))
        model = build_model_pipeline("decision_tree", class_weight=None)
        model.set_params(classifier__max_depth=8, classifier__min_samples_leaf=1)
        model.fit(x_data, y_data)
        device_state = x_data.iloc[-1].to_dict()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            save_model_artifacts(model, threshold=0.19, output_dir=output_dir)
            loaded_model, threshold = load_model_artifacts(output_dir)
            prediction, probability = predict_device_state(loaded_model, threshold, device_state)

        validated_state = validate_device_state(device_state)
        expected_frame = pd.DataFrame([validated_state], columns=list(FEATURE_COLUMNS))
        failure_index = list(loaded_model.classes_).index(1)
        expected_probability = float(loaded_model.predict_proba(expected_frame)[0][failure_index])
        self.assertEqual(tuple(expected_frame.columns), FEATURE_COLUMNS)
        self.assertEqual(threshold, 0.19)
        self.assertEqual(probability, expected_probability)
        self.assertEqual(prediction, int(expected_probability >= 0.19))

    def test_invalid_device_state_reports_clear_error(self) -> None:
        """缺失字段和无效数值会在预测前被拒绝。"""
        from predict import validate_device_state

        invalid_state = {
            "Type": "L",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": -1.0,
            "Torque [Nm]": 40.0,
        }

        with self.assertRaisesRegex(ValueError, "字段|数值"):
            validate_device_state(invalid_state)


if __name__ == "__main__":
    unittest.main()
