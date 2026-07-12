"""面向第一周学习脚本的最小冒烟测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
IOT_PROJECT_DIR = PROJECT_ROOT / "projects" / "iot_failure_prediction"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(IOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(IOT_PROJECT_DIR))


class SmokeTests(unittest.TestCase):
    """验证公共路径工具和数据集加载函数的基础行为。"""

    def test_output_directory_can_be_created(self) -> None:
        """公共函数可创建相对输出目录。"""
        from common import ensure_output_dir

        directory = ensure_output_dir("smoke_test")
        try:
            self.assertTrue(directory.is_dir())
            self.assertEqual(directory, Path("outputs") / "smoke_test")
        finally:
            directory.rmdir()

    def test_chinese_plot_font_configuration_returns_an_available_font(self) -> None:
        """共享绘图配置选择一个本机可用的中文字体。"""
        from common import configure_chinese_plot_font

        font_name = configure_chinese_plot_font()

        self.assertIsInstance(font_name, str)
        self.assertTrue(font_name)

    def test_iris_shapes_match_the_learning_note(self) -> None:
        """Iris 特征矩阵和标签向量保持教材中的标准形状。"""
        from day01_iris_exploration import load_iris_dataframe

        features, labels, _ = load_iris_dataframe()
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))

    def test_iot_feature_columns_exclude_leakage_columns(self) -> None:
        """物联网训练脚本不把标识符或故障模式标签作为特征。"""
        from train import FEATURE_COLUMNS, LEAKAGE_COLUMNS

        self.assertEqual(len(FEATURE_COLUMNS), 6)
        self.assertNotIn("Machine failure", FEATURE_COLUMNS)
        self.assertTrue(set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
