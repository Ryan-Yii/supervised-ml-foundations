"""Week 2 learning-script smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import ANY, patch

from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_week02_modules_import_and_use_expected_pipelines() -> None:
    """Distance-based models are packaged with their scaler in a Pipeline."""
    import day08_knn
    import day11_svm

    assert isinstance(day08_knn.build_knn_pipeline(3, "uniform"), Pipeline)
    assert list(day08_knn.build_knn_pipeline(3, "uniform").named_steps) == ["scaler", "knn"]
    assert isinstance(day11_svm.build_svm_pipeline("rbf", 1.0, "scale"), Pipeline)
    assert list(day11_svm.build_svm_pipeline("rbf", 1.0, "scale").named_steps) == ["scaler", "svc"]


def test_day08_split_is_stratified_and_non_overlapping() -> None:
    """The 70/15/15 Iris split is reproducible and has no shared samples."""
    from day08_knn import split_iris_data

    splits = split_iris_data()
    assert len(splits.x_train) == 105
    assert len(splits.x_validation) in {22, 23}
    assert len(splits.x_test) in {22, 23}
    train_rows = {tuple(row) for row in splits.x_train}
    validation_rows = {tuple(row) for row in splits.x_validation}
    test_rows = {tuple(row) for row in splits.x_test}
    assert train_rows.isdisjoint(validation_rows)
    assert train_rows.isdisjoint(test_rows)
    assert validation_rows.isdisjoint(test_rows)


def test_tree_models_do_not_add_unnecessary_scalers() -> None:
    """Tree and forest factories return estimators directly, not scaled Pipelines."""
    from day09_decision_tree import build_decision_tree
    from day10_random_forest import build_random_forest

    assert not isinstance(build_decision_tree(3), Pipeline)
    assert not isinstance(build_random_forest(100, 5, 1, "sqrt"), Pipeline)


def test_svm_parameter_analysis_can_generate_its_plot() -> None:
    """The SVM parameter chart has an explicit configuration x-axis label."""
    from matplotlib.axes import Axes
    from day11_svm import main

    original_set_xlabel = Axes.set_xlabel
    with patch.object(Axes, "set_xlabel", autospec=True, side_effect=original_set_xlabel) as set_xlabel:
        main()

    set_xlabel.assert_any_call(ANY, "Kernel / C / gamma configuration")
    assert (PROJECT_ROOT / "outputs" / "svm" / "svm_parameter_analysis.png").is_file()
