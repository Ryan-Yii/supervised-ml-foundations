"""Save Day 14 false-positive and false-negative records from final test predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_comparison import NUMERIC_FEATURES, TYPE_FEATURE


def save_error_analysis(x_test: pd.DataFrame, y_test: pd.Series, probabilities: np.ndarray, threshold: float, output_dir: Path) -> tuple[Path, Path, Path]:
    """Persist final-test errors and descriptive, non-causal summaries of their features."""
    predicted = (probabilities >= threshold).astype(int); results = x_test.reset_index(drop=True).copy()
    results["actual_machine_failure"] = y_test.reset_index(drop=True).astype(int); results["predicted_machine_failure"] = predicted; results["failure_probability"] = probabilities
    false_positives = results.loc[(results["actual_machine_failure"] == 0) & (results["predicted_machine_failure"] == 1)].sort_values("failure_probability", ascending=False)
    false_negatives = results.loc[(results["actual_machine_failure"] == 1) & (results["predicted_machine_failure"] == 0)].sort_values("failure_probability")
    fp_path = output_dir / "false_positives.csv"; fn_path = output_dir / "false_negatives.csv"; report_path = output_dir / "error_analysis.md"
    false_positives.to_csv(fp_path, index=False, encoding="utf-8-sig"); false_negatives.to_csv(fn_path, index=False, encoding="utf-8-sig")
    def section(title: str, rows: pd.DataFrame) -> list[str]:
        lines = [f"## {title}", f"Samples: {len(rows)}"]
        if rows.empty: return [*lines, "No samples in this error category."]
        lines.extend(f"- {column}: {value:.4f}" for column, value in rows.loc[:, list(NUMERIC_FEATURES)].mean().items()); lines.append(f"- Type distribution: {rows[TYPE_FEATURE].value_counts().to_dict()}")
        return lines
    report_path.write_text("\n".join(["# Final-test error analysis", "These descriptive summaries are prediction associations, not causal conclusions.", "", *section("False positives", false_positives), "", *section("False negatives", false_negatives)]) + "\n", encoding="utf-8")
    return fp_path, fn_path, report_path
