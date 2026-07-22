# Supervised ML Foundations

## Overview

A reproducible two-week supervised machine-learning study project covering leakage-free evaluation, model selection, and an IoT predictive-maintenance exercise. This repository documents structured learning and engineering practice rather than a novel research contribution.

The exercises use Python and small, complete experiments to study data handling, models, evaluation, model selection, and data leakage. The AI4I case study uses a public synthetic predictive-maintenance dataset, not real factory data.

## Two-Week Learning Sequence

| Day | Topic | Implemented exercise or status |
| --- | --- | --- |
| Day 01 | Iris exploration | Features, labels, missingness, and class distribution |
| Day 02 | Data split and Pipeline | 70/15/15 stratified split and leakage-aware scaling |
| Day 03 | Linear regression | MAE, MSE, RMSE, R2, and a scatter plot |
| Day 04 | Logistic regression | Classification report, probabilities, and confusion matrix |
| Day 05 | Overfitting | Decision-tree depth selection and validation curves |
| Day 06 | IoT failure prediction | AI4I data, feature encoding, model persistence, and prediction |
| Day 07 | Review and rest day | No separate implementation; review of Week 1 exercises |
| Day 08 | K-nearest neighbors | Pipeline, validation-set K selection, and test evaluation |
| Day 09 | Decision tree | Depth analysis, feature importance, and test evaluation |
| Day 10 | Random forest | Parameter analysis, feature importance, and test evaluation |
| Day 11 | Support vector machine | Scaled SVM and parameter analysis |
| Day 12 | Cross-validation and GridSearchCV | Training-only StratifiedKFold and SVM search |
| Day 13 | Fair model comparison | Same-fold Dummy, linear, distance, tree, forest, and SVM comparison |
| Day 14 | AI4I consolidation | Leakage-free selection, validation threshold, error analysis, and saved inference |

## Reproducibility and Boundaries

- Pipelines and ColumnTransformer bind preprocessing to models to reduce leakage risk.
- Day 12 and Day 13 restrict cross-validation and GridSearchCV to the training partition.
- Day 14 selects its classification threshold on validation data; the test set is reserved for final evaluation.
- AI4I 2020 is synthetic data. Results and examples cannot be generalized to real production equipment.
- Metrics, JSON files, CSV files, and code are kept as learning artefacts; the repository does not claim novel research results.

## Project Structure

~~~text
src/                                  Day 01-05 and Day 08-13 exercises
projects/iot_failure_prediction/      Day 14 AI4I workflow and documentation
outputs/                              Committed reproducibility artefacts
tests/                                Smoke, model-selection, and AI4I tests
docs/                                 Learning notes and checklists
~~~

## Setup and Validation

Python 3.10 or newer is required.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
~~~

On Windows PowerShell, activate with .venv\\Scripts\\Activate.ps1. The Day 14 training entry point downloads the public UCI dataset on its first full run; it does not store the downloaded raw dataset in this repository.

~~~bash
python src/day01_iris_exploration.py
python src/day02_data_split_pipeline.py
python src/day03_linear_regression.py
python src/day04_logistic_regression.py
python src/day05_overfitting.py
python src/day08_knn.py
python src/day09_decision_tree.py
python src/day10_random_forest.py
python src/day11_svm.py
python src/day12_cross_validation.py
python src/day13_model_comparison.py
python projects/iot_failure_prediction/train.py
python projects/iot_failure_prediction/predict.py --example
~~~

## Learning Artefacts and Reported Results

Day 1-5 results, Week 1 historical notes, Week 2 model-selection notes, and the Day 14 audit are preserved in [docs/week01_notes.md](docs/week01_notes.md), [docs/week02_model_selection.md](docs/week02_model_selection.md), [projects/iot_failure_prediction/README.md](projects/iot_failure_prediction/README.md), and the committed [final_test_metrics.json](outputs/iot_failure_prediction/final_test_metrics.json). Those outputs are records of the stated exercises and protocols, not a performance claim.

For the current Day 14 AI4I workflow, the authoritative model, threshold, and final metrics remain the code and the files under outputs/iot_failure_prediction. The workflow compares candidates on training folds, selects its threshold on validation predictions, and evaluates the reserved test set once.

## Current Limitations

- Examples use small instructional datasets or the synthetic AI4I dataset.
- The project does not yet include temporal or external validation, calibration, drift monitoring, or edge-inference measurements.
- Feature importance indicates model association, not causal explanation.

## Next Learning Directions

1. Class imbalance treatment and cost-sensitive evaluation.
2. Probability calibration and threshold analysis.
3. Explainability with permutation importance or SHAP.
4. Temporal or grouped validation and drift monitoring.
5. Edge-inference profiling and PyTorch foundations.

## Data Source and License

Day 1, 2, and 4 use scikit-learn Iris; Day 3 uses Diabetes; Day 5 uses Breast Cancer Wisconsin. The AI4I workflow uses the public [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) through [ucimlrepo](https://pypi.org/project/ucimlrepo/). Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

Released under the [MIT License](LICENSE).
