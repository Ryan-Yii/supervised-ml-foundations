# supervised-ml-foundations

第一周监督学习（Supervised Machine Learning）项目记录。项目以 Python、物联网（IoT）和边缘计算（Edge Computing）基础为起点，通过一组小而完整的实验梳理数据、模型、评估与数据泄漏（Data Leakage）。

## 本周实践重点

- 区分特征（Feature）和标签（Label），并完成基础数据探索。
- 正确划分训练集（Training Set）、验证集（Validation Set）和测试集（Test Set）。
- 使用 `Pipeline`、`ColumnTransformer` 将预处理与模型绑定，避免数据泄漏。
- 理解回归（Regression）与分类（Classification）的常用评价指标。
- 用一个物联网故障预测案例练习可复现、可保存、可预测的最小机器学习流程。

## 学习进度与实践内容

| 日期 | 主题 | 关键产物 |
| --- | --- | --- |
| Day 1 | Iris 数据探索 | 特征、标签、缺失值与类别分布检查 |
| Day 2 | 数据划分与 Pipeline | 70/15/15 分层划分和无泄漏标准化 |
| Day 3 | 线性回归（Linear Regression） | MAE、MSE、RMSE、R² 与散点图 |
| Day 4 | 逻辑回归（Logistic Regression） | 分类报告、预测概率与混淆矩阵 |
| Day 5 | 过拟合（Overfitting） | 决策树深度选择与训练/验证曲线 |
| Day 6 | IoT 故障预测 | AI4I 数据、特征编码、模型保存与单样本预测 |

## 项目结构

```text
supervised-ml-foundations/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── common.py
│   ├── day01_iris_exploration.py
│   ├── day02_data_split_pipeline.py
│   ├── day03_linear_regression.py
│   ├── day04_logistic_regression.py
│   └── day05_overfitting.py
├── projects/
│   └── iot_failure_prediction/
│       ├── README.md
│       ├── train.py
│       └── predict.py
├── outputs/
│   ├── regression/
│   ├── classification/
│   ├── overfitting/
│   └── iot_failure_prediction/
├── tests/
│   ├── test_smoke.py
│   └── test_iot_failure_prediction.py
└── docs/
    ├── week01_notes.md
    └── learning_checklist.md
```

## 运行环境

Python 3.10 及以上版本；以下命令在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若 PowerShell 禁止激活脚本，可在当前窗口临时执行 `Set-ExecutionPolicy -Scope Process Bypass`，或直接使用 `.venv\Scripts\python.exe` 运行下方命令。

## 运行命令

所有命令均从仓库根目录执行，代码仅使用相对路径保存输出。

```powershell
python src/day01_iris_exploration.py
python src/day02_data_split_pipeline.py
python src/day03_linear_regression.py
python src/day04_logistic_regression.py
python src/day05_overfitting.py
python projects/iot_failure_prediction/train.py
python projects/iot_failure_prediction/predict.py --example
python -m unittest discover -s tests -v
python -m pytest -q
```

Day 6 首次运行会通过 `ucimlrepo` 下载 UCI 数据集；它不需要手动放置 CSV，也不会保存下载的数据副本。

## 评价指标解释

- **MAE（Mean Absolute Error）**：平均绝对误差，越小越好。
- **MSE（Mean Squared Error）**：平方误差均值，会更严厉地惩罚大误差，越小越好。
- **RMSE（Root Mean Squared Error）**：MSE 的平方根，与目标值同量纲，越小越好。
- **R²（决定系数）**：回归拟合优度，不是分类准确率；在模型比“预测均值”更差时可以为负数。
- **Accuracy（准确率）**：所有预测中正确的比例；类别不平衡时不能单独作为结论。
- **Precision（精确率）**：预测为正类的样本中，真正为正类的比例。
- **Recall（召回率）**：真实正类中被模型识别出来的比例。
- **F1**：Precision 和 Recall 的调和平均，在两者都重要时更有参考价值。

## 数据泄漏说明

数据泄漏指训练阶段获得了预测时不应知道的信息，从而让离线指标虚高。

- Day 2 与 Day 4 将 `StandardScaler` 放进 `Pipeline`，并且只在训练集上 `fit`；验证集、测试集只做 `transform/predict`。
- Day 5 仅用验证集选择 `max_depth`，测试集只在最终模型确定后评估一次。
- Day 6 仅使用 `Type` 与五项正常运行参数；明确排除 `UDI`（题目中常写作 UID）、`Product ID`、`Machine failure` 和 `TWF/HDF/PWF/OSF/RNF` 故障模式标签。目标与故障模式信息绝不会进入特征矩阵。

## 运行结果摘要

下表只记录实际运行得到的结果；指标来自本仓库的固定随机种子运行。Day 6 会在首次训练时下载官方 UCI 数据。

| 实验 | 实际结果 |
| --- | --- |
| Day 1 | `X.shape=(150, 4)`、`y.shape=(150,)`；无缺失值；三个类别各 50 条 |
| Day 2 | 训练/验证/测试：105/22/23；验证集 Accuracy：0.8636 |
| Day 3 | MAE：42.7941；MSE：2900.1936；RMSE：53.8534；R²：0.4526 |
| Day 4 | 测试集 Accuracy：0.9333 |
| Day 5 | 最佳 `max_depth=1`；验证集 Accuracy：0.9529；最终测试集 Accuracy：0.8721 |
| Day 6 | 最终模型：平衡随机森林；阈值：0.4700；测试集 Accuracy：0.9815、Precision：0.7627、Recall：0.6618、F1：0.7087、ROC-AUC：0.9741、PR-AUC：0.7274 |

## 生成图表

以下图片由脚本实际运行后生成；在首次运行完成前，这些链接不会显示结果。

![线性回归真实值与预测值](outputs/regression/actual_vs_predicted.png)

![Iris 逻辑回归混淆矩阵](outputs/classification/confusion_matrix.png)

![决策树深度与准确率](outputs/overfitting/depth_vs_accuracy.png)

![AI4I 最终测试集混淆矩阵](outputs/iot_failure_prediction/confusion_matrix.png)

![AI4I 验证集阈值分析](outputs/iot_failure_prediction/threshold_analysis.png)

Day 6 运行后会生成 `outputs/iot_failure_prediction/model_comparison.csv`、`metrics.json`、`final_model.joblib`、`decision_threshold.json`、四张分析图和误报/漏报 CSV。完整流程见 [Day 6 项目说明](projects/iot_failure_prediction/README.md)。

## 当前限制

- 数据集规模较小，示例重点是学习流程而非追求业务部署精度。
- Day 6 已比较 Dummy、逻辑回归、决策树和随机森林，并完成验证集阈值选择；仍未进行时间外推验证、概率校准或业务成本敏感阈值优化。
- 缺少线上数据质量监控、漂移检测和边缘端部署；这些不属于第一周范围。

## 下一步实践计划

1. 学习交叉验证（Cross-Validation）和网格搜索（Grid Search）。
2. 比较树模型与逻辑回归，并依据业务代价选择阈值。
3. 学习特征工程、类别不平衡处理，以及模型解释方法。
4. 将经过验证的小模型导出到边缘设备可用的推理格式。

## 数据来源与引用

- Day 1、2、4 使用 scikit-learn 内置 Iris 数据集；Day 3 使用内置 Diabetes 数据集；Day 5 使用内置 Breast Cancer Wisconsin 数据集。
- Day 6 使用 [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)，通过官方 [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) 包的 `fetch_ucirepo(id=601)` 获取。
- 数据集引用：Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)。许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
