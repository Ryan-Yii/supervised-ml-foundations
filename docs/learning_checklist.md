# 第一周学习清单

以下状态记录的是学习流程和脚本验证状态；“已完成学习/复现”由学习者本人确认。

## 数据与任务

- [x] 已完成学习：能用自己的话区分 Feature、Label、Classification 与 Regression。
- [x] 已完成复现：手动检查 Iris 数据的样本数、特征数、缺失值和类别分布。
- [x] 已完成学习：解释 `X.shape=(150, 4)` 与 `y.shape=(150,)` 的含义。

## 数据划分与预处理

- [x] 已完成学习：理解训练集、验证集和测试集的不同用途。
- [x] 已完成复现：用 `stratify` 复现 70%/15%/15% 分类数据划分。
- [x] 已完成学习：说明为何 `StandardScaler` 只能在训练集上 `fit`。
- [x] 已完成复现：观察 Pipeline 如何减少 Data Leakage 风险。

## 模型与指标

- [x] 已完成学习：理解 Parameter 与 Hyperparameter 的区别。
- [x] 已完成复现：运行线性回归并核对 MAE、MSE、RMSE、R² 的输出。
- [x] 已完成学习：解释 R² 为什么不是分类 Accuracy，以及为何可能小于 0。
- [x] 已完成复现：阅读逻辑回归的 `classification_report` 和 `predict_proba`。
- [x] 已完成学习：能解释 Accuracy、Precision、Recall、F1 的取舍。
- [x] 已完成复现：根据验证集曲线选择决策树的 `max_depth`。

## IoT 故障预测

- [x] 已完成学习：说明为何 UDI/UID、Product ID、TWF/HDF/PWF/OSF/RNF 不能做特征。
- [x] 已完成复现：下载 AI4I 数据并运行训练脚本。
- [x] 已完成复现：加载已保存模型，检查示例设备的预测类别和故障概率。

## 客观脚本状态

- [x] 已由脚本验证：Day 1–Day 5 已在项目虚拟环境中运行；结果记录在根目录 README。
- [x] 已由脚本验证：Day 6 已下载官方数据、生成真实指标、模型、阈值和分析产物；自动化测试已在项目虚拟环境中运行。
