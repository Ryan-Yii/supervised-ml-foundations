# 第一周机器学习笔记

## Day 6 历史实验快照（不代表当前模型）

第一周的 Day 6 曾以固定 `random_state=42` 和 60/20/20 分层切分，比较 Dummy、Logistic Regression、Decision Tree 与 Random Forest（含部分 `class_weight="balanced"` 变体），并只在验证集选择阈值。该旧版候选流程的历史快照为：平衡随机森林、阈值 `0.4700`、测试 F1 `0.7087`。它不包含 Day 14 的 KNN/SVM 候选和训练集 GridSearchCV 调参，因此**不代表当前模型，因实验协议不同不能直接与 Day 14 比较**。当前可复现的最终结果以根 README 的 Day 14 说明和 `outputs/iot_failure_prediction/final_test_metrics.json` 为准。

## Feature 与 Label

特征（Feature）是模型用于预测的输入变量，常记作 `X`。标签（Label）是要预测的答案，常记作 `y`。例如 Day 6 中温度、扭矩和设备类型是特征，`Machine failure` 是标签。训练特征中不能包含标签本身，也不能包含由标签直接推导出的信息。

## Classification 与 Regression

分类（Classification）预测离散类别，例如“正常/故障”。回归（Regression）预测连续数值，例如疾病进展指标。逻辑回归（Logistic Regression）名称中有“回归”，但常用于分类；线性回归（Linear Regression）用于连续值预测。

## Parameter 与 Hyperparameter

参数（Parameter）是模型从训练数据中学习得到的值，例如线性回归系数。超参数（Hyperparameter）是在训练前指定的设置，例如决策树的 `max_depth`。应使用验证集选择超参数，不能利用测试集。

## Training、Validation、Test Set

- 训练集（Training Set）：用于拟合模型和预处理器。
- 验证集（Validation Set）：用于比较方案、选择超参数。
- 测试集（Test Set）：在方案确定后只用于一次最终泛化评估。

分层划分（Stratified Split）会尽可能保持各子集中类别比例一致，尤其适合分类任务。

## Standardization

标准化（Standardization）常把每一数值特征转换为均值约为 0、标准差约为 1 的尺度。`StandardScaler` 的统计量只能从训练集计算；对全量数据先标准化会把验证/测试集的信息泄漏给训练阶段。

## Pipeline

Pipeline 把预处理和模型按顺序组合。调用 `pipeline.fit(X_train, y_train)` 时，缩放器只学习训练数据，预测新数据时会自动使用相同的变换。这既减少重复代码，也降低忘记正确处理数据的风险。

## MAE、MSE、RMSE、R²

- MAE：误差绝对值的平均值，直观且与目标同量纲。
- MSE：误差平方的平均值，对大误差更敏感。
- RMSE：MSE 开平方后恢复为目标量纲。
- R²：模型相对“预测训练/测试标签均值”基线的解释程度；它不是准确率，可能为负数。

## Accuracy、Precision、Recall、F1

- Accuracy：所有样本的正确比例。
- Precision：被预测为正类的样本中真正为正类的比例。
- Recall：真实正类被找出的比例。
- F1：Precision 和 Recall 的调和平均。

故障预测常存在类别不平衡，因此必须把 Precision、Recall、F1 与 Accuracy 放在一起理解。

## Underfitting 与 Overfitting

欠拟合（Underfitting）表示模型过于简单，训练集和验证集都表现不佳。过拟合（Overfitting）表示模型在训练集表现非常好，却不能泛化到验证/测试集。Day 5 通过比较不同树深度下训练与验证准确率来选择复杂度。

## Data Leakage

数据泄漏（Data Leakage）是模型在训练或调参阶段接触到现实预测时不可获得的信息。常见来源包括：用全体数据拟合标准化器、用测试集调超参数、把 ID 或未来信息当作特征、把故障模式标签用于预测故障是否发生。防止泄漏的实践是：先划分数据；只在训练集拟合；用 Pipeline/ColumnTransformer 固化流程；最后才看测试集。
