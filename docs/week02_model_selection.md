# 第二周：模型选择与评估（Day 08–Day 14）

本周将“训练一个模型”扩展为可审查的选择流程：先隔离测试集，再仅用训练集完成交叉验证和调参；如需分类阈值，则只从验证集选择；测试集最后只评价一次。

## Day 08 · Week 2 Day 1 · K-Nearest Neighbors

KNN（K-Nearest Neighbors）以欧氏距离衡量样本接近程度：

$$d(\mathbf{x},\mathbf{z})=\sqrt{\sum_{j=1}^{p}(x_j-z_j)^2}.$$

量纲较大的特征会支配距离，因此 KNN 需在 `Pipeline` 中使用 `StandardScaler`。`K` 太小会对局部噪声敏感，容易过拟合；`K` 太大会过度平滑，容易欠拟合。KNN 的“训练”主要保存样本，通常很快；预测时要比较邻居，可能较慢。本实验在 Iris 的 70/15/15 分层划分中，以验证集选择 `K=1, weights=uniform`，最终一次测试 Accuracy 为 0.9565。

## Day 09 · Week 2 Day 2 · Decision Tree

决策树从 Root（根节点）开始，Internal Node（内部节点）按阈值分支，Leaf（叶节点）给出预测。二分类 Gini impurity 为：

$$G=1-\sum_{k}p_k^2.$$

候选阈值选择使子节点的加权不纯度最小：

$$G_{split}=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R.$$

`max_depth` 限制树的深度；`min_samples_split` 限制可继续分裂的最小样本数；`min_samples_leaf` 限制叶节点最小样本数。树依赖排序与阈值而非距离，因此通常不需标准化；过深的树会记住训练细节而过拟合。本实验验证集选出 `max_depth=1`，最终测试 Accuracy 为 0.8721。

## Day 10 · Week 2 Day 3 · Random Forest

Random Forest 对训练样本做 Bootstrap sampling（有放回抽样），训练许多树后以多数投票（Bagging）输出分类。它同时引入样本随机性和特征随机性（`max_features`），从而降低单棵树对具体训练样本的敏感性，通常更稳定。`n_estimators` 控制树数量，`max_depth`、`min_samples_leaf` 控制单树复杂度。特征重要性只表示模型预测关联，不表示故障原因或因果关系。本次验证选择 `n_estimators=50, max_depth=None, min_samples_leaf=1, max_features=0.5`，最终测试 Accuracy 为 0.9070。

## Day 11 · Week 2 Day 4 · Support Vector Machine

SVM 寻找 Maximum Margin（最大间隔）Hyperplane（超平面）；靠近边界、决定边界位置的训练点是 Support Vectors（支持向量）。Hard margin 假设完全可分，Soft margin 通过惩罚允许少量违例。Kernel trick（核技巧）以隐式特征映射实现非线性边界。`C` 越大越强调训练误差、边界可能更复杂；`C` 越小正则化更强。RBF 的 `gamma` 越大，单个样本影响范围越小、边界越曲折；越小则更平滑。SVM 依赖内积和距离，必须在 Pipeline 内标准化。ROC-AUC 用 `decision_function` 的排序分数计算，不需要设置 `probability=True`。

## Day 12 · Week 2 Day 5 · Cross-validation 与 GridSearchCV

K-fold cross-validation 将训练数据分为 $K$ 折，轮流用一折验证、其余折训练。`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 在每折保持类别比例；结果报告 mean ± standard deviation，前者反映平均表现，后者反映切分敏感性。模型参数由训练学习，超参数在训练前设定。`GridSearchCV` 的组合数等于每个参数候选数的乘积；本实验采用 3×2×2=12 个组合，并以 `refit=True` 在训练集上重拟合最佳方案。参数名必须穿过 Pipeline：`svc__C`、`svc__kernel`、`svc__gamma`。测试集不进入 `GridSearchCV.fit`，以避免选择偏差。

实际结果：最佳参数为 `C=10.0`、`kernel=rbf`、`gamma=0.01`；训练集五折 CV ROC-AUC 为 0.9964 ± 0.0049；隔离测试集最终 ROC-AUC 为 0.9977。

## Day 13 · Week 2 Day 6 · 公平模型比较

公平比较先固定同一数据集、同一训练部分与完全相同的五个分层折；只有训练部分进入 CV，隔离测试集不用于挑选模型。保留 `DummyClassifier` 作为最低基线。比较 Accuracy、Precision、Recall、F1 和 ROC-AUC 的均值与标准差；本脚本按训练 CV 的平均 ROC-AUC 排序，而不是用测试结果反复挑选。

| 模型 | CV ROC-AUC mean ± std | CV F1 mean ± std |
| --- | ---: | ---: |
| Logistic Regression | 0.9959 ± 0.0050 | 0.9825 ± 0.0078 |
| SVC | 0.9956 ± 0.0048 | 0.9756 ± 0.0113 |
| Random Forest | 0.9896 ± 0.0087 | 0.9700 ± 0.0146 |
| KNN | 0.9886 ± 0.0084 | 0.9707 ± 0.0159 |
| Decision Tree | 0.9344 ± 0.0157 | 0.9373 ± 0.0105 |
| Dummy | 0.5000 ± 0.0000 | 0.7703 ± 0.0000 |

## Day 14 · Week 2 Day 7 · AI4I 综合应用

AI4I 2020 是反映工业预测性维护特征的合成数据集，不是从真实工厂直接采集的运行数据。来源为 UCI Machine Learning Repository，dataset id=601：Matzka, S. (2020), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本次完整真实运行验证 10,000 条样本，其中 `Machine failure`：0 为 9,661，1 为 339。

目标仅为 `Machine failure`。允许特征仅有 `Type`、空气温度、过程温度、转速、扭矩和工具磨损；`UID/UDI`、`Product ID`、`TWF/HDF/PWF/OSF/RNF` 均排除。后五列直接描述故障模式，进入特征会导致目标泄漏。预处理器只在每个训练折或训练集上 `fit`。

完整流程固定 `random_state=42`，按 60/20/20 分层为训练/验证/测试（6000/2000/2000）。训练部分的 5 折 CV 比较 Dummy、Logistic Regression、KNN、Decision Tree、Random Forest 和 SVM，以及可用模型的 `class_weight="balanced"` 版本。仅对平衡 Random Forest 和平衡 SVM 做适中网格搜索；测试集不参与 CV、调参或阈值选择。

| 阶段 | 实际结果 |
| --- | --- |
| 最佳调参候选 | `DecisionTreeClassifier` |
| 实际超参数 | `max_depth=8, min_samples_leaf=1` |
| 调参 CV F1 | 0.6841 ± 0.0476 |
| 验证集阈值 | 0.19；按 F1、Recall、Precision 顺序选择 |
| 测试 Accuracy | 0.9740 |
| 测试 Precision | 0.6026 |
| 测试 Recall | 0.6912 |
| 测试 F1 | 0.6438 |
| 测试 ROC-AUC | 0.8983 |
| 测试 Average Precision（PR-AUC） | 0.5843 |
| False positives / false negatives | 31 / 21 |

## 指标与泄漏检查

Accuracy 是全部正确比例；Precision 衡量预测为正时的可靠性；Recall 衡量真实正类被找回的比例；F1 是 Precision 与 Recall 的调和平均。ROC-AUC 衡量跨阈值排序能力；类别稀少时还应看 PR-AUC / Average Precision。常见泄漏包括：在全量数据上拟合缩放器、用测试集选超参数或阈值、把 ID、未来信息或故障模式标签作为特征。测试集只应在模型、超参数和阈值都固定后使用一次。

## 运行命令与产物

```powershell
python src/day08_knn.py
python src/day09_decision_tree.py
python src/day10_random_forest.py
python src/day11_svm.py
python src/day12_cross_validation.py
python src/day13_model_comparison.py
python projects/iot_failure_prediction/train.py
python -m pytest -q
```

基础脚本的输出依次位于 `outputs/knn/`、`decision_tree/`、`random_forest/`、`svm/`、`cross_validation/`、`model_comparison/`。Day 14 输出位于 `outputs/iot_failure_prediction/`，包含基线/调参比较 CSV、最终指标 JSON、模型、阈值、混淆矩阵、ROC/PR 曲线、阈值分析、特征重要性和 FP/FN CSV。
