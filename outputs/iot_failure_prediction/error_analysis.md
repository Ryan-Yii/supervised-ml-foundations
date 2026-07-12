# 最终测试集错误分析
下列统计完全基于最终测试集一次预测；它们描述误报（False Positive）和漏报（False Negative）的实际特征，不构成因果结论。

## 误报（False Positives）
样本数：14
平均数值特征：
- Air temperature [K]: 301.2643
- Process temperature [K]: 310.7714
- Rotational speed [rpm]: 1687.2857
- Torque [Nm]: 42.5857
- Tool wear [min]: 111.3571
设备类型分布：{'L': 10, 'M': 3, 'H': 1}

## 漏报（False Negatives）
样本数：23
平均数值特征：
- Air temperature [K]: 301.0609
- Process temperature [K]: 310.5304
- Rotational speed [rpm]: 1447.5217
- Torque [Nm]: 48.2391
- Tool wear [min]: 142.7826
设备类型分布：{'L': 11, 'M': 10, 'H': 2}
