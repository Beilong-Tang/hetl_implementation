# 复现计划

## 第一阶段：数据核验

- 核验 `KDDTrain+.txt` 为 125,973 行、43 列。
- 核验训练集类别数量与论文 Table I 一致。
- 删除 `difficulty`，保留 41 个网络特征。
- 将攻击名称映射为 normal、DoS、Probe、R2L、U2R。
- 排除样本极少的 U2R，与论文保持一致。

## 第二阶段：领域构造

- 构造 DoS -> R2L、DoS -> Probe、Probe -> R2L。
- 每个域使用 995 条 normal 和 995 条 attack。
- 源域与目标域的正常记录不得重复。
- 保存每次抽样的行索引和随机种子。

## 第三阶段：预处理与基线

- 对 protocol_type、service、flag 做 one-hot 编码。
- 检测并删除常量特征。
- 分别对源域和目标域执行 Min-Max 归一化。
- 训练 CART、Random Forest、Linear SVM、KNN、Naive Bayes。

## 第四阶段：HeTL

- 实现论文目标函数和交替优化。
- 对潜表示执行正交化，并记录这是对论文实现细节的合理补充。
- 使用目标域 500 条验证标签选择 beta 和 k。
- 剩余目标域标签仅用于最终测试。

## 第五阶段：异构特征空间

- 源域和目标域分别按信息增益选择特征。
- 目标域信息增益只能使用 500 条验证标签。
- 比较手工映射、CORAL、HeMap（如可实现）和 HeTL。

## 第六阶段：报告

- 对 10 个随机种子报告均值与标准差。
- 指标包括 Accuracy、Precision、Recall、F1、ROC-AUC 和混淆矩阵。
- 单独记录论文未披露的实现选择及敏感性实验。
