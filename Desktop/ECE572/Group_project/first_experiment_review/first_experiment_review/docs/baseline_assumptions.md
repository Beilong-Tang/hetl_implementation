# No-TL baseline 实现约定

论文明确指定五种基础分类器：CART、Random Forest、Linear SVM、Naive Bayes 和 KNN，
并在 Table II、Table III 中报告 Accuracy 与 F1，在 Fig. 5 中给出 ROC/AUC。

论文没有公布这些分类器的完整超参数，因此本项目采用以下显式可复现参数：

- CART：Gini，完整生长，`min_samples_split=2`。
- Random Forest：100棵树，Gini，完整生长。
- Linear SVM：`C=1.0`，squared hinge，最多10,000次迭代。
- Gaussian Naive Bayes：`var_smoothing=1e-9`。
- KNN：`k=5`，uniform weights，Euclidean distance。

这些参数是复现实验假设，不是论文作者公开的原参数。

## 无迁移学习协议

1. 分类器只在 `X_source, y_source` 上拟合。
2. 不执行任何 HeTL 表示变换。
3. `y_target` 不参与拟合。
4. 500条目标验证样本保留给后续 HeTL 参数选择。
5. baseline 只在其余1490条目标测试样本上评价。
6. attack=1 是 Precision、Recall、F1 的正类。

输出指标包括 Accuracy、Precision、Recall、F1、ROC-AUC，以及 TN、FP、FN、TP。
