# 可视化与论文图的对应关系

## 当前可以真实生成

- `feature_distribution_dos_vs_r2l.png`：对应 Fig.1 的思想，展示 DoS 和 R2L 在
  `serror_rate`、`srv_serror_rate` 上的分布变化。论文图的精确抽样与绘图库未公开，
  所以本项目使用 seed 42 的实际源域/目标域样本和透明散点。
- `dos_to_r2l_roc.png`：对应 Fig.5。使用同一目标测试分区，分别绘制 Linear SVM、
  Naive Bayes、KNN 的 No-TL 与固定参数 HeTL ROC。论文使用哪些样本和阈值未公开。
- `probe_to_r2l_latent.png`：对应 Fig.7。使用 k=2 的 Vs、Vt 直接绘制。目标标签只用于
  绘图着色，不参与 HeTL 拟合。潜空间的旋转、符号和尺度本来就不唯一，因此坐标值无需与
  论文的 -40~40 一致，重点是相对分布和分类边界。
- `baseline_summary.png`：将论文 Table II/III 的展示扩展为本项目10个随机种子的均值和
  标准差，避免只展示一次随机抽样。
- `baseline_vs_hetl_f1.png`：用于检查当前固定参数管线，不是正式调参结果，标题明确写出
  beta=1、k=10、未调参。

## 当前不能生成

- Fig.6：需要先实现异构特征选择和论文的手工映射baseline。
- Fig.8：需要先实现 CORAL 和 HeMap；在算法完成前不能用占位数值或论文图片冒充结果。
