# Feature-based Transfer Learning for Network Security：第一次复现实验报告

## 1. 审查范围

本文件夹保存第一次正式实验的完整复现材料。实验对象是论文 *Feature-based Transfer Learning for Network Security* 中“相同特征空间、未知攻击检测”部分，对应三个任务：

- DoS → R2L
- DoS → Probe
- Probe → R2L

本报告只讨论最初的 `one-hot + 源域/目标域分别 Min-Max` 实验。后续的 baseline 消融实验和 `joint Min-Max` HeTL 实验没有放入本文件夹，避免两套结果混合。

## 2. 文件内容

| 目录或文件 | 用途 |
|---|---|
| `paper/` | 论文 PDF |
| `data/raw/` | NSL-KDD 原始 `KDDTrain+` 和 `KDDTest+` |
| `data/processed/` | 30个任务/种子的处理后 NPZ 数据和预处理清单 |
| `data/splits/` | 每条样本的原始行号、域、标签及目标验证/测试划分 |
| `src/` | 预处理、数据接口、baseline、HeTL、运行与可视化代码 |
| `configs/experiment.json` | 第一次实验的统一配置 |
| `outputs/tables/` | baseline、HeTL、参数搜索和逐次结果 |
| `outputs/figures/` | 第一次实验生成的图 |
| `experiments/logs/` | 900次正式 HeTL 候选表示的损失记录 |
| `docs/` | baseline 和 HeTL 未公开细节的实现假设 |
| `tests/` | 数据接口、baseline 和 HeTL 测试 |

文件夹共1058个文件，约58 MB。`.venv`、缓存、后续消融结果和后续联合归一化实验未收录。

## 3. 数据处理配置

第一次实验只用 `KDDTrain+.txt` 构造论文的三个迁移任务；`KDDTest+.txt` 随包提供，但没有进入本次训练或评价。

每个任务和随机种子的处理逻辑如下：

1. 将攻击名称映射为 DoS、Probe、R2L、U2R，正常流量记为 normal。
2. 从源攻击类别抽取995条攻击，从目标攻击类别抽取995条攻击。
3. 源域和目标域分别抽取995条正常样本，两个域的正常样本不重叠。
4. 源域和目标域各有1990条记录，类别均衡。
5. 删除 `difficulty`，因为它是数据集构造阶段得到的难度信息，不是实际网络特征。
6. `protocol_type`、`service`、`flag` 使用 one-hot 编码；类别词表由源域和目标域的无标签特征共同建立。
7. 删除在源域和目标域中都为常量的编码列。
8. 源域和目标域分别拟合 Min-Max scaler，将连续特征缩放到约 `[0,1]`。
9. 目标域按类别分层划分：500条作为参数验证集，其余1490条作为最终测试集。

| 任务 | 第一次实验输入维数 |
|---|---:|
| DoS → R2L | 108 |
| DoS → Probe | 107 |
| Probe → R2L | 103 |

使用10个随机种子：`42, 43, 44, 45, 46, 47, 48, 49, 50, 51`。同一任务内源域和目标域的列数、顺序和含义完全一致。

## 4. Baseline 配置

所有 baseline 只使用源域的 `X_source, y_source` 训练，在目标域1490条测试样本上评价。500条目标验证样本不参与 baseline 拟合。

| 分类器 | 第一次实验参数 |
|---|---|
| CART | Gini，`splitter=best`，不限深度，`min_samples_split=2` |
| Random Forest | 100棵树，Gini，不限深度，`min_samples_split=2` |
| Linear SVM | `LinearSVC(C=1.0, loss=squared_hinge, max_iter=10000)` |
| Gaussian Naive Bayes | `var_smoothing=1e-9` |
| KNN | `k=5`，uniform weights，Euclidean distance |

论文只公布了分类器种类，没有公布这些完整参数。因此这些参数属于本项目的复现假设。

## 5. HeTL 配置

HeTL 使用论文式(4)的目标：

```text
||S - Vs Ps||²_F + ||T - Vt Pt||²_F + beta ||Vs - Vt||²_F
```

主要配置：

| 参数 | 第一次实验设置 |
|---|---|
| β 搜索范围 | `0.001, 0.01, 0.1, 1, 10, 100` |
| 潜在维数 k | `2, 5, 10, 20, 30` |
| 每个任务/种子的候选数 | 30 |
| 学习率 | `1e-4` |
| 最大迭代次数 | 1000 |
| 收敛阈值 | 相对损失变化 `< 1e-7` |
| patience | 连续10步满足阈值 |
| 初始化 | 截断 SVD |
| 正交约束 | 每次更新后进行 QR 正交化 |
| P 更新 | 固定 V 时使用闭式解 `P = VᵀX` |
| 参数选择 | 对每种分类器最大化500条目标验证样本上的 attack F1 |
| 平分规则 | Accuracy、ROC-AUC、较小 k、较小 β |

配置文件虽然记录了两个候选学习率，但第一次正式搜索实际固定使用 `1e-4`，没有把学习率纳入网格搜索。

HeTL 是传导式实验：优化时使用全部源域和目标域特征，包括目标测试样本的无标签特征；目标测试标签只用于最终评价。NSL-KDD 的源域和目标域样本没有天然一一配对，第一次实现沿用抽样后的行顺序计算 `||Vs-Vt||`，这是论文没有解释的关键实现假设。

## 6. Baseline 与论文对比

下表中“复现”是10个随机种子的均值。F1 以 attack=1 为正类。

| 任务 | 分类器 | 论文 Acc | 复现 Acc | 论文 F1 | 复现 F1 |
|---|---|---:|---:|---:|---:|
| DoS→R2L | CART | 0.530 | 0.496 | 0.120 | 0.020 |
| DoS→R2L | RF | 0.500 | 0.500 | 0.000 | 0.003 |
| DoS→R2L | SVM | 0.490 | 0.515 | 0.000 | 0.070 |
| DoS→R2L | NB | 0.360 | 0.506 | 0.000 | 0.034 |
| DoS→R2L | KNN | 0.490 | 0.505 | 0.000 | 0.037 |
| DoS→Probe | CART | 0.630 | 0.692 | 0.260 | 0.561 |
| DoS→Probe | RF | 0.630 | 0.666 | 0.450 | 0.498 |
| DoS→Probe | SVM | 0.740 | 0.680 | 0.650 | 0.533 |
| DoS→Probe | NB | 0.540 | 0.540 | 0.540 | 0.151 |
| DoS→Probe | KNN | 0.730 | 0.790 | 0.680 | 0.731 |
| Probe→R2L | CART | 0.520 | 0.471 | 0.120 | 0.063 |
| Probe→R2L | RF | 0.500 | 0.497 | 0.010 | 0.007 |
| Probe→R2L | SVM | 0.470 | 0.513 | 0.040 | 0.082 |
| Probe→R2L | NB | 0.650 | 0.506 | 0.570 | 0.063 |
| Probe→R2L | KNN | 0.510 | 0.505 | 0.080 | 0.044 |

15组 baseline 的平均绝对误差：Accuracy `0.0455`，F1 `0.1197`，两者等权综合误差 `0.0826`。RF 在两个 R2L 任务上与论文非常接近；主要差异集中在 Naive Bayes，以及 DoS→Probe 的 CART F1。

## 7. HeTL 与论文对比

| 任务 | 分类器 | 论文 Acc | 复现 Acc | 论文 F1 | 复现 F1 |
|---|---|---:|---:|---:|---:|
| DoS→R2L | CART | 0.810 | 0.755 | 0.830 | 0.764 |
| DoS→R2L | RF | 0.780 | 0.715 | 0.800 | 0.698 |
| DoS→R2L | SVM | 0.810 | 0.721 | 0.830 | 0.708 |
| DoS→R2L | NB | 0.720 | 0.733 | 0.740 | 0.756 |
| DoS→R2L | KNN | 0.780 | 0.713 | 0.810 | 0.696 |
| DoS→Probe | CART | 0.780 | 0.668 | 0.760 | 0.678 |
| DoS→Probe | RF | 0.780 | 0.684 | 0.740 | 0.658 |
| DoS→Probe | SVM | 0.850 | 0.694 | 0.840 | 0.684 |
| DoS→Probe | NB | 0.760 | 0.617 | 0.730 | 0.664 |
| DoS→Probe | KNN | 0.780 | 0.663 | 0.750 | 0.655 |
| Probe→R2L | CART | 0.730 | 0.809 | 0.670 | 0.821 |
| Probe→R2L | RF | 0.750 | 0.811 | 0.770 | 0.815 |
| Probe→R2L | SVM | 0.750 | 0.643 | 0.790 | 0.648 |
| Probe→R2L | NB | 0.710 | 0.772 | 0.770 | 0.813 |
| Probe→R2L | KNN | 0.780 | 0.869 | 0.780 | 0.876 |

15组 HeTL 的平均绝对误差：Accuracy `0.0874`，F1 `0.0918`，两者等权综合误差 `0.0896`。

HeTL 相对本项目 baseline：

- F1 在15组比较中提高14组，唯一例外是 DoS→Probe/KNN。
- Accuracy 在15组比较中提高13组。
- Probe→R2L 的 CART、RF、NB、KNN 高于论文报告值；DoS→Probe 和多数 DoS→R2L 结果低于论文。
- 150个最终选中模型中，72个满足提前收敛条件，78个运行到1000步上限。

因此，第一次实验复现了论文的核心定性结论：HeTL 通常显著优于无迁移 baseline。但当前实现没有实现严格的数值复现。

## 8. 可能造成数值差异的因素

论文没有公开以下细节：

- 具体抽样行、随机种子和正常样本是否允许跨域重复；
- 三个类别字段的编码方法；
- Normalize 的确切方式；
- baseline 的完整超参数和软件版本；
- β、k 的搜索网格和选择指标；
- 学习率、初始化、正交化及收敛阈值；
- P 使用梯度更新还是公式中的闭式解；
- 源域与目标域无天然配对时，`||Vs-Vt||` 如何定义行对应；
- 500条目标有标签样本是否被排除在最终评价之外；
- F1 的正类和平均方式。

这些缺失信息足以造成当前的数值差异。特别需要组员检查的是：逐行对齐项的解释、one-hot 后不再是论文文字上的41维，以及78个最终模型达到迭代上限。

## 9. 复现命令

在本文件夹根目录打开终端：

```bash
uv sync
uv run python -m src.preprocessing --all-seeds
uv run python -m src.baselines --all-seeds
uv run python -m src.run_hetl --task all --all-seeds --tune --resume
uv run python -m pytest -q
```

已有 HeTL 搜索结果时，`--resume` 会跳过已经完成的任务/种子。若要从头复算，应另建输出目录或先备份现有结果，避免覆盖用于审查的结果。

## 10. 建议的组内检查顺序

1. 对照 `src/preprocessing.py` 和 `data/splits/`，确认论文数据构造理解是否合理。
2. 对照 `docs/hetl_implementation_assumptions.md`，重点讨论论文没有给出的数值实现选择。
3. 检查 `outputs/tables/baseline_summary.csv` 与 `hetl_summary.csv` 的10种子均值和标准差。
4. 检查 `hetl_results.csv`、`hetl_selected_parameters.csv` 以及对应损失日志，确认参数选择和收敛状态。
5. 将本报告的 Table II/III 对比作为组内意见记录的起点，不把当前数字表述为论文作者的原始实现。
