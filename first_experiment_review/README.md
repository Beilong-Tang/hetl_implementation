# Feature-Based Transfer Learning for Network Security - Reproduction

本项目用于复现论文 *Feature-Based Transfer Learning for Network Security*。

## 目录

- `paper/`：论文原文及论文笔记
- `data/raw/`：未经修改的 NSL-KDD 原始数据
- `data/processed/`：清洗、编码、归一化后的数据
- `data/splits/`：源域/目标域抽样索引，保证实验可重复
- `notebooks/`：数据探索和结果分析 Notebook
- `src/`：正式的数据处理、HeTL、基线模型及评价代码
- `configs/`：实验参数、随机种子和数据路径
- `scripts/`：从终端启动预处理和实验的脚本
- `tests/`：数据泄漏、样本数量和算法维度测试
- `experiments/logs/`：每次实验的参数与日志
- `outputs/figures/`：ROC、混淆矩阵和潜空间图
- `outputs/tables/`：Accuracy、F1、AUC 等结果表
- `outputs/models/`：保存的模型和投影矩阵
- `docs/`：复现过程、论文差异和实验记录

## 创建环境

```bash
cd ~/Desktop/Feature-based_transfer_learning_for_network_security_reproduction
uv sync
```

验证环境与原始数据：

```bash
uv run python scripts/check_environment.py
```

## 计划复现的任务

1. DoS -> R2L
2. DoS -> Probe
3. Probe -> R2L
4. 无迁移学习基线：CART、Random Forest、Linear SVM、KNN、Naive Bayes
5. 同构特征空间 HeTL
6. 信息增益选择后的异构特征空间 HeTL

原始数据不可被脚本覆盖。所有中间数据应写入 `data/processed/` 或 `data/splits/`。

## 正式 HeTL 实验

先运行一个任务、一个随机种子，检查损失是否下降以及所需时间：

```bash
uv run python -m src.run_hetl --task dos_to_r2l --seed 42 --tune
```

这一步遍历配置文件中的 6 个 `beta` 和 5 个隐空间维数 `k`。每种分类器分别根据
500 条目标域验证数据的 attack F1 选择参数，随后只在剩余目标测试集上报告最终指标。

确认运行正常后，可运行三个任务的 seed 42：

```bash
uv run python -m src.run_hetl --task all --seed 42 --tune --resume
```

最后运行三个任务和全部 10 个随机种子：

```bash
uv run python -m src.run_hetl --task all --all-seeds --tune --resume
```

`--resume` 会跳过已经同时生成搜索明细和测试结果的任务/种子，适合长时间实验中断后继续。
主要输出为：

- `outputs/tables/hetl_results.csv`：每个任务、种子、分类器的最终测试结果；
- `outputs/tables/hetl_summary.csv`：Accuracy、Precision、Recall、F1、AUC 的均值和标准差；
- `outputs/tables/hetl_selected_parameters.csv`：不同 `beta`、`k` 被选中的次数；
- `outputs/tables/hetl_search_*.csv`：验证集上的全部候选参数结果；
- `experiments/logs/hetl_*.csv`：每次拟合的损失变化，用于判断收敛。
