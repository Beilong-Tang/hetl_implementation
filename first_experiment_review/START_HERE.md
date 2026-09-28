# 组员审查入口

请先阅读 [EXPERIMENT_REPORT_CN.md](EXPERIMENT_REPORT_CN.md)。报告说明第一次实验的数据处理、baseline 与 HeTL 参数、论文对比、收敛情况和需要审查的实现假设。

建议首先查看：

- `outputs/tables/baseline_summary.csv`
- `outputs/tables/hetl_summary.csv`
- `outputs/tables/hetl_results.csv`
- `docs/hetl_implementation_assumptions.md`
- `src/preprocessing.py`
- `src/hetl.py`

本包仅包含第一次正式实验。后续 baseline 消融和 joint Min-Max HeTL 结果没有混入。
