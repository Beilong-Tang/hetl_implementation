"""生成与论文实验对应的静态可视化。

运行全部当前可实现图：
    uv run python -m src.visualize_results --figure all --seed 42

图与论文的对应关系：
    feature_distribution.png  -> Fig. 1 的特征分布思想
    dos_to_r2l_roc.png         -> Fig. 5 ROC曲线
    probe_to_r2l_latent.png    -> Fig. 7 投影后数据分布
    baseline_summary.png       -> Table II/III 的多随机种子汇总可视化
    baseline_vs_hetl_f1.png    -> Table III 的 No-TL/HeTL F1 对比（固定参数烟雾实验）

Fig. 6 需要异构特征空间与手工映射；Fig. 8 需要 CORAL、HeMap。对应算法尚未
实现时，本脚本不会伪造这两张图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 只生成文件，不弹出GUI窗口。这样脚本可在 VS Code 终端、CI 和无显示环境中运行。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from .baselines import build_classifiers, positive_class_scores
from .data_loader import ROOT, TransferTaskData, load_task
from .hetl import HeTLConfig, HeTLResult, fit_hetl

FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
CLASSIFIER_ORDER = ["CART", "RandomForest", "LinearSVM", "NaiveBayes", "KNN"]
COLORS = {
    "normal_source": "#4C78A8",
    "attack_source": "#E45756",
    "normal_target": "#72B7B2",
    "attack_target": "#F2CF5B",
}


def configure_style() -> None:
    """统一论文图风格。"""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存：{path}")
    return path


def feature_index(data: TransferTaskData, name: str) -> int:
    matches = np.flatnonzero(data.feature_names == name)
    if len(matches) != 1:
        raise ValueError(f"特征 {name!r} 不存在或不唯一")
    return int(matches[0])


def plot_feature_distribution(seed: int) -> Path:
    """重现Fig.1思想：同样两个流量特征在DoS和R2L中分布不同。"""
    data = load_task("dos_to_r2l", seed)
    x_index = feature_index(data, "serror_rate")
    y_index = feature_index(data, "srv_serror_rate")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    panels = [
        (axes[0], data.X_source, data.y_source, "DoS source domain", "DoS"),
        (axes[1], data.X_target, data.y_target, "R2L target domain", "R2L"),
    ]
    for axis, X, y, title, attack_name in panels:
        normal = y == 0
        attack = y == 1
        axis.scatter(
            X[normal, x_index],
            X[normal, y_index],
            s=13,
            alpha=0.35,
            label="Normal",
            color=COLORS["normal_source"],
            edgecolors="none",
        )
        axis.scatter(
            X[attack, x_index],
            X[attack, y_index],
            s=13,
            alpha=0.35,
            label=attack_name,
            color=COLORS["attack_source"],
            edgecolors="none",
        )
        axis.set_title(title)
        axis.set_xlabel("serror_rate (Min-Max scaled)")
        axis.legend()
    axes[0].set_ylabel("srv_serror_rate (Min-Max scaled)")
    fig.suptitle("Feature distribution shift: DoS vs R2L", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "feature_distribution_dos_vs_r2l.png")


def fitted_hetl(data: TransferTaskData, k: int) -> HeTLResult:
    """可视化使用的固定参数HeTL；不是正式beta/k调参结果。"""
    return fit_hetl(
        data.X_source,
        data.X_target,
        HeTLConfig(
            latent_dimension=k,
            beta=1.0,
            learning_rate=1e-4,
            max_steps=100,
            tolerance=1e-7,
            patience=10,
            initialization="svd",
            seed=data.seed,
        ),
    )


def plot_roc(seed: int) -> Path:
    """对应论文Fig.5：DoS->R2L中No-TL与HeTL的ROC曲线。"""
    data = load_task("dos_to_r2l", seed)
    representation = fitted_hetl(data, k=10)
    indices = data.test_indices
    y_test = data.y_target[indices]

    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    styles = {
        "LinearSVM": ("#4C78A8", "-"),
        "NaiveBayes": ("#F58518", "-"),
        "KNN": ("#54A24B", "-"),
    }
    for classifier_name in ("LinearSVM", "NaiveBayes", "KNN"):
        color, _ = styles[classifier_name]

        no_tl = build_classifiers(seed)[classifier_name]
        no_tl.fit(data.X_source, data.y_source)
        no_tl_scores = positive_class_scores(no_tl, data.X_target[indices])
        fpr, tpr, _ = roc_curve(y_test, no_tl_scores)
        axis.plot(
            fpr,
            tpr,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{classifier_name} No-TL (AUC={auc(fpr, tpr):.2f})",
        )

        hetl_model = build_classifiers(seed)[classifier_name]
        hetl_model.fit(representation.V_source, data.y_source)
        hetl_scores = positive_class_scores(
            hetl_model, representation.V_target[indices]
        )
        fpr, tpr, _ = roc_curve(y_test, hetl_scores)
        axis.plot(
            fpr,
            tpr,
            color=color,
            linestyle="-",
            linewidth=2.2,
            label=f"{classifier_name} HeTL (AUC={auc(fpr, tpr):.2f})",
        )

    axis.plot([0, 1], [0, 1], color="black", linestyle=":", label="Random (AUC=0.50)")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC on DoS → R2L (seed 42, fixed HeTL parameters)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "dos_to_r2l_roc.png")


def plot_latent_space(seed: int) -> Path:
    """对应论文Fig.7：用k=2直接展示Probe->R2L的共同潜空间。"""
    data = load_task("probe_to_r2l", seed)
    representation = fitted_hetl(data, k=2)
    fig, axis = plt.subplots(figsize=(7.2, 6.2))

    groups = [
        (representation.V_source, data.y_source == 0, "Normal, source", "normal_source", "o"),
        (representation.V_source, data.y_source == 1, "Probe, source", "attack_source", "^"),
        (representation.V_target, data.y_target == 0, "Normal, target", "normal_target", "s"),
        (representation.V_target, data.y_target == 1, "R2L, target", "attack_target", "x"),
    ]
    for values, mask, label, color_name, marker in groups:
        axis.scatter(
            values[mask, 0],
            values[mask, 1],
            s=18,
            alpha=0.45,
            label=label,
            color=COLORS[color_name],
            marker=marker,
            linewidths=0.6,
        )

    axis.set(
        xlabel="Latent dimension 1",
        ylabel="Latent dimension 2",
        title="HeTL projected data: Probe → R2L",
    )
    axis.legend(markerscale=1.5)
    fig.tight_layout()
    return save_figure(fig, "probe_to_r2l_latent.png")


def plot_baseline_summary() -> Path:
    """将Table II/III扩展为10随机种子的Accuracy/F1均值±标准差图。"""
    path = TABLE_DIR / "baseline_summary.csv"
    if not path.is_file():
        raise FileNotFoundError("请先运行 uv run python -m src.baselines --all-seeds")
    frame = pd.read_csv(path)

    task_titles = {
        "dos_to_r2l": "DoS → R2L",
        "dos_to_probe": "DoS → Probe",
        "probe_to_r2l": "Probe → R2L",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.36
    for axis, (task, title) in zip(axes, task_titles.items(), strict=True):
        subset = frame.set_index(["task", "classifier"]).loc[task].reindex(CLASSIFIER_ORDER)
        axis.bar(
            x - width / 2,
            subset["accuracy_mean"],
            width,
            yerr=subset["accuracy_std"],
            capsize=3,
            label="Accuracy",
            color="#4C78A8",
        )
        axis.bar(
            x + width / 2,
            subset["f1_mean"],
            width,
            yerr=subset["f1_std"],
            capsize=3,
            label="F1",
            color="#F58518",
        )
        axis.set_title(title)
        axis.set_xticks(x, CLASSIFIER_ORDER, rotation=35, ha="right")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Score (mean ± std over 10 seeds)")
    axes[0].legend()
    fig.suptitle("No-Transfer-Learning Baselines", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "baseline_summary.png")


def plot_baseline_vs_hetl(seed: int) -> Path:
    """固定参数烟雾实验的F1对比；不冒充正式调参后的论文结果。"""
    baseline_path = TABLE_DIR / "baseline_results.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["seed"] == seed]
    tasks = ["dos_to_r2l", "dos_to_probe", "probe_to_r2l"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.36
    for axis, task in zip(axes, tasks, strict=True):
        smoke_path = TABLE_DIR / f"hetl_smoke_{task}_seed_{seed}.csv"
        if not smoke_path.is_file():
            raise FileNotFoundError(
                f"缺少 {smoke_path.name}；请先运行该任务的 run_hetl 固定参数测试"
            )
        no_tl = baseline[baseline["task"] == task].set_index("classifier").reindex(CLASSIFIER_ORDER)
        hetl = pd.read_csv(smoke_path).set_index("classifier").reindex(CLASSIFIER_ORDER)
        axis.bar(x - width / 2, no_tl["f1"], width, label="No-TL", color="#9D9D9D")
        axis.bar(x + width / 2, hetl["f1"], width, label="HeTL fixed", color="#54A24B")
        axis.set_title(task.replace("_to_", " → ").replace("_", " ").title())
        axis.set_xticks(x, CLASSIFIER_ORDER, rotation=35, ha="right")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Attack F1")
    axes[0].legend()
    fig.suptitle(
        "No-TL vs HeTL (seed 42; β=1, k=10; not tuned)", fontweight="bold"
    )
    fig.tight_layout()
    return save_figure(fig, "baseline_vs_hetl_f1.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["all", "features", "roc", "latent", "baseline", "comparison"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    actions = {
        "features": lambda: plot_feature_distribution(args.seed),
        "roc": lambda: plot_roc(args.seed),
        "latent": lambda: plot_latent_space(args.seed),
        "baseline": plot_baseline_summary,
        "comparison": lambda: plot_baseline_vs_hetl(args.seed),
    }
    if args.figure == "all":
        for action in actions.values():
            action()
    else:
        actions[args.figure]()


if __name__ == "__main__":
    main()
