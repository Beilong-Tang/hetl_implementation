"""Summarize deep-learning runs and create comparable result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULTS = TABLE_DIR / "deep_learning_results.csv"
PARENT_TABLE_DIR = ROOT.parent / "first_experiment_review" / "outputs" / "tables"

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TASK_LABELS = {
    "dos_to_r2l": "DoS → R2L",
    "dos_to_probe": "DoS → Probe",
    "probe_to_r2l": "Probe → R2L",
}


def summarize_deep_learning(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["task", "model"], as_index=False)
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = grouped["seed"].nunique()["seed"]
    summary.insert(0, "source", "deep_learning")
    summary.insert(1, "approach", "DeepLearning")
    summary = summary.rename(columns={"model": "classifier"})
    return summary


def load_reference_summary(filename: str, approach: str) -> pd.DataFrame:
    path = PARENT_TABLE_DIR / filename
    reference = pd.read_csv(path)
    reference = reference.rename(columns={"classifier": "classifier"})
    reference["source"] = "first_experiment"
    reference["approach"] = approach
    reference["n_seeds"] = 10
    return reference[[
        "source", "approach", "task", "classifier", "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std",
        "roc_auc_mean", "roc_auc_std", "n_seeds",
    ]]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS)
    deep_summary = summarize_deep_learning(df)
    baseline_summary = load_reference_summary("baseline_summary.csv", "No TL")
    hetl_summary = load_reference_summary("hetl_summary.csv", "HeTL")
    combined_summary = pd.concat([baseline_summary, hetl_summary, deep_summary], ignore_index=True)
    combined_summary.to_csv(TABLE_DIR / "deep_learning_summary.csv", index=False, float_format="%.6f")
    deep_summary.to_csv(TABLE_DIR / "deep_learning_only_summary.csv", index=False, float_format="%.6f")

    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "tabular_resnet": "#2563eb",
        "cnn1d": "#059669",
        "gru": "#d97706",
        "tabular_transformer": "#7c3aed",
    }
    plot_df = df.copy()
    plot_df["task_label"] = plot_df["task"].map(TASK_LABELS)

    for metric, title in [("f1", "Deep learning attack F1"), ("accuracy", "Deep learning accuracy")]:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        sns.barplot(
            data=plot_df,
            x="task_label",
            y=metric,
            hue="model",
            errorbar="sd",
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Transfer task")
        ax.set_ylabel(metric.upper() if metric == "f1" else "Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"deep_learning_{metric}_comparison.png", dpi=180)
        plt.close(fig)

    heat = deep_summary.pivot(index="classifier", columns="task", values="f1_mean")
    heat = heat.rename(columns=TASK_LABELS)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(heat, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="YlGnBu", cbar_kws={"label": "Mean F1"}, ax=ax)
    ax.set_title("Deep learning F1 mean by task and model")
    ax.set_xlabel("Transfer task")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "deep_learning_f1_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (task, task_df) in zip(axes, plot_df.groupby("task", sort=False)):
        sns.boxplot(data=task_df, x="model", y="f1", hue="model", palette=palette, legend=False, ax=ax)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("")
        ax.set_ylabel("F1" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, 1)
    fig.suptitle("F1 stability across random seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "deep_learning_f1_seed_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compare the newly added methods with the unchanged No-TL and HeTL references.
    compare = combined_summary.copy()
    compare["task_label"] = compare["task"].map(TASK_LABELS)
    compare = compare[compare["approach"].isin(["No TL", "HeTL", "DeepLearning"])]
    compare["model_label"] = compare["approach"] + ": " + compare["classifier"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (task, task_df) in zip(axes, compare.groupby("task", sort=False)):
        task_df = task_df.sort_values("f1_mean", ascending=False)
        sns.barplot(data=task_df, x="f1_mean", y="model_label", hue="approach", dodge=False, ax=ax)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Mean attack F1")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Approach", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Traditional, HeTL, and deep-learning comparison", y=1.08)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "all_methods_f1_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {TABLE_DIR / 'deep_learning_summary.csv'}")
    print(f"saved {TABLE_DIR / 'deep_learning_only_summary.csv'}")
    print(f"saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
