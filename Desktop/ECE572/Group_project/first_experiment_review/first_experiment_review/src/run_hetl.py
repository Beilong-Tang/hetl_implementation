"""训练、选择并评价论文的 HeTL 表示。

快速连通性实验（固定参数，不做调参）：
    uv run python -m src.run_hetl --task dos_to_r2l --seed 42 --steps 100

正式 beta/k 搜索：
    uv run python -m src.run_hetl --task dos_to_r2l --seed 42 --tune

三个任务、全部随机种子的正式实验（可中断后续跑）：
    uv run python -m src.run_hetl --task all --all-seeds --tune --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .baselines import build_classifiers, positive_class_scores
from .data_loader import ROOT, VALID_TASKS, TransferTaskData, load_task
from .hetl import HeTLConfig, HeTLResult, fit_hetl

CONFIG_PATH = ROOT / "configs" / "experiment.json"
OUTPUT_DIR = ROOT / "outputs" / "tables"
LOG_DIR = ROOT / "experiments" / "logs"


def classification_metrics(
    y_true: np.ndarray, prediction: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    """计算与 baseline 完全一致的指标定义。"""
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, prediction, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_representation(
    task_data: TransferTaskData,
    representation: HeTLResult,
    classifier_name: str,
    partition: str,
    classifier_factory=build_classifiers,
) -> dict[str, object]:
    """在 Vs 上训练分类器，并在指定 Vt 分区评价。"""
    if partition == "validation":
        indices = task_data.validation_indices
    elif partition == "test":
        indices = task_data.test_indices
    else:
        raise ValueError("partition 必须是 validation 或 test")

    # 每次评价都创建新分类器，防止不同候选参数之间共享已拟合状态。
    model = classifier_factory(task_data.seed)[classifier_name]
    model.fit(representation.V_source, task_data.y_source)
    prediction = model.predict(representation.V_target[indices])
    scores = positive_class_scores(model, representation.V_target[indices])
    return classification_metrics(task_data.y_target[indices], prediction, scores)


def save_loss_history(
    task: str,
    seed: int,
    beta: float,
    k: int,
    result: HeTLResult,
    log_dir: Path | None = None,
) -> Path:
    """保存损失曲线原始数据，便于确认是否收敛。"""
    directory = LOG_DIR if log_dir is None else log_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"hetl_{task}_seed_{seed}_beta_{beta:g}_k_{k}.csv"
    pd.DataFrame(result.history).to_csv(path, index=False)
    return path


def fit_candidate(
    task_data: TransferTaskData,
    beta: float,
    k: int,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
    patience: int,
    initialization: str,
) -> HeTLResult:
    config = HeTLConfig(
        latent_dimension=k,
        beta=beta,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tolerance=tolerance,
        patience=patience,
        initialization=initialization,
        seed=task_data.seed,
    )
    return fit_hetl(task_data.X_source, task_data.X_target, config)


def run_fixed(args: argparse.Namespace, settings: dict) -> pd.DataFrame:
    """固定参数快速运行；用于检查管线，不作为正式调参结果。"""
    task_data = load_task(args.task, args.seed)
    result = fit_candidate(
        task_data,
        beta=args.beta,
        k=args.k,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        tolerance=settings["tolerance"],
        patience=settings["patience"],
        initialization=settings["initialization"],
    )
    save_loss_history(args.task, args.seed, args.beta, args.k, result)

    rows = []
    for classifier_name in build_classifiers(args.seed):
        metrics = evaluate_representation(task_data, result, classifier_name, partition="test")
        row = {
            "task": args.task,
            "seed": args.seed,
            "approach": "HeTL fixed smoke test",
            "classifier": classifier_name,
            "beta": args.beta,
            "latent_dimension": args.k,
            "learning_rate": args.learning_rate,
            "steps_run": result.steps_run,
            "converged": result.converged,
            "final_loss": result.history[-1]["loss"],
            **metrics,
        }
        rows.append(row)
        print(
            f"{classifier_name:12s} accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}"
        )
    return pd.DataFrame(rows)


def run_tuning(
    args: argparse.Namespace,
    settings: dict,
    *,
    task_data: TransferTaskData | None = None,
    classifier_factory=build_classifiers,
    log_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用500条目标验证标签为每种分类器选择 beta 和 k，再评价测试集。"""
    if task_data is None:
        task_data = load_task(args.task, args.seed)
    classifier_names = list(classifier_factory(args.seed))
    best: dict[str, tuple[tuple[float, ...], dict[str, object], HeTLResult]] = {}
    search_rows: list[dict[str, object]] = []

    for beta in settings["beta_grid"]:
        for k in settings["latent_dimensions"]:
            print(f"拟合 beta={beta:g}, k={k} ...")
            result = fit_candidate(
                task_data,
                beta=beta,
                k=k,
                learning_rate=args.learning_rate,
                max_steps=args.steps,
                tolerance=settings["tolerance"],
                patience=settings["patience"],
                initialization=settings["initialization"],
            )
            save_loss_history(args.task, args.seed, beta, k, result, log_dir=log_dir)

            for classifier_name in classifier_names:
                metrics = evaluate_representation(
                    task_data,
                    result,
                    classifier_name,
                    partition="validation",
                    classifier_factory=classifier_factory,
                )
                row = {
                    "task": args.task,
                    "seed": args.seed,
                    "classifier": classifier_name,
                    "beta": beta,
                    "latent_dimension": k,
                    "learning_rate": args.learning_rate,
                    "steps_run": result.steps_run,
                    "converged": result.converged,
                    "final_loss": result.history[-1]["loss"],
                    **{f"validation_{name}": value for name, value in metrics.items()},
                }
                search_rows.append(row)

                # 论文未说明选择指标。本项目优先最大化 attack F1；相同时依次看
                # Accuracy、AUC，再偏好更小k和更小beta，规则记录在文档中。
                rank = (
                    float(metrics["f1"]),
                    float(metrics["accuracy"]),
                    float(metrics["roc_auc"]),
                    -float(k),
                    -float(beta),
                )
                if classifier_name not in best or rank > best[classifier_name][0]:
                    best[classifier_name] = (rank, row, result)

    test_rows: list[dict[str, object]] = []
    for classifier_name in classifier_names:
        _, selected, representation = best[classifier_name]
        metrics = evaluate_representation(
            task_data,
            representation,
            classifier_name,
            partition="test",
            classifier_factory=classifier_factory,
        )
        test_rows.append(
            {
                "task": args.task,
                "seed": args.seed,
                "approach": "HeTL",
                "classifier": classifier_name,
                "beta": selected["beta"],
                "latent_dimension": selected["latent_dimension"],
                "learning_rate": args.learning_rate,
                "selection_metric": "target_validation_f1",
                "validation_f1": selected["validation_f1"],
                "validation_accuracy": selected["validation_accuracy"],
                "steps_run": selected["steps_run"],
                "converged": selected["converged"],
                "final_loss": selected["final_loss"],
                **metrics,
            }
        )
        print(
            f"选择 {classifier_name:12s}: beta={selected['beta']:g}, "
            f"k={selected['latent_dimension']}, validation_f1={selected['validation_f1']:.4f}, "
            f"test_f1={metrics['f1']:.4f}"
        )
    return pd.DataFrame(search_rows), pd.DataFrame(test_rows)


def tuning_paths(task: str, seed: int) -> tuple[Path, Path]:
    """返回单个任务/种子的搜索明细与测试结果路径。"""
    return (
        OUTPUT_DIR / f"hetl_search_{task}_seed_{seed}.csv",
        OUTPUT_DIR / f"hetl_results_{task}_seed_{seed}.csv",
    )


def save_tuning_outputs(
    task: str,
    seed: int,
    search: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[Path, Path]:
    """立即保存一个实验单元，避免长时间批量实验中断后丢失进度。"""
    search_path, result_path = tuning_paths(task, seed)
    search.to_csv(search_path, index=False)
    results.to_csv(result_path, index=False)
    return search_path, result_path


def save_aggregate_results(result_frames: list[pd.DataFrame]) -> tuple[Path, Path, Path]:
    """合并逐次结果，并生成跨随机种子的均值、标准差和参数选择次数。"""
    results = pd.concat(result_frames, ignore_index=True)
    results_path = OUTPUT_DIR / "hetl_results.csv"
    summary_path = OUTPUT_DIR / "hetl_summary.csv"
    parameter_path = OUTPUT_DIR / "hetl_selected_parameters.csv"
    results.to_csv(results_path, index=False)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = (
        results.groupby(["task", "approach", "classifier"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(summary_path, index=False)

    selected_parameters = (
        results.groupby(["task", "classifier", "beta", "latent_dimension"], sort=False)
        .size()
        .rename("selection_count")
        .reset_index()
    )
    selected_parameters.to_csv(parameter_path, index=False)
    return results_path, summary_path, parameter_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *VALID_TASKS], default="dos_to_r2l")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="使用配置文件中的全部随机种子",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若单个任务/种子的搜索和结果文件都已存在，则跳过该实验单元",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    settings = config["hetl"]
    if args.steps is None:
        args.steps = settings["max_steps"] if args.tune else 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = list(VALID_TASKS) if args.task == "all" else [args.task]
    seeds = config["random_seeds"] if args.all_seeds else [args.seed]

    if not args.tune:
        for task in tasks:
            for seed in seeds:
                run_args = argparse.Namespace(**{**vars(args), "task": task, "seed": seed})
                results = run_fixed(run_args, settings)
                result_path = OUTPUT_DIR / f"hetl_smoke_{task}_seed_{seed}.csv"
                results.to_csv(result_path, index=False)
                print(f"\n固定参数连通性结果：{result_path}")
        return

    result_frames: list[pd.DataFrame] = []
    for task in tasks:
        for seed in seeds:
            search_path, result_path = tuning_paths(task, seed)
            if args.resume and search_path.exists() and result_path.exists():
                print(f"\n跳过已完成实验：task={task}, seed={seed}")
                result_frames.append(pd.read_csv(result_path))
                continue

            print(f"\n正式搜索：task={task}, seed={seed}")
            run_args = argparse.Namespace(**{**vars(args), "task": task, "seed": seed})
            search, results = run_tuning(run_args, settings)
            save_tuning_outputs(task, seed, search, results)
            result_frames.append(results)
            print(f"验证搜索：{search_path}\n测试结果：{result_path}")

    results_path, summary_path, parameter_path = save_aggregate_results(result_frames)
    print(
        f"\n全部逐次结果：{results_path}"
        f"\n均值与标准差：{summary_path}"
        f"\n参数选择次数：{parameter_path}"
    )


if __name__ == "__main__":
    main()
