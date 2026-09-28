"""运行论文 Table II/III 对应的 No-Transfer-Learning baselines。

实验逻辑：
    1. 只使用 (X_source, y_source) 拟合分类器；
    2. 不执行 HeTL，不使用任何目标域标签训练分类器；
    3. 只在 target_test_indices 指向的最终测试分区报告指标；
    4. target_validation_indices 留给后续 HeTL 选择 beta 和 k。

论文只公布了分类器种类，没有公布具体超参数。本文件显式给出一套可复现的
标准参数，并将参数写入结果 CSV。它们属于复现实验假设，不应声称是作者原参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data_loader import ROOT, VALID_TASKS, TransferTaskData, load_task

CONFIG_PATH = ROOT / "configs" / "experiment.json"
OUTPUT_DIR = ROOT / "outputs" / "tables"


def build_classifiers(seed: int) -> dict[str, ClassifierMixin]:
    """创建论文采用的五种基础分类器；每个任务必须调用一次以获得新模型。"""
    return {
        "CART": DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "LinearSVM": LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        ),
        "NaiveBayes": GaussianNB(var_smoothing=1e-9),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
        ),
    }


def positive_class_scores(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """取得 attack=1 的连续分数，用于 ROC-AUC，而不是错误地用0/1预测算AUC。"""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        positive_column = int(np.flatnonzero(classes == 1)[0])
        return probabilities[:, positive_column]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    raise TypeError(f"{type(model).__name__} 不提供概率或 decision_function")


def explicit_parameters(model: ClassifierMixin) -> str:
    """以稳定JSON形式记录所有实际分类器参数。"""
    return json.dumps(model.get_params(deep=False), sort_keys=True, default=str)


def evaluate_classifier(
    task_data: TransferTaskData,
    classifier_name: str,
    model: ClassifierMixin,
) -> dict[str, object]:
    """在源域拟合，并仅对目标测试分区计算指标。"""
    X_test = task_data.X_target[task_data.test_indices]
    y_test = task_data.y_target[task_data.test_indices]

    started = perf_counter()
    model.fit(task_data.X_source, task_data.y_source)
    fit_seconds = perf_counter() - started

    prediction = model.predict(X_test)
    scores = positive_class_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()

    return {
        "task": task_data.task,
        "seed": task_data.seed,
        "approach": "No TL",
        "classifier": classifier_name,
        "source_rows": len(task_data.y_source),
        "target_test_rows": len(y_test),
        "n_features": task_data.X_source.shape[1],
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, prediction, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fit_seconds": fit_seconds,
        "parameters": explicit_parameters(model),
    }


def run_baselines(tasks: list[str], seeds: list[int]) -> pd.DataFrame:
    """运行指定任务和随机种子，并返回逐次实验结果。"""
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for task in tasks:
            task_data = load_task(task, seed)
            print(
                f"\n{task} seed={seed}: "
                f"source={task_data.X_source.shape}, target={task_data.X_target.shape}"
            )
            for classifier_name, model in build_classifiers(seed).items():
                result = evaluate_classifier(task_data, classifier_name, model)
                rows.append(result)
                print(
                    f"  {classifier_name:12s} "
                    f"accuracy={result['accuracy']:.4f} "
                    f"f1={result['f1']:.4f} auc={result['roc_auc']:.4f}"
                )
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame) -> tuple[Path, Path]:
    """保存逐次结果，并按 task/classifier 汇总均值和标准差。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "baseline_results.csv"
    summary_path = OUTPUT_DIR / "baseline_summary.csv"
    results.to_csv(result_path, index=False)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "fit_seconds"]
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
    return result_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["all", *VALID_TASKS],
        default="all",
        help="默认运行三个任务",
    )
    parser.add_argument("--seed", type=int, default=42, help="默认随机种子为42")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="使用 configs/experiment.json 中的全部随机种子",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    tasks = list(VALID_TASKS) if args.task == "all" else [args.task]
    seeds = config["random_seeds"] if args.all_seeds else [args.seed]
    results = run_baselines(tasks, seeds)
    result_path, summary_path = save_results(results)
    print(f"\n逐次结果：{result_path}")
    print(f"汇总结果：{summary_path}")


if __name__ == "__main__":
    main()
