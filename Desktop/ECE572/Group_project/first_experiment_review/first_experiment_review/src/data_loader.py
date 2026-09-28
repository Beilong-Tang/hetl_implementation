"""处理后迁移任务的稳定读取接口。

其他组员不需要了解 preprocessing.py 的内部实现，只需通过 ``load_task``
获取源域、目标域、验证/测试索引和特征名称。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
VALID_TASKS = ("dos_to_r2l", "dos_to_probe", "probe_to_r2l")


@dataclass(frozen=True)
class TransferTaskData:
    """一个 source -> target 实验所需的全部数组。"""

    task: str
    seed: int
    X_source: NDArray[np.float64]
    y_source: NDArray[np.int8]
    X_target: NDArray[np.float64]
    y_target: NDArray[np.int8]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    feature_names: NDArray[np.str_]

    def validate(self) -> None:
        """在训练前检查接口契约，尽早发现错位和标签泄漏风险。"""
        if self.X_source.ndim != 2 or self.X_target.ndim != 2:
            raise ValueError("X_source 和 X_target 必须是二维矩阵")
        if self.y_source.ndim != 1 or self.y_target.ndim != 1:
            raise ValueError("y_source 和 y_target 必须是一维向量")
        if self.X_source.shape[0] != len(self.y_source):
            raise ValueError("X_source 的行数与 y_source 长度不一致")
        if self.X_target.shape[0] != len(self.y_target):
            raise ValueError("X_target 的行数与 y_target 长度不一致")
        if self.X_source.shape[1] != self.X_target.shape[1]:
            raise ValueError("No-TL baseline 要求同一任务的源域和目标域列数一致")
        if self.X_source.shape[1] != len(self.feature_names):
            raise ValueError("feature_names 数量与特征列数不一致")
        if set(np.unique(self.y_source)) != {0, 1}:
            raise ValueError("源域标签必须同时包含 normal=0 和 attack=1")
        if set(np.unique(self.y_target)) != {0, 1}:
            raise ValueError("目标域标签必须同时包含 normal=0 和 attack=1")

        validation = set(self.validation_indices.tolist())
        test = set(self.test_indices.tolist())
        if validation & test:
            raise ValueError("目标验证集和测试集索引发生重叠")
        if validation | test != set(range(len(self.y_target))):
            raise ValueError("目标验证集和测试集没有完整覆盖目标域")
        if not np.isfinite(self.X_source).all() or not np.isfinite(self.X_target).all():
            raise ValueError("特征矩阵包含 NaN 或无穷值")


def task_path(task: str, seed: int) -> Path:
    """返回任务文件路径，并拒绝拼写错误的任务名。"""
    if task not in VALID_TASKS:
        raise ValueError(f"未知任务 {task!r}；可选值为 {VALID_TASKS}")
    return PROCESSED_DIR / f"{task}_seed_{seed}.npz"


def load_task(task: str, seed: int = 42) -> TransferTaskData:
    """加载并验证一个预处理后的迁移任务。"""
    path = task_path(task, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 {path}。请先运行：uv run python src/preprocessing.py"
        )

    # np.array(..., copy=True) 让数组脱离 NPZ 文件句柄，退出 with 后仍可安全使用。
    with np.load(path, allow_pickle=False) as archive:
        data = TransferTaskData(
            task=task,
            seed=seed,
            X_source=np.array(archive["X_source"], dtype=np.float64, copy=True),
            y_source=np.array(archive["y_source"], dtype=np.int8, copy=True),
            X_target=np.array(archive["X_target"], dtype=np.float64, copy=True),
            y_target=np.array(archive["y_target"], dtype=np.int8, copy=True),
            validation_indices=np.array(
                archive["target_validation_indices"], dtype=np.int64, copy=True
            ),
            test_indices=np.array(
                archive["target_test_indices"], dtype=np.int64, copy=True
            ),
            feature_names=np.array(archive["feature_names"], dtype=str, copy=True),
        )
    data.validate()
    return data
