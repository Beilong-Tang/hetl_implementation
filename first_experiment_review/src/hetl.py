"""论文 HeTL（Heterogeneous Transfer Learning）核心实现。

论文目标函数（式4）：

    ||S - Vs Ps||_F^2 + ||T - Vt Pt||_F^2 + beta ||Vs - Vt||_F^2

约束：

    Vs.T @ Vs = I,  Vt.T @ Vt = I

矩阵形状：

    S  : (l, m)   源域输入
    T  : (l, n)   目标域输入
    Vs : (l, k)   源域潜表示
    Vt : (l, k)   目标域潜表示
    Ps : (k, m)   源域重构矩阵
    Pt : (k, n)   目标域重构矩阵

论文没有公开初始化、正交约束的数值实现和收敛容差。本实现的补充选择在
docs/hetl_implementation_assumptions.md 中逐项说明。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class HeTLConfig:
    """HeTL训练参数。"""

    latent_dimension: int = 10
    beta: float = 1.0
    learning_rate: float = 1e-3
    max_steps: int = 1_000
    tolerance: float = 1e-7
    patience: int = 10
    initialization: str = "svd"
    seed: int = 42


@dataclass(frozen=True)
class HeTLResult:
    """训练完成后的表示、投影矩阵和收敛记录。"""

    V_source: NDArray[np.float64]
    V_target: NDArray[np.float64]
    P_source: NDArray[np.float64]
    P_target: NDArray[np.float64]
    history: tuple[dict[str, float], ...]
    steps_run: int
    converged: bool


def _canonical_qr(matrix: torch.Tensor) -> torch.Tensor:
    """QR正交化，并固定符号以提高跨平台可重复性。"""
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def _initial_representation(data: torch.Tensor, k: int, method: str, seed: int) -> torch.Tensor:
    """生成满足 V.T @ V = I 的初始潜表示。"""
    if method == "svd":
        # 截断SVD给出单独重构每个域时最自然的低秩起点。论文只写了
        # Initialize 而没有说明方法，因此这是稳定、确定性的复现假设。
        u, _, _ = torch.linalg.svd(data, full_matrices=False)
        return _canonical_qr(u[:, :k])
    if method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_matrix = torch.randn(
            data.shape[0], k, generator=generator, dtype=data.dtype, device=data.device
        )
        return _canonical_qr(random_matrix)
    raise ValueError("initialization 必须是 'svd' 或 'random'")


def _closed_form_projection(V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """论文式(7)(8)：固定V时求最小二乘最优P。

    一般形式为 (V.T V)^-1 V.T X。由于每轮都正交化，V.T V=I，故可稳定地
    简化为 V.T X，避免显式矩阵求逆。
    """
    return V.T @ X


def _loss_components(
    source: torch.Tensor,
    target: torch.Tensor,
    V_source: torch.Tensor,
    V_target: torch.Tensor,
    P_source: torch.Tensor,
    P_target: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_reconstruction = torch.sum((source - V_source @ P_source) ** 2)
    target_reconstruction = torch.sum((target - V_target @ P_target) ** 2)
    alignment = torch.sum((V_source - V_target) ** 2)
    total = source_reconstruction + target_reconstruction + beta * alignment
    return total, source_reconstruction, target_reconstruction, alignment


def fit_hetl(
    X_source: NDArray[np.floating],
    X_target: NDArray[np.floating],
    config: HeTLConfig,
) -> HeTLResult:
    """拟合HeTL并返回源域与目标域的共同k维表示。

    重要：论文的 ||Vs-Vt|| 项要求两个域具有相同行数，并把相同行号视为一对。
    NSL-KDD 的源/目标记录没有天然一一对应关系。为避免使用目标标签泄漏，本实现
    不按类别重新排列，而是严格使用 preprocessing.py 保存的现有行顺序。
    """
    source_np = np.asarray(X_source, dtype=np.float64)
    target_np = np.asarray(X_target, dtype=np.float64)
    if source_np.ndim != 2 or target_np.ndim != 2:
        raise ValueError("X_source 和 X_target 必须是二维矩阵")
    if source_np.shape[0] != target_np.shape[0]:
        raise ValueError("论文的逐行对齐项要求源域和目标域样本数相同")
    if not np.isfinite(source_np).all() or not np.isfinite(target_np).all():
        raise ValueError("输入包含 NaN 或无穷值")
    if config.beta < 0:
        raise ValueError("beta 必须非负")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须为正")
    if config.max_steps < 1 or config.patience < 1:
        raise ValueError("max_steps 和 patience 必须为正整数")

    max_rank = min(
        source_np.shape[0], source_np.shape[1], target_np.shape[1]
    )
    if not 1 <= config.latent_dimension <= max_rank:
        raise ValueError(f"latent_dimension 必须在 1 到 {max_rank} 之间")

    # 使用float64与NumPy预处理数据保持一致；本数据规模较小，CPU更稳定。
    source = torch.as_tensor(source_np, dtype=torch.float64, device="cpu")
    target = torch.as_tensor(target_np, dtype=torch.float64, device="cpu")
    V_source = _initial_representation(
        source, config.latent_dimension, config.initialization, config.seed
    )
    V_target = _initial_representation(
        target, config.latent_dimension, config.initialization, config.seed + 1
    )

    history: list[dict[str, float]] = []
    previous_loss: float | None = None
    stable_steps = 0
    converged = False

    for step in range(config.max_steps):
        # 式(7)(8)：固定V，闭式更新P。
        P_source = _closed_form_projection(V_source, source).detach()
        P_target = _closed_form_projection(V_target, target).detach()

        # 式(5)(6)：固定P，对Vs和Vt同时做一步梯度下降。
        V_source = V_source.detach().requires_grad_(True)
        V_target = V_target.detach().requires_grad_(True)
        total, _, _, _ = _loss_components(
            source,
            target,
            V_source,
            V_target,
            P_source,
            P_target,
            config.beta,
        )
        gradient_source, gradient_target = torch.autograd.grad(
            total, (V_source, V_target)
        )

        with torch.no_grad():
            V_source = _canonical_qr(
                V_source - config.learning_rate * gradient_source
            )
            V_target = _canonical_qr(
                V_target - config.learning_rate * gradient_target
            )

            # 更新V后重新计算最优P，再记录论文原始（未按元素平均）目标函数。
            P_source = _closed_form_projection(V_source, source)
            P_target = _closed_form_projection(V_target, target)
            total, source_loss, target_loss, alignment = _loss_components(
                source,
                target,
                V_source,
                V_target,
                P_source,
                P_target,
                config.beta,
            )
            loss_value = float(total)
            history.append(
                {
                    "step": float(step + 1),
                    "loss": loss_value,
                    "source_reconstruction": float(source_loss),
                    "target_reconstruction": float(target_loss),
                    "alignment": float(alignment),
                }
            )

        if previous_loss is not None:
            relative_change = abs(previous_loss - loss_value) / max(
                abs(previous_loss), 1e-12
            )
            stable_steps = stable_steps + 1 if relative_change < config.tolerance else 0
            if stable_steps >= config.patience:
                converged = True
                break
        previous_loss = loss_value

    # 输出前再次验证论文的正交约束。
    identity = torch.eye(config.latent_dimension, dtype=torch.float64)
    if not torch.allclose(V_source.T @ V_source, identity, atol=1e-8, rtol=1e-8):
        raise RuntimeError("V_source 未满足正交约束")
    if not torch.allclose(V_target.T @ V_target, identity, atol=1e-8, rtol=1e-8):
        raise RuntimeError("V_target 未满足正交约束")

    return HeTLResult(
        V_source=V_source.detach().numpy().copy(),
        V_target=V_target.detach().numpy().copy(),
        P_source=P_source.detach().numpy().copy(),
        P_target=P_target.detach().numpy().copy(),
        history=tuple(history),
        steps_run=len(history),
        converged=converged,
    )
