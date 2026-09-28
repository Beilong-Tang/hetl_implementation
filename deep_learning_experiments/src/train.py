"""Train small deep-learning baselines on the existing NSL-KDD task splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parent / "first_experiment_review"
sys.path.insert(0, str(PARENT))
from src.data_loader import load_task  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x + self.net(x))


class TabularResNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, blocks: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_layer = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(hidden, dropout) for _ in range(blocks)))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.blocks(self.in_layer(x))).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.unsqueeze(1)).squeeze(-1)


class SequenceModel(nn.Module):
    def __init__(self, kind: str, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, hidden)
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.rnn(self.embed(x.unsqueeze(-1)))
        return self.out(h[:, -1]).squeeze(-1)


class TabularTransformer(nn.Module):
    def __init__(self, n_features: int, dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.encoder(self.embed(x.unsqueeze(-1))).mean(dim=1)).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    if name == "tabular_resnet":
        return TabularResNet(n_features)
    if name == "cnn1d":
        return CNN1D(n_features)
    if name == "gru":
        return SequenceModel("gru", n_features)
    if name == "lstm":
        return SequenceModel("lstm", n_features)
    if name == "tabular_transformer":
        return TabularTransformer(n_features)
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return (scores >= 0.5).astype(np.int8), scores


def metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
    }


def run_one(task: str, seed: int, model_name: str, epochs: int, patience: int, batch_size: int) -> dict[str, object]:
    seed_everything(seed)
    data = load_task(task, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, data.X_source.shape[1]).to(device)
    counts = np.bincount(data.y_source.astype(int), minlength=2)
    weights = torch.tensor([len(data.y_source) / (2 * max(c, 1)) for c in counts], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    x = torch.as_tensor(data.X_source, dtype=torch.float32)
    y = torch.as_tensor(data.y_source, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(torch.stack([-logits, logits], dim=1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        vp, vs = predict(model, data.X_target[data.validation_indices], device)
        val_f1 = f1_score(data.y_target[data.validation_indices], vp, zero_division=0)
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, scores = predict(model, data.X_target[data.test_indices], device)
    result = metrics(data.y_target[data.test_indices], pred, scores)
    return {"task": task, "seed": seed, "model": model_name, "device": str(device), "epochs": epoch, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["dos_to_r2l", "dos_to_probe", "probe_to_r2l"])
    parser.add_argument("--models", nargs="+", default=["tabular_resnet", "cnn1d", "gru", "tabular_transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for task in args.tasks:
        for model_name in args.models:
            for seed in args.seeds:
                row = run_one(task, seed, model_name, args.epochs, args.patience, args.batch_size)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    out = HERE / "outputs" / "tables" / "deep_learning_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
