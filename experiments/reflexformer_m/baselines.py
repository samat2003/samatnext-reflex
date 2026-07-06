#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classical baselines for ReflexFormer-M synthetic telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import CLASS_NAMES, ReflexFormerMClassifier, count_parameters
from experiments.reflexformer_m.train import classification_report, fit_normalizer, load_split, prepare_features, resolve_device


def summary_features(x: torch.Tensor) -> np.ndarray:
    arr = x.numpy()
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    min_v = arr.min(axis=1)
    max_v = arr.max(axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    return np.concatenate([mean, std, min_v, max_v, q25, q75], axis=1).astype(np.float32)


def flat_features(x: torch.Tensor, max_examples: int | None = None) -> np.ndarray:
    arr = x.numpy()
    if max_examples is not None:
        arr = arr[:max_examples]
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def report_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, seconds: float) -> dict:
    metrics = classification_report(torch.from_numpy(y_true), torch.from_numpy(y_pred), len(CLASS_NAMES))
    failsafe_recall = metrics["per_class"][CLASS_NAMES.index("FAILSAFE_NEEDED")]["recall"]
    print(
        f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"failsafe_recall={failsafe_recall:.4f} max_pred_share={metrics['max_predicted_class_share']:.4f} "
        f"train_eval_seconds={seconds:.2f}"
    )
    print("  per-class f1:", {cls: round(m["f1"], 4) for cls, m in zip(CLASS_NAMES, metrics["per_class"])})
    print("  predicted distribution:", {cls: metrics["pred_counts"][i] for i, cls in enumerate(CLASS_NAMES)})
    metrics["name"] = name
    metrics["seconds"] = seconds
    return metrics


class TemporalCNNClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).mean(dim=-1)
        return self.head(h)


class GRUClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h)


def train_torch_baseline(
    name: str,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
) -> dict:
    started = time.perf_counter()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
    model.eval()
    def predict(loader: DataLoader) -> torch.Tensor:
        preds = []
        with torch.inference_mode():
            for x, _ in loader:
                preds.append(model(x.to(device)).argmax(dim=-1).cpu())
        return torch.cat(preds)

    train_pred = predict(DataLoader(TensorDataset(train_x[: min(4096, len(train_x))], train_y[: min(4096, len(train_y))]), batch_size=batch_size))
    train_metrics = classification_report(train_y[: min(4096, len(train_y))], train_pred, len(CLASS_NAMES))
    val_pred = predict(val_loader)
    val_metrics = classification_report(val_y, val_pred, len(CLASS_NAMES))
    test_pred = predict(test_loader)
    metrics = report_metrics(name, test_y.numpy(), test_pred.numpy().astype(np.int64), time.perf_counter() - started)
    print(f"  train_acc_sample={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
    metrics["parameter_count"] = count_parameters(model)
    metrics["train_accuracy_sample"] = train_metrics["accuracy"]
    metrics["val_accuracy"] = val_metrics["accuracy"]
    metrics["val_macro_f1"] = val_metrics["macro_f1"]
    print(f"  parameters: {metrics['parameter_count']:,}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ReflexFormer-M classical baselines")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--max-train", type=int, default=20_000)
    p.add_argument("--logreg-max-iter", type=int, default=500)
    p.add_argument("--out-json", type=Path, default=Path("results/reflexformer_m_baselines.json"))
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--skip-sequence", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--torch-epochs", type=int, default=None)
    p.add_argument("--torch-batch-size", type=int, default=256)
    p.add_argument("--torch-lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true", default=False)
    p.add_argument("--norm-clip", type=float, default=8.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_x, train_y = load_split(args.data_dir, "train")
    val_x, val_y = load_split(args.data_dir, "val")
    test_x, test_y = load_split(args.data_dir, "test")
    n = min(args.max_train, train_x.shape[0])
    train_x = train_x[:n]
    train_y = train_y[:n]
    if args.normalize:
        normalizer = fit_normalizer(train_x, args.norm_clip)
        train_x = prepare_features(train_x, normalizer, False)
        val_x = prepare_features(val_x, normalizer, False)
        test_x = prepare_features(test_x, normalizer, False)
    torch_epochs = args.epochs if args.torch_epochs is None else args.torch_epochs

    y_train = train_y.numpy()
    y_test = test_y.numpy()
    train_summary = summary_features(train_x)
    test_summary = summary_features(test_x)
    results = []
    device = resolve_device(args.device)

    baselines = [
        (
            "LogisticRegression(flattened_sequence)",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=args.logreg_max_iter, C=1.0, class_weight="balanced", n_jobs=-1),
            ),
            flat_features(train_x),
            flat_features(test_x),
        ),
        (
            "RandomForest(summary_statistics)",
            RandomForestClassifier(n_estimators=180, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=42),
            train_summary,
            test_summary,
        ),
        (
            "HistGradientBoosting(summary_statistics)",
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, l2_regularization=0.02, random_state=42),
            train_summary,
            test_summary,
        ),
    ]
    if not args.skip_mlp:
        baselines.append(
            (
                "MLP(last_timestep)",
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, batch_size=256, random_state=42, early_stopping=True),
                ),
                train_x[:, -1, :].numpy(),
                test_x[:, -1, :].numpy(),
            )
        )

    for name, model, x_train, x_test in baselines:
        started = time.perf_counter()
        model.fit(x_train, y_train)
        pred = model.predict(x_test).astype(np.int64)
        results.append(report_metrics(name, y_test, pred, time.perf_counter() - started))

    if not args.skip_sequence:
        results.append(
            train_torch_baseline(
                "TemporalCNN(sequence)",
                TemporalCNNClassifier(train_x.shape[2]),
                train_x,
                train_y,
                test_x,
                test_y,
                val_x,
                val_y,
                device,
                torch_epochs,
                args.torch_batch_size,
                args.torch_lr,
                args.grad_clip,
            )
        )
        results.append(
            train_torch_baseline(
                "GRU(sequence)",
                GRUClassifier(train_x.shape[2]),
                train_x,
                train_y,
                test_x,
                test_y,
                val_x,
                val_y,
                device,
                torch_epochs,
                args.torch_batch_size,
                args.torch_lr,
                args.grad_clip,
            )
        )
        results.append(
            train_torch_baseline(
                "Transformer-S(sequence)",
                ReflexFormerMClassifier(
                    seq_len=train_x.shape[1],
                    feature_dim=train_x.shape[2],
                    d_model=128,
                    n_layers=3,
                    n_heads=4,
                    mlp_ratio=2.0,
                    dropout=0.1,
                ),
                train_x,
                train_y,
                test_x,
                test_y,
                val_x,
                val_y,
                device,
                torch_epochs,
                args.torch_batch_size,
                args.torch_lr,
                args.grad_clip,
            )
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
