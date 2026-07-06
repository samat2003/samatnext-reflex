#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train ReflexFormer-M on synthetic telemetry advisory labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import CLASS_NAMES, ReflexFormerMClassifier, apply_preset, count_parameters


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_split(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=False)
    return obj["x"].float(), obj["y"].long()


def fit_normalizer(x: torch.Tensor, clip: float) -> dict:
    flat = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, x.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)
    return {"mean": mean, "std": std, "clip": float(clip)}


def apply_normalizer(x: torch.Tensor, normalizer: dict | None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if not normalizer:
        return x
    mean = normalizer["mean"].to(dtype=x.dtype)
    std = normalizer["std"].to(dtype=x.dtype)
    clip = float(normalizer.get("clip", 8.0))
    return ((x - mean) / std).clamp(-clip, clip)


def add_delta_features(x: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(x)
    delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([x, delta], dim=-1)


def prepare_features(x: torch.Tensor, normalizer: dict | None = None, add_deltas: bool = False) -> torch.Tensor:
    x = apply_normalizer(x, normalizer)
    if add_deltas:
        x = add_delta_features(x)
    return x


def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> dict:
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf[int(t), int(p)] += 1

    per_class = []
    for i in range(num_classes):
        tp = conf[i, i].item()
        fp = conf[:, i].sum().item() - tp
        fn = conf[i, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(conf[i, :].sum())})
    accuracy = (y_true == y_pred).float().mean().item() if y_true.numel() else 0.0
    macro_f1 = sum(m["f1"] for m in per_class) / num_classes
    pred_counts = torch.bincount(y_pred, minlength=num_classes).tolist()
    true_counts = torch.bincount(y_true, minlength=num_classes).tolist()
    max_share = max(pred_counts) / max(sum(pred_counts), 1)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "max_predicted_class_share": max_share,
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(dim=-1).cpu())
        ys.append(y.cpu())
    return classification_report(torch.cat(ys), torch.cat(preds), len(CLASS_NAMES))


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def make_class_weights(y: torch.Tensor, mode: str, failsafe_weight: float, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        weights = torch.ones(len(CLASS_NAMES), dtype=torch.float32)
    elif mode == "auto":
        counts = torch.bincount(y, minlength=len(CLASS_NAMES)).float().clamp_min(1.0)
        weights = counts.sum() / (len(CLASS_NAMES) * counts)
        weights = weights / weights.mean()
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    weights[CLASS_NAMES.index("FAILSAFE_NEEDED")] *= failsafe_weight
    return weights.to(device)


def balanced_checkpoint_score(metrics: dict, min_class_recall_floor: float = 0.35) -> float:
    failsafe_idx = CLASS_NAMES.index("FAILSAFE_NEEDED")
    failsafe_recall = metrics["per_class"][failsafe_idx]["recall"]
    max_share = metrics["max_predicted_class_share"]
    min_recall = min(m["recall"] for m in metrics["per_class"])
    score = (
        metrics["macro_f1"]
        + 0.15 * failsafe_recall
        - 0.50 * max(0.0, max_share - 0.45)
        - 0.50 * max(0.0, min_class_recall_floor - min_recall)
    )
    if is_hard_fail_checkpoint(metrics, min_class_recall_floor):
        score -= 10.0
    return score


def is_hard_fail_checkpoint(metrics: dict, min_class_recall_floor: float = 0.35) -> bool:
    max_share = metrics.get("max_predicted_class_share", 0.0)
    pred_counts = metrics.get("pred_counts", [])
    total = max(sum(pred_counts), 1)
    pred_shares = [c / total for c in pred_counts]
    used_classes = sum(1 for c in pred_counts if c > 0)
    return (
        max_share > 0.70
        or metrics.get("macro_f1", 0.0) < 0.30
        or used_classes < 3
        or any(share == 0.0 for share in pred_shares)
    )


def warn_for_collapse(metrics: dict) -> list[str]:
    warnings = []
    if metrics["max_predicted_class_share"] > 0.75:
        warnings.append(f"WARNING: one predicted class share is {metrics['max_predicted_class_share']:.3f} (>0.75)")
    for name, cls_metrics in zip(CLASS_NAMES, metrics["per_class"]):
        if cls_metrics["recall"] < 0.02:
            warnings.append(f"WARNING: near-zero recall for {name}: {cls_metrics['recall']:.3f}")
    return warnings


def build_model(args: argparse.Namespace, feature_dim: int, seq_len: int) -> ReflexFormerMClassifier:
    return ReflexFormerMClassifier(
        seq_len=seq_len,
        feature_dim=feature_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        pooling=getattr(args, "pooling", "cls"),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
    normalizer: dict | None,
    feature_dim: int,
    seq_len: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "class_names": CLASS_NAMES,
            "parameter_count": count_parameters(model),
            "feature_dim": feature_dim,
            "seq_len": seq_len,
            "normalizer": normalizer,
            "simulation_only": True,
        },
        path,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ReflexFormer-M")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/reflexformer_m"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--preset", choices=["s", "m"], default="m")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=3.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weights", choices=["auto", "none"], default="none")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--failsafe-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-failsafe-recall", type=float, default=0.0)
    p.add_argument("--min-class-recall-floor", type=float, default=0.35)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--norm-clip", type=float, default=8.0)
    p.add_argument("--add-deltas", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args = apply_preset(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_x_raw, train_y = load_split(args.data_dir, "train")
    val_x_raw, val_y = load_split(args.data_dir, "val")
    normalizer = fit_normalizer(train_x_raw, args.norm_clip) if args.normalize else None
    train_x = prepare_features(train_x_raw, normalizer, args.add_deltas)
    val_x = prepare_features(val_x_raw, normalizer, args.add_deltas)
    seq_len, feature_dim = train_x.shape[1], train_x.shape[2]
    print(f"normalize: {args.normalize} add_deltas: {args.add_deltas} feature_dim: {feature_dim}")

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, feature_dim, seq_len).to(device)
    print(f"parameter count: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs * len(train_loader), 1))
    class_weights = make_class_weights(train_y, args.class_weights, args.failsafe_weight, device)
    if class_weights is not None:
        print("class weights:", {name: round(float(class_weights[i].detach().cpu()), 4) for i, name in enumerate(CLASS_NAMES)})
    if args.focal_loss:
        criterion = FocalLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        started = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{total_loss / max(n_seen, 1):.4f}")

        train_loss = total_loss / max(n_seen, 1)
        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch_seconds"] = time.perf_counter() - started
        metrics["primary_score"] = balanced_checkpoint_score(metrics, args.min_class_recall_floor)
        metrics["best_eligible"] = not is_hard_fail_checkpoint(metrics, args.min_class_recall_floor)
        history.append(metrics)
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, args, epoch, metrics, normalizer, feature_dim, seq_len)
        if metrics["best_eligible"] and metrics["primary_score"] > best_score:
            best_score = metrics["primary_score"]
            epochs_without_improvement = 0
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, args, epoch, metrics, normalizer, feature_dim, seq_len)
        else:
            epochs_without_improvement += 1

        failsafe_idx = CLASS_NAMES.index("FAILSAFE_NEEDED")
        failsafe_recall = metrics["per_class"][failsafe_idx]["recall"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_failsafe_recall={failsafe_recall:.4f} primary_score={metrics['primary_score']:.4f} "
            f"best_eligible={metrics['best_eligible']} max_pred_share={metrics['max_predicted_class_share']:.3f}"
        )
        print("per-class f1:", {name: round(m["f1"], 4) for name, m in zip(CLASS_NAMES, metrics["per_class"])})
        print("per-class recall:", {name: round(m["recall"], 4) for name, m in zip(CLASS_NAMES, metrics["per_class"])})
        print("predicted distribution:", {name: metrics["pred_counts"][i] for i, name in enumerate(CLASS_NAMES)})
        print("confusion matrix rows=true cols=pred:")
        for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
            print(f"  {name}: {row}")
        if args.min_failsafe_recall and failsafe_recall < args.min_failsafe_recall:
            print(f"WARNING: FAILSAFE_NEEDED recall {failsafe_recall:.4f} below target {args.min_failsafe_recall:.4f}")
        for warning in warn_for_collapse(metrics):
            print(warning)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping: no primary-score improvement for {args.patience} epochs")
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        f.write("\n")
    if not (args.out_dir / "best.pt").exists():
        print("WARNING: no eligible best checkpoint; copying last.pt to best.pt for downstream inspection")
        last = torch.load(args.out_dir / "last.pt", map_location="cpu", weights_only=False)
        torch.save(last, args.out_dir / "best.pt")
    print(f"best validation primary score: {best_score:.4f}")
    print(f"saved: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
