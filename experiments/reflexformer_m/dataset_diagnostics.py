#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for ReflexFormer-M synthetic datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.baselines import summary_features
from experiments.reflexformer_m.make_dataset import FEATURE_NAMES
from experiments.reflexformer_m.model import CLASS_NAMES
from experiments.reflexformer_m.train import classification_report


def load_obj(data_dir: Path, split: str) -> dict:
    return torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=False)


def slope_features(x: torch.Tensor) -> np.ndarray:
    arr = x.numpy()
    q = arr.shape[1] // 4
    first = arr[:, :q, :].mean(axis=1)
    second = arr[:, q : 2 * q, :].mean(axis=1)
    third = arr[:, 2 * q : 3 * q, :].mean(axis=1)
    fourth = arr[:, 3 * q :, :].mean(axis=1)
    return np.concatenate([second - first, third - second, fourth - third, fourth - first], axis=1).astype(np.float32)


def fit_probe(name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict:
    if train_x.shape[1] > 200:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced"))
    else:
        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=42)
    model.fit(train_x, train_y)
    pred = model.predict(test_x).astype(np.int64)
    metrics = classification_report(torch.from_numpy(test_y), torch.from_numpy(pred), len(CLASS_NAMES))
    print(
        f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"max_pred_share={metrics['max_predicted_class_share']:.4f}"
    )
    print("  per-class f1:", {cls: round(m["f1"], 4) for cls, m in zip(CLASS_NAMES, metrics["per_class"])})
    return metrics


def print_feature_stats(x: torch.Tensor, y: torch.Tensor) -> None:
    arr = x.numpy()
    labels = y.numpy()
    print("per-feature mean/std by class:")
    for idx, name in enumerate(FEATURE_NAMES):
        pieces = []
        for cls_idx, cls in enumerate(CLASS_NAMES):
            vals = arr[labels == cls_idx, :, idx]
            pieces.append(f"{cls}=mean:{vals.mean():.3f}/std:{vals.std():.3f}")
        print(f"  {name}: " + " | ".join(pieces))


def paired_counterexamples(x: torch.Tensor, y: torch.Tensor, limit: int = 5) -> None:
    feats = summary_features(x)
    n = min(len(feats), 2500)
    feats = feats[:n]
    labels = y.numpy()[:n]
    nn = NearestNeighbors(n_neighbors=8, metric="euclidean").fit(feats)
    distances, indices = nn.kneighbors(feats)
    print("nearest-summary paired counterexample candidates:")
    found = 0
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            if labels[i] != labels[j]:
                print(
                    f"  idx={i} {CLASS_NAMES[labels[i]]} vs idx={int(j)} {CLASS_NAMES[labels[j]]} "
                    f"summary_distance={float(dist):.4f}"
                )
                found += 1
                break
        if found >= limit:
            break
    if found == 0:
        print("  none found in sampled subset")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze ReflexFormer-M dataset difficulty")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--max-train", type=int, default=20_000)
    p.add_argument("--print-feature-stats", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    meta_path = args.data_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"difficulty: {metadata.get('difficulty', 'unknown')}")
        print(f"metadata label_noise_rate: {metadata.get('label_noise_rate', 'unknown')}")

    train = load_obj(args.data_dir, "train")
    test = load_obj(args.data_dir, "test")
    train_x = train["x"].float()[: args.max_train]
    train_y = train["y"].long()[: args.max_train]
    test_x = test["x"].float()
    test_y = test["y"].long()

    print("class distribution:")
    for split, obj in [("train", train), ("test", test)]:
        counts = torch.bincount(obj["y"].long(), minlength=len(CLASS_NAMES)).tolist()
        print(f"  {split}: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES))))

    if "clean_y" in train and "clean_y" in test:
        train_noise = (train["clean_y"].long() != train["y"].long()).float().mean().item()
        test_noise = (test["clean_y"].long() != test["y"].long()).float().mean().item()
        print(f"label noise estimate from clean_y: train={train_noise:.4f} test={test_noise:.4f}")

    if args.print_feature_stats:
        print_feature_stats(test_x, test_y)

    y_train = train_y.numpy()
    y_test = test_y.numpy()
    print("probe baselines:")
    fit_probe("last_timestep_only", train_x[:, -1, :].numpy(), y_train, test_x[:, -1, :].numpy(), y_test)
    fit_probe("summary_statistics", summary_features(train_x), y_train, summary_features(test_x), y_test)
    fit_probe("temporal_slope_derivative", slope_features(train_x), y_train, slope_features(test_x), y_test)
    paired_counterexamples(test_x, test_y)


if __name__ == "__main__":
    main()
