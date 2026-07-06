#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inference benchmark for ReflexFormer-M."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import FEATURE_DIM, ReflexFormerMClassifier, apply_preset, count_parameters
from experiments.reflexformer_m.train import resolve_device


def load_or_init_model(args: argparse.Namespace, device: torch.device) -> tuple[ReflexFormerMClassifier, bool]:
    ckpt_args = argparse.Namespace(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        pooling=getattr(args, "pooling", "cls"),
    )
    checkpoint_loaded = False
    checkpoint = None
    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved_args = checkpoint.get("args", {})
        for name in ["d_model", "n_layers", "n_heads", "mlp_ratio", "pooling", "add_deltas"]:
            if name in saved_args:
                setattr(ckpt_args, name, saved_args[name])
        if "feature_dim" in checkpoint:
            args.feature_dim = int(checkpoint["feature_dim"])
        checkpoint_loaded = True
    model = ReflexFormerMClassifier(
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        d_model=ckpt_args.d_model,
        n_layers=ckpt_args.n_layers,
        n_heads=ckpt_args.n_heads,
        mlp_ratio=ckpt_args.mlp_ratio,
        dropout=0.0,
        pooling=getattr(ckpt_args, "pooling", "cls"),
    )
    if checkpoint_loaded and checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_loaded


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return mapping[name]


def benchmark_batch(model: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int, device: torch.device) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            arr = np.asarray(times, dtype=np.float64)
        else:
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(x)
                times.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(times, dtype=np.float64)

    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p90": float(np.percentile(arr, 90)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark ReflexFormer-M inference")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/reflexformer_m/best.pt"))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--feature-dim", type=int, default=FEATURE_DIM)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--preset", choices=["s", "m"], default="m")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=3.0)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args = apply_preset(args)
    device = resolve_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    model, checkpoint_loaded = load_or_init_model(args, device)
    if dtype in (torch.float16, torch.bfloat16) and device.type != "cuda":
        print("non-fp32 dtype requested on CPU; using fp32")
        dtype = torch.float32
    model = model.to(dtype=dtype)
    if args.compile:
        model = torch.compile(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print("checkpoint:", args.checkpoint if checkpoint_loaded else "not loaded; random initialized model")
    print(f"device: {device_name}")
    print(f"dtype: {dtype}")
    print(f"parameter count: {count_parameters(model):,}")
    print("Batch-1 latency is the single-example latency.")
    print("Batched throughput is not single-example latency.")
    print("Tokens/sec is input telemetry-token throughput, not language generation speed and not training speed.")
    print()
    print("| batch | mean_ms | p50_ms | p90_ms | p95_ms | amortized_ms_per_example | examples_per_sec | input_tokens_per_sec | peak_cuda_mem_mb |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for batch in args.batch_sizes:
        x = torch.randn(batch, args.seq_len, args.feature_dim, device=device, dtype=dtype)
        warmup = args.warmup if batch <= 64 else max(10, args.warmup // 2)
        iters = args.iters if batch <= 64 else max(50, args.iters // 2)
        result = benchmark_batch(model, x, warmup, iters, device)
        latency_ms = result["latency_ms_mean"]
        amortized_ms = latency_ms / batch
        examples_sec = batch * 1000.0 / latency_ms
        tokens_sec = examples_sec * args.seq_len
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
        row = {
            "batch": batch,
            **result,
            "amortized_ms_per_example": amortized_ms,
            "examples_per_sec": examples_sec,
            "input_tokens_per_sec": tokens_sec,
            "peak_cuda_mem_mb": peak_mb,
            "device": device_name,
            "dtype": str(dtype),
            "parameter_count": count_parameters(model),
            "compile": bool(args.compile),
        }
        rows.append(row)
        print(
            f"| {batch} | {latency_ms:.4f} | {result['latency_ms_p50']:.4f} | "
            f"{result['latency_ms_p90']:.4f} | {result['latency_ms_p95']:.4f} | "
            f"{amortized_ms:.4f} | {examples_sec:.2f} | {tokens_sec:.2f} | {peak_mb:.1f} |"
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote CSV: {args.csv}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint) if checkpoint_loaded else None,
            "checkpoint_loaded": checkpoint_loaded,
            "device": device_name,
            "dtype": str(dtype),
            "parameter_count": count_parameters(model),
            "seq_len": args.seq_len,
            "feature_dim": args.feature_dim,
            "batch_1_latency_is_single_example_latency": True,
            "batched_throughput_is_not_single_example_latency": True,
            "tokens_per_second_definition": "input telemetry-token throughput, not language generation or training speed",
            "rows": rows,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
