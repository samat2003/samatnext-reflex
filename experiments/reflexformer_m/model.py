#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ReflexFormer-M simulation-only telemetry risk classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn


CLASS_NAMES = ["LOW_RISK", "CAUTION", "AVOIDANCE_NEEDED", "FAILSAFE_NEEDED"]
FEATURE_DIM = 29
PRESETS = {
    "s": {"d_model": 256, "n_layers": 6, "n_heads": 4, "mlp_ratio": 2.0},
    "m": {"d_model": 384, "n_layers": 10, "n_heads": 6, "mlp_ratio": 3.0},
}


@dataclass(frozen=True)
class ReflexFormerMConfig:
    seq_len: int = 128
    feature_dim: int = FEATURE_DIM
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.1
    num_classes: int = len(CLASS_NAMES)
    pooling: str = "cls"


class ReflexFormerMClassifier(nn.Module):
    """Compact transformer encoder classifier for synthetic drone telemetry.

    The model predicts conservative advisory classes only. It does not output
    actuator, motor, navigation, or autopilot commands.
    """

    def __init__(
        self,
        seq_len: int = 128,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 384,
        n_layers: int = 10,
        n_heads: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
        num_classes: int = len(CLASS_NAMES),
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.config = ReflexFormerMConfig(
            seq_len=seq_len,
            feature_dim=feature_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=num_classes,
            pooling=pooling,
        )

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.embed_dropout = nn.Dropout(dropout)

        ff_dim = int(d_model * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, feature_dim = x.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds configured max {self.config.seq_len}")
        if feature_dim != self.config.feature_dim:
            raise ValueError(f"feature_dim {feature_dim} != configured {self.config.feature_dim}")

        h = self.input_proj(x)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed[:, : seq_len + 1]
        h = self.embed_dropout(h)
        h = self.encoder(h)
        if self.config.pooling == "mean":
            h = self.norm(h[:, 1:].mean(dim=1))
        else:
            h = self.norm(h[:, 0])
        return self.head(h)


class ReflexFormerSClassifier(ReflexFormerMClassifier):
    """Smaller preset for faster Jetson-class simulation experiments."""

    def __init__(
        self,
        seq_len: int = 128,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.1,
        num_classes: int = len(CLASS_NAMES),
        pooling: str = "cls",
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            feature_dim=feature_dim,
            d_model=256,
            n_layers=6,
            n_heads=4,
            mlp_ratio=2.0,
            dropout=dropout,
            num_classes=num_classes,
            pooling=pooling,
        )


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = getattr(args, "preset", None)
    if preset in PRESETS:
        for key, value in PRESETS[preset].items():
            setattr(args, key, value)
    return args


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReflexFormer-M model smoke test")
    p.add_argument("--preset", choices=["s", "m"], default="m")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--feature-dim", type=int, default=FEATURE_DIM)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=3.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args = apply_preset(args)
    model = ReflexFormerMClassifier(
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    x = torch.randn(2, args.seq_len, args.feature_dim)
    logits = model(x)
    print(f"model: ReflexFormer-M")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"parameter count: {count_parameters(model):,}")
    print(f"classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
