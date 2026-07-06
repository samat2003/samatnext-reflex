#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic telemetry data for ReflexFormer-M.

This is simulation-only data. It is not collected from real aircraft and is not
validated for live-flight use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


FEATURE_NAMES = [
    "position_delta_x",
    "position_delta_y",
    "position_delta_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "roll",
    "pitch",
    "yaw",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "altitude",
    "obstacle_distance_front",
    "obstacle_distance_left",
    "obstacle_distance_right",
    "obstacle_distance_down",
    "battery_pct",
    "wind_x",
    "wind_y",
    "wind_z",
    "waypoint_error",
    "gps_noise",
    "controller_error",
    "vertical_speed",
    "mission_progress",
]

CLASS_NAMES = ["LOW_RISK", "CAUTION", "AVOIDANCE_NEEDED", "FAILSAFE_NEEDED"]
FEATURE_DIM = len(FEATURE_NAMES)
LABEL_NOISE = {"easy": 0.0, "medium": 0.03, "hard": 0.07, "temporal_hard": 0.05}


def _smooth_noise(rng: np.random.Generator, seq_len: int, scale: float, dim: int = 1) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=(seq_len, dim)).astype(np.float32)
    return np.cumsum(noise, axis=0) / np.sqrt(np.arange(1, seq_len + 1, dtype=np.float32))[:, None]


def _base_sequence(rng: np.random.Generator, seq_len: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    velocity = rng.normal(0.0, 0.35, size=(seq_len, 3)).astype(np.float32)
    velocity += _smooth_noise(rng, seq_len, 0.05, 3)
    position_delta = np.cumsum(velocity, axis=0) * (1.0 / max(seq_len, 1))
    acceleration = np.gradient(velocity, axis=0).astype(np.float32)

    x[:, 0:3] = position_delta
    x[:, 3:6] = velocity
    x[:, 6:9] = acceleration
    x[:, 9] = rng.normal(0.0, 0.03, seq_len)
    x[:, 10] = rng.normal(0.0, 0.03, seq_len)
    x[:, 11] = np.sin(t * np.pi * 2.0 + rng.uniform(-0.2, 0.2)) * 0.08
    x[:, 12:15] = rng.normal(0.0, 0.04, size=(seq_len, 3))
    x[:, 15] = rng.uniform(18.0, 70.0) + rng.normal(0.0, 0.7, seq_len)
    x[:, 16:20] = rng.uniform(18.0, 60.0, size=(seq_len, 4))
    x[:, 20] = np.linspace(rng.uniform(0.55, 0.95), rng.uniform(0.45, 0.9), seq_len)
    x[:, 21:24] = rng.normal(0.0, 1.0, size=(seq_len, 3))
    x[:, 24] = np.abs(rng.normal(0.5, 0.2, seq_len))
    x[:, 25] = np.abs(rng.normal(0.15, 0.05, seq_len))
    x[:, 26] = np.abs(rng.normal(0.12, 0.05, seq_len))
    x[:, 27] = x[:, 5] + rng.normal(0.0, 0.08, seq_len)
    x[:, 28] = np.clip(t + rng.normal(0.0, 0.01, seq_len), 0.0, 1.0)
    return x


def _make_example_for_class(rng: np.random.Generator, seq_len: int, label: int) -> np.ndarray:
    x = _base_sequence(rng, seq_len)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

    if label == 0:  # LOW_RISK
        x[:, 3:6] *= 0.55
        x[:, 6:9] *= 0.45
        x[:, 9:11] += rng.normal(0.0, 0.015, size=(seq_len, 2))
        x[:, 12:15] *= 0.45
        x[:, 15] = rng.uniform(35.0, 80.0) + rng.normal(0.0, 0.5, seq_len)
        x[:, 16:20] = rng.uniform(28.0, 80.0, size=(seq_len, 4))
        x[:, 20] = np.linspace(rng.uniform(0.62, 1.0), rng.uniform(0.55, 0.95), seq_len)
        x[:, 21:24] = rng.normal(0.0, 0.7, size=(seq_len, 3))
        x[:, 24] = np.abs(rng.normal(0.35, 0.15, seq_len))
        x[:, 25] = np.abs(rng.normal(0.10, 0.03, seq_len))
        x[:, 26] = np.abs(rng.normal(0.08, 0.03, seq_len))
    elif label == 1:  # CAUTION
        x[:, 3:6] += rng.normal(0.0, 0.25, size=(seq_len, 3))
        x[:, 9:11] += rng.normal(0.0, 0.055, size=(seq_len, 2))
        x[:, 12:15] += rng.normal(0.0, 0.08, size=(seq_len, 3))
        x[:, 16:20] = rng.uniform(8.0, 25.0, size=(seq_len, 4))
        x[:, 20] = np.linspace(rng.uniform(0.28, 0.55), rng.uniform(0.18, 0.48), seq_len)
        x[:, 21:24] = rng.normal(0.0, 2.2, size=(seq_len, 3))
        x[:, 24] = np.abs(rng.normal(1.5, 0.5, seq_len))
        x[:, 25] = np.abs(rng.normal(0.35, 0.12, seq_len))
        x[:, 26] = np.abs(rng.normal(0.45, 0.15, seq_len))
    elif label == 2:  # AVOIDANCE_NEEDED
        closing = np.linspace(rng.uniform(18.0, 30.0), rng.uniform(1.5, 5.5), seq_len)
        side_clearance = rng.uniform(4.0, 14.0, size=(seq_len, 2))
        x[:, 16] = closing + rng.normal(0.0, 0.5, seq_len)
        x[:, 17:19] = side_clearance + rng.normal(0.0, 0.5, size=(seq_len, 2))
        x[:, 19] = np.linspace(rng.uniform(8.0, 18.0), rng.uniform(1.2, 4.0), seq_len)
        x[:, 3] += rng.choice([-1.0, 1.0]) * np.linspace(0.5, 3.2, seq_len)
        x[:, 4] += rng.choice([-1.0, 1.0]) * np.linspace(0.3, 2.6, seq_len)
        x[:, 24] = np.abs(np.linspace(1.5, 5.0, seq_len) + rng.normal(0.0, 0.35, seq_len))
        x[:, 26] = np.abs(np.linspace(0.8, 2.4, seq_len) + rng.normal(0.0, 0.25, seq_len))
        x[:, 27] = -np.abs(np.linspace(0.2, 2.5, seq_len) + rng.normal(0.0, 0.15, seq_len))
        x[:, 21:24] += rng.normal(0.0, 2.0, size=(seq_len, 3))
    else:  # FAILSAFE_NEEDED
        mode = rng.integers(0, 5)
        if mode == 0:
            x[:, 20] = np.linspace(rng.uniform(0.10, 0.18), rng.uniform(0.01, 0.07), seq_len)
        elif mode == 1:
            x[:, 9] = rng.choice([-1.0, 1.0]) * (0.35 + 0.8 * t) + rng.normal(0.0, 0.06, seq_len)
            x[:, 10] = rng.choice([-1.0, 1.0]) * (0.30 + 0.75 * t) + rng.normal(0.0, 0.06, seq_len)
            x[:, 12:15] += rng.normal(0.0, 0.9, size=(seq_len, 3))
        elif mode == 2:
            x[:, 15] = np.linspace(rng.uniform(12.0, 25.0), rng.uniform(-2.0, 2.0), seq_len)
            x[:, 19] = np.linspace(rng.uniform(7.0, 15.0), rng.uniform(0.0, 1.5), seq_len)
            x[:, 27] = -np.abs(np.linspace(2.0, 6.0, seq_len) + rng.normal(0.0, 0.3, seq_len))
        elif mode == 3:
            x[:, 26] = np.abs(np.linspace(2.5, 8.0, seq_len) + rng.normal(0.0, 0.7, seq_len))
            x[:, 24] = np.abs(np.linspace(2.0, 7.0, seq_len) + rng.normal(0.0, 0.6, seq_len))
        else:
            corrupt_mask = rng.random((seq_len, FEATURE_DIM)) < 0.10
            x[corrupt_mask] += rng.normal(0.0, 12.0, size=int(corrupt_mask.sum()))
            x[:, 25] = np.abs(np.linspace(1.5, 5.0, seq_len) + rng.normal(0.0, 0.5, seq_len))
    x += rng.normal(0.0, 0.025, size=x.shape).astype(np.float32)
    return x.astype(np.float32)


def _corrupt_window(rng: np.random.Generator, x: np.ndarray, strength: str) -> None:
    seq_len = x.shape[0]
    prob = 0.10 if strength == "medium" else 0.28 if strength == "temporal_hard" else 0.18
    if rng.random() > prob:
        return
    width = int(rng.integers(max(4, seq_len // 16), max(5, seq_len // 5)))
    start = int(rng.integers(0, max(1, seq_len - width)))
    cols = rng.choice(FEATURE_DIM, size=int(rng.integers(3, 9)), replace=False)
    mode = rng.choice(["zero", "hold", "spike", "clip"] if strength == "temporal_hard" else ["zero", "hold", "spike"])
    if mode == "zero":
        x[start : start + width, cols] = 0.0
    elif mode == "hold":
        x[start : start + width, cols] = x[max(start - 1, 0), cols]
    else:
        if mode == "clip":
            x[start : start + width, cols] = np.clip(x[start : start + width, cols], -1.0, 1.0)
        else:
            x[start : start + width, cols] += rng.normal(0.0, 2.5 if strength == "medium" else 4.5, size=(width, len(cols)))
    x[start : start + width, 25] += abs(rng.normal(0.4, 0.25))


def _bump(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((t - center) ** 2) / max(width, 1e-4))


def _ramp_after(t: np.ndarray, start: float, amp: float) -> np.ndarray:
    return amp * np.clip((t - start) / max(1.0 - start, 1e-4), 0.0, 1.0)


def _temporal_base(rng: np.random.Generator, seq_len: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    x[:, 28] = np.clip(t + rng.normal(0.0, 0.01, seq_len), 0.0, 1.0)
    x[:, 15] = 32.0 + rng.normal(0.0, 0.45, seq_len)
    x[:, 16] = 24.0 + rng.normal(0.0, 0.65, seq_len)
    x[:, 17] = 20.0 + rng.normal(0.0, 0.8, seq_len)
    x[:, 18] = 20.0 + rng.normal(0.0, 0.8, seq_len)
    x[:, 19] = 12.0 + rng.normal(0.0, 0.45, seq_len)
    x[:, 20] = 0.55 + rng.normal(0.0, 0.012, seq_len)
    x[:, 21:24] = rng.normal(0.0, 0.55, size=(seq_len, 3))
    x[:, 24] = 1.0 + rng.normal(0.0, 0.10, seq_len)
    x[:, 25] = 0.18 + rng.normal(0.0, 0.04, seq_len)
    x[:, 26] = 0.24 + rng.normal(0.0, 0.05, seq_len)
    x[:, 9:11] = rng.normal(0.0, 0.025, size=(seq_len, 2))
    x[:, 11] = np.sin(2.0 * np.pi * t + rng.uniform(-0.3, 0.3)) * 0.04
    x[:, 12:15] = rng.normal(0.0, 0.035, size=(seq_len, 3))
    x[:, 3:6] = rng.normal(0.0, 0.14, size=(seq_len, 3))
    x[:, 6:9] = np.gradient(x[:, 3:6], axis=0).astype(np.float32)
    return x


def _finish_kinematics(x: np.ndarray) -> np.ndarray:
    x[:, 27] = np.gradient(x[:, 15]).astype(np.float32) + x[:, 5] * 0.15
    x[:, 0:3] = np.cumsum(x[:, 3:6], axis=0) / max(x.shape[0], 1)
    x[:, 6:9] = np.gradient(x[:, 3:6], axis=0).astype(np.float32)
    return x


def _apply_temporal_corruption(rng: np.random.Generator, x: np.ndarray) -> None:
    _corrupt_window(rng, x, "temporal_hard")
    seq_len = x.shape[0]
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    if rng.random() < 0.24:
        center = rng.uniform(0.15, 0.8)
        x[:, 25] += _bump(t, center, rng.uniform(0.004, 0.018), rng.uniform(0.8, 2.2))
    if rng.random() < 0.20:
        start = int(rng.integers(0, seq_len - max(5, seq_len // 12)))
        width = int(rng.integers(max(5, seq_len // 20), max(6, seq_len // 8)))
        x[start : start + width, rng.choice([16, 17, 18, 19])] = rng.choice([0.0, 1.0, 60.0])


def _force_late_endpoint_recovery(x: np.ndarray) -> None:
    """Remove easy final-timestep leakage while preserving earlier order cues."""
    seq_len = x.shape[0]
    start = int(seq_len * 0.78)
    alpha = np.linspace(0.0, 1.0, seq_len - start, dtype=np.float32)
    targets = {
        3: 0.0,
        4: 0.0,
        5: 0.0,
        9: 0.0,
        10: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 32.0,
        16: 18.0,
        19: 10.0,
        20: 0.42,
        21: 0.0,
        22: 0.0,
        23: 0.0,
        24: 1.4,
        25: 0.25,
        26: 0.45,
    }
    for col, target in targets.items():
        x[start:, col] = (1.0 - alpha) * x[start:, col] + alpha * target


def _rank_remap_temporal_channels(x: np.ndarray) -> None:
    """Equalize aggregate distributions while retaining temporal order."""
    targets = {
        0: (0.0, 0.18),
        1: (0.0, 0.18),
        2: (0.0, 0.14),
        3: (0.0, 1.0),
        4: (0.0, 0.7),
        5: (0.0, 0.55),
        6: (0.0, 0.24),
        7: (0.0, 0.22),
        8: (0.0, 0.20),
        9: (0.0, 0.28),
        10: (0.0, 0.28),
        11: (0.0, 0.14),
        12: (0.0, 0.38),
        13: (0.0, 0.30),
        14: (0.0, 0.30),
        15: (32.0, 4.0),
        16: (18.0, 7.0),
        17: (20.0, 4.0),
        18: (20.0, 4.0),
        19: (10.0, 3.0),
        20: (0.42, 0.13),
        21: (0.0, 2.0),
        22: (0.0, 1.1),
        23: (0.0, 1.1),
        24: (1.4, 1.5),
        25: (0.25, 0.9),
        26: (0.45, 1.1),
        27: (0.0, 0.65),
    }
    base_shape = np.linspace(-1.0, 1.0, x.shape[0], dtype=np.float32)
    for col, (center, scale) in targets.items():
        order = np.argsort(x[:, col], kind="mergesort")
        remapped = np.empty_like(x[:, col])
        remapped[order] = center + scale * base_shape
        x[:, col] = remapped


def _aggregate_jitter(rng: np.random.Generator, x: np.ndarray) -> None:
    for col in range(FEATURE_DIM - 1):
        vals = x[:, col]
        scale = max(float(vals.std()), 1e-3)
        vals *= rng.normal(1.0, 0.10)
        vals += rng.normal(0.0, 0.25 * scale)
        x[:, col] = vals


def _make_temporal_hard_example_for_class(rng: np.random.Generator, seq_len: int, label: int) -> np.ndarray:
    x = _temporal_base(rng, seq_len)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    scenario = int(rng.integers(0, 5))

    if label == 0:
        # Same adverse extrema as higher-risk classes, but recovery/correction happens before the end.
        if scenario == 0:
            x[:, 16] += -_bump(t, 0.35, 0.018, 14.0) + _bump(t, 0.72, 0.030, 5.0)
        elif scenario == 1:
            x[:, 15] += -_bump(t, 0.42, 0.035, 6.0) + _ramp_after(t, 0.55, 3.0)
        elif scenario == 2:
            x[:, 21] += _bump(t, 0.22, 0.010, 4.2)
            x[:, 3] += _bump(t, 0.34, 0.020, 1.4) - _bump(t, 0.58, 0.025, 1.3)
            x[:, 26] += _bump(t, 0.45, 0.030, 1.2)
        elif scenario == 3:
            x[:, 25] += _bump(t, 0.30, 0.012, 1.8)
            x[:, 24] += _bump(t, 0.46, 0.022, 1.1) - _bump(t, 0.70, 0.035, 0.7)
        else:
            x[:, 20] += -_ramp_after(t, 0.10, 0.18) + _ramp_after(t, 0.62, 0.15)
            x[:, 24] += _bump(t, 0.38, 0.025, 1.2)
    elif label == 1:
        # Caution has unresolved but non-accelerating trends.
        if scenario == 0:
            x[:, 16] += -_ramp_after(t, 0.25, 10.0) + _ramp_after(t, 0.72, 5.0)
        elif scenario == 1:
            x[:, 15] += -_ramp_after(t, 0.18, 5.0) + _ramp_after(t, 0.75, 2.0)
        elif scenario == 2:
            x[:, 21] += _bump(t, 0.50, 0.035, 3.2)
            x[:, 3] += _bump(t, 0.60, 0.040, 1.0)
        elif scenario == 3:
            x[:, 25] += _ramp_after(t, 0.30, 0.8)
            x[:, 24] += _ramp_after(t, 0.62, 1.1)
        else:
            x[:, 26] += _ramp_after(t, 0.25, 0.9) - _ramp_after(t, 0.72, 0.35)
            x[:, 12] += _bump(t, 0.55, 0.025, 0.5)
    elif label == 2:
        # Avoidance: adverse trend is worsening or delayed correction is out of phase.
        if scenario == 0:
            # Paired with LOW_RISK obstacle minimum, but direction is still closing late.
            x[:, 16] += -_ramp_after(t, 0.20, 14.0) + _bump(t, 0.42, 0.020, 2.0)
        elif scenario == 1:
            x[:, 21] += _bump(t, 0.24, 0.012, 4.0)
            x[:, 3] += _ramp_after(t, 0.42, 2.0)
            x[:, 26] += _bump(t, 0.72, 0.040, 1.2)
        elif scenario == 2:
            x[:, 15] += -_ramp_after(t, 0.30, 5.5)
            x[:, 19] += -_ramp_after(t, 0.50, 7.0)
        elif scenario == 3:
            x[:, 25] += _bump(t, 0.22, 0.012, 1.6)
            x[:, 24] += _ramp_after(t, 0.48, 2.4)
        else:
            x[:, 26] += _ramp_after(t, 0.25, 1.8)
            x[:, 16] += -_ramp_after(t, 0.55, 8.0)
    else:
        # Failsafe: early warning followed by later failure, or accumulation despite stable instantaneous state.
        if scenario == 0:
            x[:, 20] += -_ramp_after(t, 0.12, 0.26)
            x[:, 24] += (1.0 - x[:, 28]) * 0.8 + _ramp_after(t, 0.58, 2.4)
        elif scenario == 1:
            x[:, 15] += -_bump(t, 0.30, 0.030, 3.0) - _ramp_after(t, 0.54, 8.0)
            x[:, 3:6] += rng.normal(0.0, 0.04, size=(seq_len, 3))
        elif scenario == 2:
            x[:, 25] += _bump(t, 0.24, 0.010, 2.0)
            x[:, 24] += _ramp_after(t, 0.60, 4.4)
            x[:, 26] += _ramp_after(t, 0.35, 2.2)
        elif scenario == 3:
            x[:, 26] += np.cumsum(np.abs(rng.normal(0.018, 0.010, seq_len))).astype(np.float32)
            x[:, 3:6] *= 0.35
            x[:, 12:15] += _ramp_after(t, 0.55, 0.8)[:, None]
        else:
            x[:, 21] += _bump(t, 0.18, 0.010, 3.6) - _bump(t, 0.62, 0.020, 3.2)
            x[:, 3] += _bump(t, 0.72, 0.030, 2.1)
            x[:, 10] += _ramp_after(t, 0.65, 0.55)

    # Small spurious signals shared across classes; not intentionally label-coded.
    x[:, 11] += rng.normal(0.0, 0.06, seq_len)
    x[:, 18] += rng.normal(0.0, 0.75, seq_len)
    _apply_temporal_corruption(rng, x)
    _rank_remap_temporal_channels(x)
    _force_late_endpoint_recovery(x)
    x += rng.normal(0.0, 0.055, size=x.shape).astype(np.float32)
    _rank_remap_temporal_channels(x)
    _force_late_endpoint_recovery(x)
    x = _finish_kinematics(x)
    _rank_remap_temporal_channels(x)
    _force_late_endpoint_recovery(x)
    _aggregate_jitter(rng, x)
    return x.astype(np.float32)


def _make_harder_example_for_class(
    rng: np.random.Generator,
    seq_len: int,
    label: int,
    difficulty: str,
) -> np.ndarray:
    """Harder generator with overlap and temporal interactions.

    Labels are scenario-driven rather than single-feature-threshold-driven. The
    ranges intentionally overlap, and risk evidence can be early, delayed, or
    spread across weak signals.
    """

    x = _base_sequence(rng, seq_len)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    hard = difficulty == "hard"
    overlap = 1.0 if difficulty == "medium" else 1.35
    scenario = int(rng.integers(0, 6))

    # Shared overlapping background: all classes may contain moderate noise,
    # winds, battery decline, controller error, and non-critical obstacles.
    x[:, 3:6] += rng.normal(0.0, 0.35 * overlap, size=(seq_len, 3))
    x[:, 9:11] += rng.normal(0.0, 0.045 * overlap, size=(seq_len, 2))
    x[:, 12:15] += rng.normal(0.0, 0.07 * overlap, size=(seq_len, 3))
    x[:, 15] = rng.uniform(18.0, 72.0) + rng.normal(0.0, 1.4 * overlap, seq_len)
    x[:, 16:20] = rng.uniform(7.0, 55.0, size=(seq_len, 4))
    x[:, 20] = np.linspace(rng.uniform(0.30, 0.95), rng.uniform(0.16, 0.88), seq_len)
    x[:, 21:24] = rng.normal(0.0, 1.8 * overlap, size=(seq_len, 3))
    x[:, 24] = np.abs(rng.normal(1.0, 0.7 * overlap, seq_len))
    x[:, 25] = np.abs(rng.normal(0.25, 0.18 * overlap, seq_len))
    x[:, 26] = np.abs(rng.normal(0.35, 0.28 * overlap, seq_len))
    x[:, 27] = x[:, 5] + rng.normal(0.0, 0.25 * overlap, seq_len)

    if label == 0:
        # Stable overall, but with distractors that also appear in risk classes.
        x[:, 3:6] *= rng.uniform(0.55, 0.9)
        x[:, 16:20] += rng.uniform(8.0, 22.0)
        x[:, 20] += rng.uniform(0.05, 0.18)
        if scenario in (0, 1):
            x[:, 21] += np.exp(-((t - 0.32) ** 2) / 0.008) * rng.uniform(1.5, 3.0)
        if scenario == 2:
            x[:, 16] -= np.linspace(0.0, rng.uniform(4.0, 9.0), seq_len)
        x[:, 24] *= rng.uniform(0.35, 0.8)
        x[:, 26] *= rng.uniform(0.3, 0.75)
    elif label == 1:
        # Moderate caution can be wind, drift, lower battery, or weak combined risks.
        if scenario in (0, 1):
            gust = np.exp(-((t - rng.uniform(0.25, 0.55)) ** 2) / 0.015)
            x[:, 21] += gust * rng.uniform(2.0, 4.0)
            x[:, 24] += np.clip(t - 0.45, 0, 1) * rng.uniform(0.8, 1.8)
        elif scenario == 2:
            x[:, 20] = np.linspace(rng.uniform(0.45, 0.7), rng.uniform(0.20, 0.42), seq_len)
            x[:, 24] += rng.uniform(0.4, 1.2)
        elif scenario == 3:
            x[:, 16] = np.linspace(rng.uniform(24.0, 40.0), rng.uniform(8.0, 16.0), seq_len)
        else:
            x[:, 25] += np.linspace(0.0, rng.uniform(0.4, 1.2), seq_len)
            x[:, 24] += np.clip(t - 0.55, 0, 1) * rng.uniform(0.6, 1.6)
        x[:, 26] += np.linspace(0.0, rng.uniform(0.3, 1.1), seq_len)
    elif label == 2:
        if scenario in (0, 1):
            # Obstacle closes early or mid-sequence, then can partially recover.
            pivot = rng.uniform(0.35, 0.70)
            close = np.interp(t, [0.0, pivot, 1.0], [rng.uniform(24.0, 45.0), rng.uniform(2.0, 7.0), rng.uniform(6.0, 14.0)])
            x[:, 16] = close + rng.normal(0.0, 0.9 * overlap, seq_len)
            x[:, 24] += np.clip(t - pivot + 0.2, 0, 1) * rng.uniform(1.0, 2.5)
        elif scenario == 2:
            # Wind gust causes delayed trajectory deviation.
            gust_center = rng.uniform(0.18, 0.42)
            x[:, 21] += np.exp(-((t - gust_center) ** 2) / 0.01) * rng.uniform(3.0, 5.5)
            x[:, 3] += np.clip(t - gust_center - 0.12, 0, 1) * rng.choice([-1.0, 1.0]) * rng.uniform(1.6, 3.0)
            x[:, 24] += np.clip(t - gust_center - 0.18, 0, 1) * rng.uniform(1.5, 3.2)
        elif scenario == 3:
            # Unsafe descent near obstacle from accumulated vertical trend.
            x[:, 15] -= np.linspace(0.0, rng.uniform(8.0, 20.0), seq_len)
            x[:, 19] = np.linspace(rng.uniform(12.0, 24.0), rng.uniform(2.0, 6.0), seq_len)
            x[:, 27] -= np.linspace(0.3, rng.uniform(1.8, 3.5), seq_len)
        else:
            # Mixed weak signals combine into avoidance.
            x[:, 16] -= np.linspace(0.0, rng.uniform(7.0, 14.0), seq_len)
            x[:, 24] += np.linspace(0.0, rng.uniform(1.2, 2.8), seq_len)
            x[:, 26] += np.linspace(0.0, rng.uniform(0.9, 2.0), seq_len)
    else:
        if scenario == 0:
            # Battery falls while return-distance proxy stays high.
            return_distance = 1.0 - x[:, 28]
            x[:, 20] = np.linspace(rng.uniform(0.32, 0.52), rng.uniform(0.03, 0.16), seq_len)
            x[:, 24] += return_distance * rng.uniform(1.5, 3.5)
        elif scenario == 1:
            # Attitude instability accumulates gradually, not just at the end.
            x[:, 9] += np.cumsum(rng.normal(0.0, 0.045, seq_len)) + np.linspace(0, rng.choice([-1, 1]) * rng.uniform(0.28, 0.7), seq_len)
            x[:, 10] += np.cumsum(rng.normal(0.0, 0.04, seq_len)) + np.linspace(0, rng.choice([-1, 1]) * rng.uniform(0.24, 0.65), seq_len)
            x[:, 12:15] += rng.normal(0.0, 0.45, size=(seq_len, 3))
        elif scenario == 2:
            # Altitude drifts downward across the window.
            x[:, 15] = np.linspace(rng.uniform(20.0, 42.0), rng.uniform(-1.0, 8.0), seq_len)
            x[:, 27] -= np.linspace(0.5, rng.uniform(2.5, 5.0), seq_len)
        elif scenario == 3:
            # Controller error accumulation.
            x[:, 26] += np.cumsum(np.abs(rng.normal(0.025, 0.02, seq_len)))
            x[:, 24] += np.linspace(0.0, rng.uniform(2.0, 4.2), seq_len)
        elif scenario == 4:
            # GPS noise increases before waypoint error becomes large.
            x[:, 25] += np.linspace(0.0, rng.uniform(1.8, 3.5), seq_len)
            x[:, 24] += np.clip(t - 0.55, 0, 1) * rng.uniform(3.0, 6.0)
        else:
            # Corruption plus weak physical degradation.
            x[:, 20] -= np.linspace(0.0, rng.uniform(0.15, 0.32), seq_len)
            x[:, 26] += np.linspace(0.0, rng.uniform(1.4, 3.4), seq_len)
            _corrupt_window(rng, x, difficulty)

    # Spurious distractors: weak label-correlated signals should not determine labels.
    distractor = (label - 1.5) * (0.10 if difficulty == "medium" else 0.06)
    x[:, 11] += distractor + rng.normal(0.0, 0.13 if hard else 0.09, seq_len)
    x[:, 28] = np.clip(x[:, 28] + rng.normal(distractor * 0.3, 0.05 if hard else 0.035, seq_len), 0.0, 1.0)

    # Extra overlap and corruption applied to all classes.
    if rng.random() < (0.22 if hard else 0.14):
        weak = rng.choice([16, 20, 24, 25, 26])
        x[:, weak] += rng.normal(0.0, 1.2 if hard else 0.8, seq_len)
    _corrupt_window(rng, x, difficulty)
    x += rng.normal(0.0, 0.10 if difficulty == "medium" else 0.16, size=x.shape).astype(np.float32)
    return x.astype(np.float32)


def make_split(
    rng: np.random.Generator,
    size: int,
    seq_len: int,
    difficulty: str = "easy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = np.arange(size, dtype=np.int64) % len(CLASS_NAMES)
    rng.shuffle(labels)
    clean_labels = labels.copy()
    x = np.empty((size, seq_len, FEATURE_DIM), dtype=np.float32)
    for i, label in enumerate(labels):
        if difficulty == "easy":
            x[i] = _make_example_for_class(rng, seq_len, int(label))
        elif difficulty == "temporal_hard":
            x[i] = _make_temporal_hard_example_for_class(rng, seq_len, int(label))
        else:
            x[i] = _make_harder_example_for_class(rng, seq_len, int(label), difficulty)

    noise_rate = LABEL_NOISE[difficulty]
    if noise_rate > 0:
        flip_mask = rng.random(size) < noise_rate
        for idx in np.where(flip_mask)[0]:
            choices = [c for c in range(len(CLASS_NAMES)) if c != labels[idx]]
            labels[idx] = int(rng.choice(choices))
    return torch.from_numpy(x), torch.from_numpy(labels), torch.from_numpy(clean_labels)


def save_split(out_dir: Path, name: str, x: torch.Tensor, y: torch.Tensor, clean_y: torch.Tensor) -> None:
    torch.save(
        {"x": x, "y": y, "clean_y": clean_y, "class_names": CLASS_NAMES, "feature_names": FEATURE_NAMES},
        out_dir / f"{name}.pt",
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ReflexFormer-M synthetic telemetry")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--train-size", type=int, default=80_000)
    p.add_argument("--val-size", type=int, default=10_000)
    p.add_argument("--test-size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "temporal_hard"], default="easy")
    p.add_argument("--out-dir", type=Path, default=Path("data/reflexformer_m"))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    metadata = {
        "experiment": "ReflexFormer-M",
        "simulation_only": True,
        "seq_len": args.seq_len,
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "label_noise_rate": LABEL_NOISE[args.difficulty],
        "splits": {"train": args.train_size, "val": args.val_size, "test": args.test_size},
        "safety_scope": "Synthetic advisory-label research only; no live-flight or actuator control.",
        "hardening_notes": [
            "medium/hard use overlapping feature ranges",
            "medium/hard add missing or corrupted telemetry windows",
            "medium/hard include delayed temporal-risk scenarios",
            "medium/hard include weak spurious distractors and mixed-risk cases",
            "temporal_hard uses paired order/trend/recovery counterexamples",
            "temporal_hard labels depend on event order, trend direction, delayed consequences, and recovery vs non-recovery",
            "no single feature threshold is intended to perfectly determine class",
        ],
    }

    distributions: dict[str, dict[str, int]] = {}
    clean_distributions: dict[str, dict[str, int]] = {}
    for split, size in (("train", args.train_size), ("val", args.val_size), ("test", args.test_size)):
        x, y, clean_y = make_split(rng, size, args.seq_len, args.difficulty)
        save_split(args.out_dir, split, x, y, clean_y)
        counts = torch.bincount(y, minlength=len(CLASS_NAMES)).tolist()
        clean_counts = torch.bincount(clean_y, minlength=len(CLASS_NAMES)).tolist()
        distributions[split] = {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))}
        clean_distributions[split] = {CLASS_NAMES[i]: int(clean_counts[i]) for i in range(len(CLASS_NAMES))}
        print(
            f"{split}: x={tuple(x.shape)} y={tuple(y.shape)} "
            f"distribution={distributions[split]} clean_distribution={clean_distributions[split]}"
        )

    metadata["class_distribution"] = distributions
    metadata["clean_class_distribution"] = clean_distributions
    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
