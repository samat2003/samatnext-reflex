# AGENTS.md

## Project: samatnext-reflex

This repo is for simulation-only research on ultra-fast transformer reflex models for drone safety-advisory experiments.

## Scope

Allowed:
- Synthetic drone telemetry generation
- PyTorch transformer models
- Training, evaluation, and latency benchmarking
- Simulation-only risk/action classification
- Documentation of results, claims, and limitations

Not allowed:
- No real drone autopilot integration
- No PX4, ArduPilot, MAVLink, DJI SDK, or live-flight control
- No actuator, motor, weapon, or navigation-control code
- No claims of certified aviation safety
- No claims of real-world deployment readiness

## Coding rules

- Keep scripts runnable from repo root.
- Use clear CLI arguments.
- Support CPU fallback.
- Use CUDA acceleration when available.
- Keep benchmark claims honest:
  - batch-1 latency is single-example latency
  - batched throughput is not single-example latency
  - tokens/sec means input telemetry-token throughput, not LLM generation speed
- Save generated data under `data/`.
- Save checkpoints under `checkpoints/`.
- Keep generated data and checkpoints out of git.

## Preferred experiment layout

experiments/reflexformer_m/
  README.md
  CLAIMS_AND_LIMITATIONS.md
  make_dataset.py
  model.py
  train.py
  eval.py
  benchmark.py
  requirements.txt

## Quality bar

Before finalizing changes, run a small smoke test:
- model import / parameter count
- small dataset generation
- short training run
- evaluation
- latency benchmark

Always report:
- files changed
- commands run
- metrics
- benchmark results
- limitations
