# samatnext-reflex

Simulation-only research code for compact transformer reflex models on synthetic drone telemetry risk classification.

The main experiment is `experiments/reflexformer_m`, which contains the ReflexFormer temporal-hard v1 dataset generator, PyTorch models, training/evaluation scripts, benchmarks, baselines, seed sweep tooling, and frozen result notes.

This repository does not contain or support real drone autopilot integration, PX4, ArduPilot, MAVLink, DJI SDK, actuator control, motor control, weapons, live-flight control, certified aviation safety claims, or deployment-readiness claims.

## Temporal-hard v1 result

ReflexFormer-S v1 is a compact transformer model for synthetic temporal drone-risk classification.

Frozen seed-42 result:

- Accuracy: 0.9230
- Macro F1: 0.9230
- FAILSAFE recall: 0.9324
- Safety-critical miss rate: 0.0475
- Batch-1 fp16 p95 latency: 1.4686 ms on RTX 5070 Ti Laptop GPU

Lightweight 3-seed sweep:

- Mean macro F1: 0.8984 ± 0.0061
- Mean FAILSAFE recall: 0.9325 ± 0.0191
- Mean safety-critical miss rate: 0.0769 ± 0.0014
- No class collapse observed

Tracked machine-readable summaries are in `artifacts/temporal-hard-v1/`. Generated datasets, checkpoints, model exports, and full run outputs are intentionally ignored.

## Start here

```bash
python3 -m py_compile experiments/reflexformer_m/model.py
python experiments/reflexformer_m/make_dataset.py --help
python experiments/reflexformer_m/train.py --help
```

## See also

- `experiments/reflexformer_m/README.md`
- `experiments/reflexformer_m/RESULTS_TEMPORAL_HARD_V1.md`
- `experiments/reflexformer_m/CLAIMS_AND_LIMITATIONS.md`
- `artifacts/temporal-hard-v1/README.md`
