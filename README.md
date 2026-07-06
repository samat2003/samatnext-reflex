# samatnext-reflex

Simulation-only research code for compact transformer reflex models on synthetic drone telemetry risk classification.

The main experiment is `experiments/reflexformer_m`, which contains the ReflexFormer temporal-hard v1 dataset generator, PyTorch models, training/evaluation scripts, benchmarks, baselines, seed sweep tooling, and frozen result notes.

This repository does not contain or support real drone autopilot integration, PX4, ArduPilot, MAVLink, DJI SDK, actuator control, motor control, weapons, live-flight control, certified aviation safety claims, or deployment-readiness claims.

Start here:

```bash
python3 -m py_compile experiments/reflexformer_m/model.py
python experiments/reflexformer_m/make_dataset.py --help
python experiments/reflexformer_m/train.py --help
```

See:

- `experiments/reflexformer_m/README.md`
- `experiments/reflexformer_m/RESULTS_TEMPORAL_HARD_V1.md`
- `experiments/reflexformer_m/CLAIMS_AND_LIMITATIONS.md`
