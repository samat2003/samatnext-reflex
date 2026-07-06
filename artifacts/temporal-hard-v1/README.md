# Temporal-Hard V1 Frozen Artifacts

This folder contains lightweight, tracked machine-readable summaries for the ReflexFormer temporal-hard v1 result.

These artifacts are copied from the frozen metrics documented in `experiments/reflexformer_m/RESULTS_TEMPORAL_HARD_V1.md` and are intended for quick portfolio/review use without committing generated datasets, checkpoints, or full `results/` output directories.

## Included files

- `reflexformer_s_v1_eval.json` — seed-42 ReflexFormer-S v1 evaluation summary.
- `reflexformer_s_v1_benchmark.csv` — seed-42 ReflexFormer-S v1 fp16 benchmark summary on RTX 5070 Ti Laptop GPU.
- `reflexformer_s_v1_benchmark.json` — same benchmark summary in JSON form.
- `reflexformer_s_temporal_hard_3seed_sweep.csv` — lightweight 3-seed ReflexFormer-S mini sweep summary.
- `reflexformer_s_temporal_hard_3seed_sweep.json` — same sweep in JSON form.

## Scope

These are synthetic, simulation-only metrics. They do not imply real-world drone safety, live-flight readiness, certified aviation safety, obstacle-avoidance guarantees, autopilot capability, actuator-control reliability, or deployment readiness.

Generated datasets, checkpoints, ONNX/TensorRT exports, and full experiment output folders remain intentionally ignored by git.
