# ReflexFormer Temporal-Hard V1 Results

This file freezes the ReflexFormer temporal-hard v1 smoke result for simulation-only research. It is not evidence of real drone safety, live-flight readiness, certified aviation behavior, or deployment readiness.

## Dataset Settings

- Generator: `experiments/reflexformer_m/make_dataset.py`
- Difficulty: `temporal_hard`
- Sequence length: 128
- Features per timestep: 29
- Train examples: 20,000
- Validation examples: 3,000
- Test examples: 3,000
- Seed: 42
- Temporal-hard label noise target: 5%
- Training/eval preprocessing for ReflexFormer: train-split normalization plus delta channels via `--add-deltas`

## Label Distribution

| Split | LOW_RISK | CAUTION | AVOIDANCE_NEEDED | FAILSAFE_NEEDED |
|---|---:|---:|---:|---:|
| train | 4,977 | 4,997 | 5,019 | 5,007 |
| val | 739 | 741 | 751 | 769 |
| test | 751 | 755 | 740 | 754 |

Estimated label noise from the stored clean labels was 0.0476 on train and 0.0563 on test.

## Diagnostics

| Probe | Accuracy | Macro F1 | Max Pred Share |
|---|---:|---:|---:|
| last timestep only | 0.3313 | 0.3227 | 0.3217 |
| summary statistics | 0.4777 | 0.4758 | 0.2873 |
| temporal slope/derivative | 0.9390 | 0.9390 | 0.2510 |

The temporal derivative probe remains very strong. That is useful as a sanity check that the temporal-hard split really contains temporal signal, but it also means any model-quality claim should compare against derivative-aware or sequence-aware baselines.

## Baselines

| Model | Test Accuracy | Macro F1 | FAILSAFE Recall | Max Pred Share |
|---|---:|---:|---:|---:|
| LogisticRegression(flattened sequence) | 0.7723 | 0.7718 | 0.7997 | 0.2580 |
| RandomForest(summary statistics) | 0.2730 | 0.2728 | 0.2984 | 0.2727 |
| HistGradientBoosting(summary statistics) | 0.4743 | 0.4729 | 0.5040 | 0.2793 |
| MLP(last timestep) | 0.2970 | 0.2874 | 0.2984 | 0.3553 |
| TemporalCNN(sequence) | 0.8650 | 0.8648 | 0.8753 | 0.2643 |
| GRU(sequence) | 0.8830 | 0.8825 | 0.9164 | 0.2867 |
| small Transformer-S(sequence) | 0.9240 | 0.9239 | 0.9297 | 0.2657 |

## ReflexFormer Metrics

| Model | Params | Test Accuracy | Macro F1 | FAILSAFE Recall | Safety-Critical Miss Rate | Max Pred Share |
|---|---:|---:|---:|---:|---:|---:|
| ReflexFormer-S stable | 3,278,340 | 0.9230 | 0.9230 | 0.9324 | 0.0475 | 0.2550 |
| ReflexFormer-M stable | 15,014,404 | 0.9130 | 0.9130 | 0.9257 | 0.0596 | 0.2537 |

ReflexFormer-S beat ReflexFormer-M in this smoke run. This result does not justify a "bigger is better" claim.

## Seed Stability

A lightweight 3-seed mini sweep was run for ReflexFormer-S with `temporal_hard`, 12,000 train examples, 2,000 validation examples, 2,000 test examples, and 12 training epochs. This is smaller than the clean seed-42 v1 reproduction pass above, so the absolute scores are not directly comparable to the 20,000/3,000/3,000, 20-epoch result.

| Seed | Accuracy | Macro F1 | FAILSAFE Recall | Safety-Critical Miss Rate | Max Pred Share |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9055 | 0.9054 | 0.9499 | 0.0778 | 0.2530 |
| 2 | 0.8940 | 0.8942 | 0.9120 | 0.0775 | 0.2700 |
| 3 | 0.8955 | 0.8955 | 0.9356 | 0.0753 | 0.2595 |
| mean | 0.8983 | 0.8984 | 0.9325 | 0.0769 | 0.2608 |
| std | 0.0063 | 0.0061 | 0.0191 | 0.0014 | 0.0086 |

No batch-1 p95 latency was collected in this mini sweep because benchmark execution was not enabled. Across these three tested seeds, compact transformer performance appears stable: no class collapse was observed, macro F1 variance was low, FAILSAFE recall stayed above 0.91, and max predicted class share stayed well below the collapse threshold.

## Benchmark

RTX 5070 Ti Laptop GPU, fp16 inference, sequence length 128. Batch-1 latency is single-example latency. Batched throughput is not single-example latency.

| Model | Batch | Mean ms | p50 ms | p90 ms | p95 ms | Examples/sec |
|---|---:|---:|---:|---:|---:|---:|
| ReflexFormer-S | 1 | 1.0005 | 0.7384 | 1.1756 | 1.4686 | 999.54 |
| ReflexFormer-S | 4 | 0.7604 | 0.6249 | 1.2574 | 1.5090 | 5,260.48 |
| ReflexFormer-S | 16 | 1.0737 | 1.0355 | 1.1593 | 1.3212 | 14,902.07 |
| ReflexFormer-S | 64 | 3.4460 | 3.5086 | 3.6834 | 3.7387 | 18,572.29 |
| ReflexFormer-S | 256 | 12.5875 | 12.6606 | 13.6264 | 13.8944 | 20,337.70 |
| ReflexFormer-M | 1 | 1.5557 | 1.4320 | 2.3118 | 2.5467 | 642.80 |
| ReflexFormer-M | 4 | 1.4239 | 1.2858 | 1.8132 | 2.3587 | 2,809.11 |
| ReflexFormer-M | 16 | 2.9167 | 2.8305 | 3.2580 | 3.4900 | 5,485.64 |
| ReflexFormer-M | 64 | 10.0909 | 10.0589 | 10.5007 | 10.5945 | 6,342.34 |
| ReflexFormer-M | 256 | 39.7600 | 39.6642 | 40.1727 | 40.3023 | 6,438.63 |

Tokens/sec in benchmark output means input telemetry-token throughput, not language generation speed and not training speed.

The ReflexFormer-S benchmark table above is from the clean v1 reproduction pass. A prior stable smoke run measured batch-1 p95 at 2.3678 ms on the same GPU, so latency should be treated as run- and environment-dependent.

## Interpretation

- The temporal-hard v1 generator is less dominated by last-timestep and pure summary statistics than the earlier hard-mode generator.
- Sequence-aware models beat last-timestep and summary-stat baselines.
- On the clean seed-42 run, compact transformer models beat GRU: ReflexFormer-S macro F1 0.9230 and small Transformer-S baseline macro F1 0.9239 vs GRU macro F1 0.8825.
- ReflexFormer-M also beat GRU, but underperformed ReflexFormer-S.
- The small Transformer-S baseline and ReflexFormer-S v1 are effectively tied on this run; ReflexFormer-S is not uniquely best.
- No "bigger is better" claim is supported.
- The temporal derivative diagnostic is still very strong; claims about temporal reasoning should include derivative-aware probes and sequence baselines.
- No real-world drone safety claim is supported.

## Limitations

- This is simulation-only research on synthetic telemetry.
- No real-world drone safety claim is supported.
- No live-flight readiness, certified aviation safety, autopilot, obstacle-avoidance guarantee, actuator-control, motor-control, PX4, ArduPilot, MAVLink, DJI SDK, or deployment-readiness claim is supported.
- Synthetic-rule accuracy can overestimate model capability.
- The 3-seed mini sweep is a lightweight stability check, not a full robustness study.
- The mini sweep did not rerun GRU or the small Transformer-S baseline per seed, so average cross-seed superiority over those baselines is not established here.
- A model that predicts only `FAILSAFE_NEEDED` is collapsed and not useful, even if some safety metrics look favorable.

## Reproduction Commands

Generate the temporal-hard v1 smoke dataset:

```bash
python experiments/reflexformer_m/make_dataset.py \
  --seq-len 128 \
  --train-size 20000 \
  --val-size 3000 \
  --test-size 3000 \
  --seed 42 \
  --difficulty temporal_hard
```

Run diagnostics:

```bash
python experiments/reflexformer_m/dataset_diagnostics.py \
  --data-dir data/reflexformer_m
```

Run baselines:

```bash
python experiments/reflexformer_m/baselines.py \
  --data-dir data/reflexformer_m \
  --epochs 12 \
  --device auto \
  --normalize
```

Train ReflexFormer-S:

```bash
python experiments/reflexformer_m/train.py \
  --data-dir data/reflexformer_m \
  --out-dir checkpoints/reflexformer_s_v1 \
  --preset s \
  --epochs 20 \
  --batch-size 128 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --device auto \
  --class-weights none \
  --failsafe-weight 1.0 \
  --label-smoothing 0.02 \
  --patience 6 \
  --grad-clip 1.0 \
  --normalize \
  --add-deltas
```

Evaluate ReflexFormer-S:

```bash
python experiments/reflexformer_m/eval.py \
  --data-dir data/reflexformer_m \
  --checkpoint checkpoints/reflexformer_s_v1/best.pt \
  --device auto \
  --json-out results/reflexformer_s_v1_eval.json
```

Benchmark ReflexFormer-S:

```bash
python experiments/reflexformer_m/benchmark.py \
  --checkpoint checkpoints/reflexformer_s_v1/best.pt \
  --batch-sizes 1 4 16 64 256 \
  --seq-len 128 \
  --device auto \
  --dtype fp16 \
  --csv results/reflexformer_s_v1_benchmark.csv \
  --json results/reflexformer_s_v1_benchmark.json
```

Optional seed sweep:

```bash
python experiments/reflexformer_m/seed_sweep.py \
  --seeds 1 2 3 \
  --preset s \
  --difficulty temporal_hard \
  --train-size 12000 \
  --val-size 2000 \
  --test-size 2000 \
  --epochs 12 \
  --device auto \
  --out-csv results/reflexformer_s_temporal_hard_3seed_sweep.csv \
  --out-json results/reflexformer_s_temporal_hard_3seed_sweep.json
```

Comparison report:

```bash
python experiments/reflexformer_m/compare_results.py \
  --baseline-json results/reflexformer_m_baselines.json \
  --s-eval-json results/reflexformer_s_v1_eval.json \
  --m-eval-json results/reflexformer_m_stable_eval.json \
  --seed-sweep-json results/reflexformer_s_temporal_hard_3seed_sweep.json
```
