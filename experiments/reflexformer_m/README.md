# ReflexFormer-M

ReflexFormer-M is a simulation-only PyTorch experiment for fast transformer-based drone safety-advisory research. It consumes synthetic structured telemetry sequences and predicts one of four conservative advisory classes:

- `LOW_RISK`
- `CAUTION`
- `AVOIDANCE_NEEDED`
- `FAILSAFE_NEEDED`

It is not a drone autopilot and does not emit actuator, motor, navigation, PX4, ArduPilot, MAVLink, DJI SDK, or live-flight commands.

## Why A Transformer

This experiment uses a compact transformer encoder because attention can model interactions across the full telemetry window: obstacle trend, attitude instability, battery decline, waypoint error, wind, and controller error can all affect the advisory label. This is a different research direction from spiking neural networks; it favors standard PyTorch training, simple batching, and direct latency benchmarking over event-driven dynamics.

## Dataset

`make_dataset.py` creates synthetic telemetry tensors under `data/reflexformer_m/`:

- `train.pt`
- `val.pt`
- `test.pt`
- `metadata.json`

Each example has shape `[seq_len, 29]`, with default `seq_len=128`. Labels are generated from synthetic scenario templates plus random noise. The templates cover stable flight, caution states, obstacle/trajectory risk, and failsafe-like synthetic hazards such as critically low battery, severe attitude instability, altitude loss, controller error, or sensor corruption.

Training and evaluation default to train-split feature normalization with clipped normalized values. Use `--no-normalize` to disable it. `--add-deltas` concatenates first-difference channels, so the model sees both telemetry values and `x[t] - x[t-1]`; this doubles the feature dimension from 29 to 58 and helped the temporal-hard smoke runs.

Difficulty modes:

- `easy`: original separable synthetic templates. Perfect F1 is expected and is not meaningful evidence of real-world capability.
- `medium`: overlapping feature distributions, 3% label noise, corrupted windows, delayed-risk cases, weak distractors, and mixed-risk scenarios.
- `hard`: stronger overlap, 7% label noise, more corruption, and more temporal cases where risk has to be inferred from sequence evolution.
- `temporal_hard`: 5% label noise with paired order/trend/recovery cases. Label-driving channels are distribution-equalized to reduce final-timestep and pure-summary leakage; labels depend more on event order, trend direction, delayed consequences, and recovery vs non-recovery.

Generate the default full dataset:

```bash
python experiments/reflexformer_m/make_dataset.py \
  --seq-len 128 \
  --train-size 80000 \
  --val-size 10000 \
  --test-size 10000 \
  --seed 42 \
  --difficulty easy
```

Generate a harder smoke dataset:

```bash
python experiments/reflexformer_m/make_dataset.py \
  --seq-len 128 \
  --train-size 12000 \
  --val-size 2000 \
  --test-size 2000 \
  --seed 42 \
  --difficulty hard
```

Generate a temporal-hard smoke dataset:

```bash
python experiments/reflexformer_m/make_dataset.py \
  --seq-len 128 \
  --train-size 20000 \
  --val-size 3000 \
  --test-size 3000 \
  --seed 42 \
  --difficulty temporal_hard
```

## Model

Default architecture:

- Transformer encoder classifier
- Sequence length: 128
- Feature dimension: 29
- `d_model=384`
- `n_layers=10`, configurable 8-12
- `n_heads=6`
- MLP ratio: `3.0`, configurable 2.0-3.0
- Learned CLS token and learned positional embeddings
- GELU activations
- Output logits over four advisory classes

The default model is in the requested roughly 10M-25M parameter range.

Smoke test:

```bash
python experiments/reflexformer_m/model.py
```

## Training

```bash
python experiments/reflexformer_m/train.py \
  --data-dir data/reflexformer_m \
  --out-dir checkpoints/reflexformer_m \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4 \
  --device auto
```

The trainer uses AdamW, cosine learning-rate decay, CUDA mixed precision when available, and CPU fallback. It saves:

- `checkpoints/reflexformer_m/best.pt`
- `checkpoints/reflexformer_m/last.pt`

The best checkpoint is selected by a balanced score that combines validation macro F1 and `FAILSAFE_NEEDED` recall while penalizing class collapse, high max predicted class share, and low per-class recall floors. A checkpoint is not eligible as best if it uses fewer than three predicted classes, has an empty predicted class, exceeds 70% max predicted class share, or has macro F1 below 0.30.

Stable temporal-hard training for ReflexFormer-S:

```bash
python experiments/reflexformer_m/train.py \
  --data-dir data/reflexformer_m \
  --out-dir checkpoints/reflexformer_s_stable \
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

Stable temporal-hard training for ReflexFormer-M:

```bash
python experiments/reflexformer_m/train.py \
  --data-dir data/reflexformer_m \
  --out-dir checkpoints/reflexformer_m_stable \
  --preset m \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-4 \
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

The temporal-hard split is balanced, so aggressive class weights and failsafe weighting are not the default. Failsafe recall must be interpreted with max predicted class share and the confusion matrix; an all-failsafe model has high failsafe recall and low miss rate but is collapsed and not useful.

Overfit tests catch basic architecture or training-loop failures:

```bash
python experiments/reflexformer_m/overfit_test.py \
  --data-dir data/reflexformer_m \
  --preset s \
  --device auto
```

## Evaluation

```bash
python experiments/reflexformer_m/eval.py \
  --data-dir data/reflexformer_m \
  --checkpoint checkpoints/reflexformer_m/best.pt \
  --device auto
```

Evaluation reports test accuracy, macro F1, per-class precision/recall/F1, confusion matrix, invalid output rate, max predicted class share, and class distribution.

## Baselines

Run classical baselines before making model-quality claims:

```bash
python experiments/reflexformer_m/baselines.py \
  --data-dir data/reflexformer_m \
  --epochs 12 \
  --device auto \
  --normalize
```

The baseline script reports:

- logistic regression on the flattened sequence
- random forest on summary statistics
- histogram gradient boosting on summary statistics
- an optional last-timestep MLP baseline
- sequence-aware TemporalCNN, GRU, and small Transformer-S baselines

These comparisons help detect whether ReflexFormer-M is learning temporal structure or mostly exploiting simple aggregate features.

Run dataset diagnostics before baselines:

```bash
python experiments/reflexformer_m/dataset_diagnostics.py \
  --data-dir data/reflexformer_m
```

Diagnostics report class balance, label noise from `clean_y`, last-timestep probe signal, pure-summary probe signal, temporal slope/derivative probe signal, and nearest-summary paired counterexample candidates. If pure summary-stat baselines beat the transformer, that is a red flag: the generator may leak labels through aggregate feature distributions rather than requiring temporal reasoning. Model-quality claims require beating simple baselines and cheaper sequence-aware baselines on the same split.

## Ablations

Small ablations can compare sequence length, depth, width, and pooling:

```bash
python experiments/reflexformer_m/ablate.py \
  --data-dir data/reflexformer_m \
  --quick \
  --device auto
```

## Benchmark

```bash
python experiments/reflexformer_m/benchmark.py \
  --checkpoint checkpoints/reflexformer_m/best.pt \
  --batch-sizes 1 4 16 64 256 \
  --seq-len 128 \
  --device auto \
  --dtype fp16 \
  --csv results/reflexformer_m_benchmark.csv
```

Optional:

```bash
python experiments/reflexformer_m/benchmark.py --compile
```

Benchmark definitions:

- Batch-1 latency is single-example latency.
- Batched throughput is not single-example latency.
- Tokens/sec means input telemetry-token throughput, where tokens = `batch_size * seq_len`.
- Tokens/sec is not language generation speed and not training speed.
- p50/p90/p95 summarize repeated inference calls and are usually more informative than a single timing number.
- Peak CUDA memory is reported when CUDA is available.
- Latency depends on hardware, PyTorch version, dtype, thermals, and whether CUDA is available.

## Expected Results

Current easy-mode smoke result on an RTX 5070 Ti Laptop GPU:

- Parameters: 15,003,268
- Batch-1 latency: 2.0185 ms with fp16
- Easy synthetic test accuracy / macro F1: 1.0000 / 1.0000
- No class collapse

Perfect F1 on easy synthetic data is expected because the original templates are highly separable. It should not be used as a model-quality claim. Hard-mode results and baseline comparisons are required before making even simulation-only model-quality claims.

Historical hard-mode result before temporal-hard redesign:

- ReflexFormer-M hard-mode test accuracy / macro F1: 0.6900 / 0.6804
- `FAILSAFE_NEEDED` recall: 0.3219
- RandomForest(summary) and HistGradientBoosting(summary): macro F1 0.9305
- Interpretation: the hard generator was still dominated by summary statistics and was not a good temporal-reasoning benchmark.

Temporal-hard diagnostic result on the 20k/3k/3k smoke split:

| Probe | Macro F1 |
|---|---:|
| last timestep only | 0.3227 |
| summary statistics | 0.4758 |
| temporal slope/derivative | 0.9390 |

Stable temporal-hard smoke results on the same split:

| Model | Test Macro F1 | FAILSAFE Recall | Safety-Critical Miss Rate | Max Pred Share |
|---|---:|---:|---:|---:|
| LogisticRegression(flattened) | 0.7718 | 0.7997 | not reported | 0.2580 |
| RandomForest(summary) | 0.2728 | 0.2984 | not reported | 0.2727 |
| HistGradientBoosting(summary) | 0.4729 | 0.5040 | not reported | 0.2793 |
| MLP(last timestep) | 0.2874 | 0.2984 | not reported | 0.3553 |
| TemporalCNN(sequence) | 0.8622 | 0.8581 | not reported | 0.2823 |
| GRU(sequence) | 0.8910 | 0.9191 | not reported | 0.2530 |
| small Transformer-S baseline | 0.9251 | 0.9231 | not reported | 0.2587 |
| ReflexFormer-S stable | 0.9230 | 0.9324 | 0.0475 | 0.2550 |
| ReflexFormer-M stable | 0.9130 | 0.9257 | 0.0596 | 0.2537 |

On this smoke split, ReflexFormer-S and the small Transformer-S baseline beat the GRU baseline, while ReflexFormer-M did not beat ReflexFormer-S. This is a simulation-only result and may change with seeds, dataset size, and tuning.

Stable fp16 batch-1 benchmark results on an RTX 5070 Ti Laptop GPU:

| Model | Params | Mean ms | p95 ms |
|---|---:|---:|---:|
| ReflexFormer-S stable | 3,278,340 | 1.4119 | 2.3678 |
| ReflexFormer-M stable | 15,014,404 | 1.5557 | 2.5467 |

## Safety Limitations

This experiment is research simulation only. Any real drone safety system would need deterministic safety control, formal requirements, real sensor validation, adversarial and environmental testing, certification processes, and human oversight. ReflexFormer-M provides advisory class logits only.
