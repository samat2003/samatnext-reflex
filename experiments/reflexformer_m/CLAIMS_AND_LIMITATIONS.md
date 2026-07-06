# ReflexFormer-M Claims And Limitations

ReflexFormer-M is a research simulation experiment only.

## Explicit Non-Claims

- No live-flight readiness claim.
- No certified aviation safety claim.
- No real autopilot claim.
- No direct actuator-control claim.
- No motor-control, PX4, ArduPilot, MAVLink, DJI SDK, weapon, or live-flight integration claim.
- No obstacle-avoidance guarantee.
- No real-world generalization claim.
- No claim that synthetic accuracy implies real drone safety.

## What The Model Does

The model consumes synthetic telemetry sequences and predicts conservative advisory classes:

- `LOW_RISK`
- `CAUTION`
- `AVOIDANCE_NEEDED`
- `FAILSAFE_NEEDED`

These are labels for simulation research, not commands. A real-world system would require a deterministic safety controller, certification, extensive testing, operational constraints, and human oversight.

## Dataset Limits

The dataset is generated from deterministic synthetic rules plus random noise. It is useful for controlled model, training, and latency experiments, but it is not evidence of performance on real drones, real sensors, real operators, real weather, or real obstacle fields.

## Benchmark Limits

- Latency depends on hardware, software versions, dtype, thermal state, and batch size.
- Batch-1 latency is the only reported single-example latency.
- Batched throughput must not be described as batch-1 latency.
- Tokens/sec is input telemetry-token throughput, not LLM generation speed.
- Tokens/sec is not training speed.
- Benchmarks time inference only unless explicitly stated otherwise.

## Accuracy Limits

Synthetic test accuracy measures how well the model learned this generator's rules. It does not prove safety, reliability, regulatory compliance, or obstacle avoidance in the real world.

Easy-mode synthetic-rule accuracy can substantially overestimate model capability because the class templates are intentionally simple and separable. Hard-mode evaluation and baseline comparisons are required before making model-quality claims, even within this simulation-only setting. A transformer result is only meaningful if it is compared against simpler baselines such as logistic regression, random forests, gradient boosting, and last-timestep MLPs on the same synthetic splits.

High synthetic accuracy can also be caused by generator leakage: labels may be recoverable from final timesteps, min/max/mean statistics, or other artifacts that do not demonstrate temporal reasoning. Summary-stat baselines and last-timestep probes must be compared before claiming that a model learned sequence dynamics.

For safety-advisory research, `FAILSAFE_NEEDED` recall and safety-critical miss rate matter more than raw accuracy, but they are not sufficient alone. They must be reported with macro F1, per-class recall, predicted class distribution, max predicted class share, and confusion matrices. A model that predicts `FAILSAFE_NEEDED` for every input has perfect failsafe recall and zero safety-critical miss rate, but it is collapsed and not a useful advisory classifier.

Failsafe recall alone must not dominate checkpoint selection or model claims. Balanced checkpointing should reject collapsed models, empty predicted classes, excessive max predicted class share, and very low per-class recall, even when the failsafe recall looks high.
