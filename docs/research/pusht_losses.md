# PushT MAE versus MSE

The authorized MSE experiment completed on November 11, 2026, at 23:18 UTC.
MSE succeeded on one fewer goal than MAE and reduced mean goal distance by
5.36%. This single-seed result does not establish a clearly superior loss.
The comparison reuses the completed prefix-history MAE experiment; it does
not rerun that baseline or change the ongoing 160,000-update MAE experiment.

Following this comparison, the maintainer chose MAE as the default for future
RoboJEPA fine-tuning. This is a workflow decision, not a claim of statistically
established superiority. The MSE configuration and results remain available
for reproducibility; no additional MSE run is scheduled.

| Objective | Success | Mean goal distance (lower is better) | Median planning latency | P95 planning latency |
| --- | --- | --- | --- | --- |
| MAE | 34/64 (53.125%) | 176.070528 | 1.067998 s | 1.070705 s |
| MSE | 33/64 (51.5625%) | 166.630342 | 1.087046 s | 1.096317 s |

Both models succeeded on 28 goals; six succeeded only with MAE, five only
with MSE, and 25 failed with both. Success differs by 1.5625 percentage points.
The measured MSE median latency is 1.78% higher. Timing uses the same GPU model
on different physical devices; this small difference does not establish a
compute-cost difference between the losses, which share the inference path.

Training job `8399845.0` completed 3,000 updates in 1,159.873 seconds
(19 minutes 20 seconds) on H200. Evaluation job `8405753.0` completed all
64 goals on A100-SXM4-40GB. Both jobs exited successfully. The bounded
controller has stopped and no additional run is scheduled.
W&B run: [82312534](https://wandb.ai/yibozhong657-none/nanowm/runs/82312534).

[Full comparison](../../reports/pusht/loss-ablation/reports/loss-comparison/comparison.json),
[CSV table](../../reports/pusht/loss-ablation/reports/loss-comparison/comparison.csv),
and [completion record](../../reports/pusht/loss-ablation/completion.json)
accompany the per-episode evaluation logs and all 3,000 training updates.

The MSE configuration is
[robojepa22m_prefix_mse_finetune.yaml](../../configs/pusht/robojepa22m_prefix_mse_finetune.yaml).
Both models start from the same official RoboJEPA-22M cooldown-150k predictor
and frozen V-JEPA 2.1-G encoder. They use seed 42, batch four, 3,000 updates,
the same 17,750/935 training/dev episode split, identical action normalization
and chronological five-action chunks, and the same optimizer settings. This is
12,000 window presentations, or 0.1875% of the official SD-VAE NanoWM training
exposure denominator; it is not a subset percentage of unique trajectories.

Both teacher-forced prediction and three-step free rollout retain their causal
prefixes, and their losses are equally weighted. The sole optimization change
is replacing elementwise absolute error with squared error. MSE gradients are
not rescaled, and learning rates, warmup, clipping and validation cadence remain
unchanged. Native MAE and MSE loss values must not be compared numerically as a
measure of which method performs better.

Existing checkpoints/configurations without `loss_type` continue to mean MAE.
The new explicit `loss_type: mse` is stored in checkpoints; exact resume rejects
a changed configuration. Metrics use `teacher_mse` and `rollout_mse`, so MSE
values are never mislabeled as MAE.

## Validation and execution

Before formal training, 57 tests passed and one was skipped. Pretrained
loading (106 predictor tensors), image/spatial-token and causal/action gates,
finite MSE gradients, eight diagnostic updates with an actual exit-85
upload/restart and bitwise state comparison, and two diagnostic closed-loop
goals passed. The diagnostic goals are not a model-quality estimate.
[Validation evidence](../../reports/pusht/loss-ablation/validation.json) records
the checks and job identities. Diagnostics created no W&B run and did not
initialize formal training. No failure retry or budget extension was used.
Future launches must perform smoke checks and formal training in the same
GPU allocation, as required by the updated repository instructions.

Formal training uses H200 to match the previous MAE run. W&B project is `nanowm`,
entity `yibozhong657-none`, display name
`pusht-robojepa22m-prefix3-mse-3k-bs4-seed42`. The four-hour training cap was
not reached; the run stopped at the authorized 3,000 updates. Reported training
time excludes setup, queueing, diagnostics and evaluation. The execution record is
`world-models/jobs/pusht-mse/state.json`.

## Comparison protocol

Evaluate both models at update 3,000. The existing MAE evaluated checkpoint is
its step-3,000 best checkpoint, which also corresponds to its final update. For
MSE, evaluate `latest.pt` at step 3,000 regardless of which checkpoint minimizes
MSE dev loss. This avoids selecting different training budgets using different
dev objectives. Best checkpoints are still saved separately.

Use the existing 64 official-val goals, seeds, 100-physical-action limit,
three-chunk horizon and replanning interval, and CEM 64 candidates / 10 elites /
five iterations. The planner keeps its existing latent MSE cost for both models.
Evaluate on A100-SXM4-40GB with the same runtime, synchronized timing and warmup
exclusion. Report GPU model equality without claiming the same physical device.

The primary results are planning success, original seven-dimensional state L2
goal distance, and planning latency. The existing MAE reference is 34/64
(53.125%), distance 176.070528, median 1.067998 s and P95 1.070705 s. The
[comparison script](../../scripts/pusht/compare_losses.py) checks training recipe,
split, checkpoint step, full finite update history, goals, protocol and runtime,
then independently recomputes episode aggregates and paired success counts.
One training seed and 64 tasks cannot establish general superiority of a loss.
