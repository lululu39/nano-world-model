# PushT MAE versus MSE

One additional MSE run is authorized on November 11, 2026. The comparison reuses
the completed prefix-history MAE experiment; it does not rerun that baseline or
change the ongoing 160,000-update MAE experiment. Results are pending.

At 22:37 UTC on November 11, recovery/smoke job `8393873.0` has completed:
57 tests passed, one skipped; pretrained tensors, causal/action alignment,
finite MSE gradients, exact eight-update exit-85 recovery, and two-goal
closed-loop execution passed. The diagnostic goals are not a model-quality
estimate. Formal job `8399845.0` was released at 22:34 UTC and is queued for
H200; formal training has not started. After training, the bounded controller
runs the matched 64-goal evaluation and stops.
[Validation evidence](../../reports/pusht/loss-ablation/validation.json) records
the checks and job identities. W&B run
[82312534](https://wandb.ai/yibozhong657-none/nanowm/runs/82312534) is created
only when formal training initializes.

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

Before formal training, run analytic loss/gradient checks, legacy-MAE checks,
pretrained loading and causal/action gates, eight diagnostic updates with an
actual exit-85 upload/restart and bitwise state comparison, and two diagnostic
closed-loop goals. Diagnostics do not create W&B runs or initialize formal
training. The controller stops on failure, retains a bounded debug shell, and
does not blindly resubmit or extend the budget.

Formal training uses H200 to match the previous MAE run. W&B project is `nanowm`,
entity `yibozhong657-none`, display name
`pusht-robojepa22m-prefix3-mse-3k-bs4-seed42`. The training cap is four hours;
the H200 pilot suggests approximately 20 minutes of training, excluding setup,
queueing, diagnostics and evaluation. The single-instance execution record is
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
