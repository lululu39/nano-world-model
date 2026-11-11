# PushT 10% training-window budget

Status at 20:48 UTC on November 11, 2026: recovery diagnostic `8358066.0`
passed, and formal job `8375071.0` was audited while held and released. It is
queued, with no formal training updates observed yet. Worker source remains
`0e29f14`; controller source is `703a10e`. The 160,000-update budget and run ID
are unchanged. This document defines the experiment budget, not a result.

The denominator is the official NanoWM PushT checkpoint's cumulative training
window exposure: 100,000 updates, eight GPUs and batch eight per GPU, giving
6,400,000 four-frame windows. The checkpoint's actual batch/accumulation settings
are recorded in [weight-loading.json](../../reports/pusht/nanowm-official/weight-loading.json).
The eight-GPU training setup is stated in
[NanoWM section 4.6](https://arxiv.org/html/2605.23993v1#S4.SS6) and the
[released launcher](../../src/scripts/train/main/dino_wm_pusht.sh).

| Budget | Optimizer updates | Global batch | Window presentations | Fraction of NanoWM exposure |
|---|---:|---:|---:|---:|
| Official NanoWM | 100,000 | 64 | 6,400,000 | 100% |
| Completed RoboJEPA prefix pilot | 3,000 | 4 | 12,000 | 0.1875% |
| Authorized RoboJEPA prefix run | 160,000 | 4 | 640,000 | 10% |

The earlier pilot was **0.1875%**, not 0.8%, by this denominator. Window
presentations are not unique trajectories or unique images. This configuration
does not select a 10% subset of the dataset and does not add another data source.
It uses the same 17,750-episode training pool, with 935 episodes reserved for
internal validation, but processes 53.33 times as many training windows. Each
window contains four images at five-step intervals and three ordered five-action
chunks. The target corresponds to 2,560,000 frame presentations and 1,920,000
action-conditioned transitions. External encoder/predictor pretraining is not
included in this budget comparison.

## Configuration

Use [robojepa22m_prefix_10pct_finetune.yaml](../../configs/pusht/robojepa22m_prefix_10pct_finetune.yaml).
The training algorithm is unchanged from the completed prefix experiment:

- Official pretrained RoboJEPA-22M predictor and frozen V-JEPA encoder.
- One actual observation followed by retained predicted history, at most three
  input frames; no KV cache, RGB decoder or reconstruction loss.
- Seed 42, batch four, predictor LR 5e-5, adapter LR 5e-4, AdamW weight decay
  0.01, 100-update warmup, gradient clipping at 1.0 and BF16.
- Equal weights for teacher-forced and autoregressive feature MAE.
- Same data order/split, action scaling and normalization, preprocessing, and 64
  fixed dev windows. Validate every 250 updates and select best by dev loss.
- Same 64 official-val planning goals and shared MPC/CEM protocol; these goals
  are excluded from training and checkpoint selection.
- W&B entity `yibozhong657-none`, project `nanowm`, a separate run identity.

Start a new run from the released weights in an empty output directory. The
completed 3,000-update checkpoint is not an exact resume for a modified training
budget: the existing resume contract requires the same configuration and run ID.
Once this new run is initialized, its interruptions resume model, optimizer, RNG
and batch position under the same identity and configuration.

## Time and execution

The measured H200 prefix pilot took 1,146.39 seconds for 3,000 updates. Linear
extrapolation gives approximately **17 H200 GPU-hours** for 160,000 updates.
This is an estimate, not a measured long-run throughput result. Queueing,
input/environment setup, restart initialization, archive transfers and final
planning evaluation add wall time.

The authorized timed-training cap is 86,400 seconds (24 hours), replacing the
pilot's four-hour cap so it cannot silently stop far below the target. If that
cap is reached before update 160,000, report the actual window count and mark
the 10% target incomplete; do not infer completion from process exit code zero.
The existing 1,700-second checkpoint/exit-85 cadence is retained. Persist latest
and best checkpoints and restore the same full training state after interruptions.

The tensor code, batch size and H200/A100 backend are unchanged. The worker now
accepts `PUSHT_TRAIN_CONFIG=configs/pusht/robojepa22m_prefix_10pct_finetune.yaml`.
Its `recovery` phase compares eight uninterrupted diagnostic updates with four
updates followed by an actual Condor exit-85 archive transfer and restart, then
four more updates. Model, optimizer, RNG, data position and every update metric
must match exactly. Diagnostics never initialize formal training or create a
W&B run. Formal execution requires the passing diagnostic marker.

The bounded controller reads the step budget from its experiment state, pins
resumed training to the GPU model used at startup, and submits the original
64-goal shared evaluation only after all 160,000 updates complete. Training
summaries explicitly report `completed`, `target_steps`, `windows_seen` and
`stop_reason`; hitting the time cap does not count as completing 10%.

The trainer accepts the prepared configuration directly inside the established
compute-node runtime:

```bash
python scripts/pusht/train.py \
  --config configs/pusht/robojepa22m_prefix_10pct_finetune.yaml \
  --assets assets --output results/robojepa-prefix-10pct \
  --run-id <new-persistent-run-id>
```

The shell must supply the same pinned weights and external-source locations as
the completed experiment. This example is not a scheduler submission command.

## Submitted execution

The persistent run identity is `854d4fd1` in project `nanowm`, entity
`yibozhong657-none`. The [W&B run](https://wandb.ai/yibozhong657-none/nanowm/runs/854d4fd1)
is created only when formal training starts. Diagnostics do not use that run.
The local execution record is `world-models/jobs/pusht-10pct/state.json`, with
`controller.log`, held-submission audits and per-phase scheduler logs beside it.
The packaged training source remains fixed; the controller-only correction
is recorded separately below.
It stops on failure or the stated training cap and cannot create another formal
run or enlarge the budget. Actual completion and planning results are pending.

## Recovery check and controller correction

The H200 diagnostic passed 44 tests (one optional test skipped), strict loading
of 106 pretrained predictor tensors, action/image/token and causal-history
checks, and eight diagnostic updates. Four updates followed by exit 85,
checkpoint upload and process restart produced exactly the same model,
optimizer, RNG, data position and update metrics as eight uninterrupted updates.
The [recovery report](../../reports/pusht/ten-percent/recovery/recovery-check.json)
and [transfer evidence](../../reports/pusht/ten-percent/recovery/transfer-verification.json)
are preserved with the scheduler log and hardware checks.

The first controller incorrectly required `NumJobStarts >= 2`. The actual
checkpoint restart occurred in the same sandbox and allocation, so that count
remained one. This is expected behavior for
[HTCondor self-checkpointing](https://htcondor.readthedocs.io/en/main/users-manual/self-checkpointing-applications.html).
The controller halted after the diagnostic, before submitting any formal run.
The corrected check requires completed checkpoint and final output transfers,
the exit-85 configuration and both checkpoint payloads, in addition to all the
unchanged bitwise numerical checks. Five stdlib scheduler regression tests pass,
including the recorded single-allocation case and missing-transfer failures.
No GPU diagnostic or formal run was duplicated. This verifies an actual process
restart and upload, not eviction onto a different node.
