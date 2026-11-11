# PushT 10% training-window budget

Status: configuration prepared; no job, automation or W&B run has been created
for this budget. This document defines a proposed experiment, not a result.

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
| Proposed RoboJEPA prefix run | 160,000 | 4 | 640,000 | 10% |

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

The proposed timed-training cap is 86,400 seconds (24 hours), replacing the
pilot's four-hour cap so it cannot silently stop far below the target. If that
cap is reached before update 160,000, report the actual window count and mark
the 10% target incomplete; do not infer completion from process exit code zero.
The existing 1,700-second checkpoint/exit-85 cadence is retained. Persist latest
and best checkpoints and restore the same full training state after interruptions.

The tensor code, batch size and H200/A100 backend are unchanged and already
passed the recorded smoke and recovery gates. No additional GPU test is needed
merely to validate this configuration file. Any worker/controller changes for
executing the longer budget need their own bounded recovery check before launch.
In particular, the existing prefix worker/controller hardcodes the pilot config
and 3,000-step completion: do not submit it unchanged for this configuration.

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
