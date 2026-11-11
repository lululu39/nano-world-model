# PushT prediction history comparison

Completed on November 11, 2026. Retaining the predicted prefix increased success
from 32 to 34 of the same 64 tasks and reduced mean goal distance by 21.54%, while
increasing median planning time by 31.25%. These are single-seed observations;
the two additional successes do not establish a statistically reliable advantage.

| Method | Success | Mean goal distance | Planning median / P95 (s) |
|---|---:|---:|---:|
| Official NanoWM PushT checkpoint | 31/64 (48.44%) | 156.759 | 31.409 / 31.439 |
| RoboJEPA-22M, previous frame only | 32/64 (50.00%) | 224.400 | 0.814 / 0.828 |
| RoboJEPA-22M, retained predicted prefix | 34/64 (53.13%) | 176.071 | 1.068 / 1.071 |

All rows use the same 64 goals, episode seeds, 100-physical-action budget,
64-candidate / 10-elite / 5-iteration CEM, three-chunk horizon and replanning
interval, and locked dependencies. Latency is the CUDA-synchronized complete CEM
call including online image encoding; warmup and simulator/rendering are excluded.
All evaluations used A100-SXM4-40GB, but the new evaluation ran on a different
physical GPU (gpulab2005 rather than gpulab2004). Goal distance is the original
seven-dimensional state L2 with mixed position, angle and velocity units, not
meters or latent-space distance. NanoWM still has the lowest mean goal distance.
Its median latency is 29.41 times the prefix model's measured latency.

Both RoboJEPA runs used seed 42, 3,000 batch-four updates and the same train/dev
split and optimizer recipe. The prefix run completed in 1,146.39 seconds on H200;
the initial run took 1,144.60 seconds. Its best dev loss was 0.09438245 versus
0.09553591, with best/latest both at step 3,000. The full predictor was initialized
from the official checkpoint again; the encoder stayed frozen. Planning used the
strictly reloaded best checkpoint, including its three-frame history contract.

A [proposed 10% training-window budget](pusht_10pct.md) defines a larger run using
the same algorithm. Only its configuration has been prepared; it is not a new
experimental result or a submitted job.

The NanoWM row is reused from the completed initial comparison. Differences in
representations, model sizes and external pretraining remain; this is not an
isolated diffusion-versus-deterministic architecture comparison. The initial
paper-protocol limitations in [pusht_results.md](pusht_results.md) still apply.

## Implementation and protocol

This follow-up changes only RoboJEPA's access to prediction history. Both methods
still begin each CEM rollout with one actual observation. The new RoboJEPA run
retains that observation and its own predictions: inputs are `[z0]`,
`[z0, predicted_z1]`, then `[z0, predicted_z1, predicted_z2]`. Each input position
receives its outgoing five-action chunk. No future observations enter free rollout.

`context_frames: 1` describes the observed context;
`rollout_context_frames: 3` describes the maximum prediction history. Older
configurations/checkpoints default to one-frame history. Evaluation reads the
history contract from the trained checkpoint rather than overriding it. The
pretrained predictor retains its native eight-frame causal attention window.
Full prefixes are recomputed with KV caching disabled, keeping gradient feedback
and cache semantics unchanged from the initial experiment.

Teacher forcing processes the three source frames together under the native
causal mask. Its targets are the following three frames. Free rollout starts
from only the first observed frame and retains predicted history. The two MAE
losses remain equally weighted. Tests check parallel versus prefix teacher
forcing, future-input causality, earlier-frame/action sensitivity, exact action
ordering, train/planning equivalence and checkpoint recovery.

The authorized run uses `configs/pusht/robojepa22m_prefix_finetune.yaml`: the same
official pretrained 22M predictor, frozen encoder, seed 42, batch 4, 3,000 updates,
learning rates, normalization, train/dev episode split, and fixed dev windows.
It starts from released weights, not from the initial PushT checkpoint or a
diagnostic checkpoint. The initial run and its outputs remain intact.

The GPU debug job ran the full regression suite, the actual-model history and
resume gate, eight diagnostic updates and two closed-loop planning episodes.
Formal training began only after all diagnostics passed. Training used H200;
evaluation used the same
A100-SXM4-40GB model and locked runtime as the first comparison. It reuses the
same 64 goals, seeds and CEM/MPC settings. Existing NanoWM and one-frame RoboJEPA
results are retained; the new run has separate checkpoints and evaluation logs.

Worker phases are `PUSHT_PHASE=debug|train|evaluate` with
`scripts/pusht/prefix_worker.sh`. Inputs include the pinned code/external-source
archives, uv runtime, dataset/weights and original goal artifact. Training also
requires the passing debug marker in `approval/debug.complete`. Exit 85 returns
the complete resumable state. Any other failure stops subsequent phases.

The H200 diagnostic completed successfully: 44 tests passed and one optional
RT-1 test was skipped. The actual pretrained predictor passed future-input
causality, parallel/prefix equivalence, historical feature/action sensitivity,
physical action replay, frozen-encoder gradients and bitwise next-update resume.
Eight diagnostic updates and two closed-loop episodes completed. The diagnostic
checkpoint is excluded from formal initialization. See
[debug evidence](../../reports/pusht/prefix-history/debug).

The H200 training and A100 evaluation both exited successfully on their first
allocation. The actual-model gate also passed on A100 before evaluation. All
3,000 training losses/gradients were finite. The controller exited after the
single authorized run and evaluation; no further runs were submitted.

## Reproducibility

- [Three-method JSON/CSV table and independent checks](../../reports/pusht/prefix-history/comparison)
- [Actual training config, all update logs and loading/provenance](../../reports/pusht/prefix-history/training)
- [All 64 episode records, targets, protocol and runtime](../../reports/pusht/prefix-history/reports/robojepa-prefix)
- [A100 gate](../../reports/pusht/prefix-history/reports/hardware-gate.json)
- [Jobs, source revision, physical GPUs and checkpoint location](../../reports/pusht/prefix-history/experiment.json)
- [Completed W&B training run](https://wandb.ai/yibozhong657-none/nanochat/runs/2b32b1fd)

The submitted source was `e21cc3f`. This existing run remains in its original
W&B project `nanochat`; future runs default to `nanowm`. The actual resolved
training config is preserved with the results. Regenerate the comparison using:

```bash
python scripts/pusht/compare_history.py \
  --previous reports/pusht/shared-benchmark \
  --current reports/pusht/prefix-history \
  --output reports/pusht/prefix-history/comparison
```

This checks matched training recipes and train/dev splits, every recorded
training update, and identical evaluation goals/protocols/GPU models/dependencies.
It independently recomputes successes, mean goal distance and median latency
from per-episode records.
