# PushT prediction history comparison

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

The GPU debug job runs the full regression suite, the actual-model history and
resume gate, eight diagnostic updates and two closed-loop planning episodes.
Only a passing diagnostic authorizes execution of the already approved formal
run. Training uses a full H100/H200 where available; evaluation uses the same
A100-SXM4-40GB model and locked runtime as the first comparison. It reuses the
same 64 goals, seeds and CEM/MPC settings. Existing NanoWM and one-frame RoboJEPA
results are retained; the new run has separate checkpoints and evaluation logs.

Worker phases are `PUSHT_PHASE=debug|train|evaluate` with
`scripts/pusht/prefix_worker.sh`. Inputs include the pinned code/external-source
archives, uv runtime, dataset/weights and original goal artifact. Training also
requires the passing debug marker in `approval/debug.complete`. Exit 85 returns
the complete resumable state. Any other failure stops subsequent phases.

Results are pending. No improvement in accuracy or latency is assumed.
