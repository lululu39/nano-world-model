# PushT pretrained-model milestones

The immediate scope is the released NanoWM PushT checkpoint followed by pretrained
RoboJEPA-22M transfer. No new baseline, decoder, reconstruction objective, or broad
architecture changes are part of this experiment.

## Protocol provenance

[NanoWM section 4.4 / Table 4](https://arxiv.org/html/2605.23993v1#S4.SS4)
reports 25% success with relative actions, frame interval 5, goal horizon 3,
64 CEM candidates, and five optimization iterations. The SD-VAE model has four
latent channels on a 32 x 32 grid and one context plus three generated frames.

The paper does **not** specify its planning episode count, seeds, elite count,
maximum action budget, or complete planning sampler settings. Its section 4.1
gives general defaults of 250 DDIM steps and sequential scheduling, while the
released planning documentation recommends 20 steps and full-sequence scheduling.
These differences prevent an assertion that an exact original Table 4 run has
been recovered. We do not search seeds or change thresholds to obtain 25%.

- `configs/pusht/paper_reproduction.yaml` records the published settings, the
  released planning entrypoint's 20-step full-sequence sampler, and every additional
  assumption. The application-specific entrypoint takes precedence over the
  general video-generation defaults. Results are a
  reproduction attempt with incomplete protocol provenance.
- `configs/pusht/shared_benchmark.yaml` uses the same goals and physical action
  budget, with the explicitly documented 20-step full-sequence diffusion sampler.
  Both models share the exact simulator, CEM implementation and settings, fixed
  goal artifact, episode seeds, and success criterion.

The reference 25% is not an engineering test threshold. Neither a small smoke
subset nor a confidence interval containing 25% establishes exact reproduction.

## Assets and interfaces

`scripts/pusht/prepare.py` stages the official OSF PushT train/validation archive,
NanoWM config and safetensors, and only the encoder/quantization portion of the
SD-VAE. It records resolved model revisions and file sizes, without hashing file
contents. `load_nanowm` constructs the model from the checkpoint configuration,
strictly loads all model tensors, and records every weight shape. The VAE decoder
is never instantiated. Its encoder preserves posterior sampling and scaling.

Physical action `a[t]` is the relative target displacement from the current agent
position during source frame `t`. Five successive two-dimensional displacements
form `[dx0,dy0,...,dx4,dy4]`, connecting observations at `t` and `t+5`. Dataset
displacements are divided by 100, then standardized with training-only means and
sample standard deviations plus 1e-6. The simulator multiplies the denormalized
action by 100. Both models use the same ten-dimensional planner variables and
normalization statistics; no chunk averaging, reversal, or future-state input is
allowed.

RoboJEPA starts with the released 22M cooldown-150k predictor and V-JEPA 2.1-G
target encoder. Loading compares every predictor tensor with the released tensor
before any optimizer update. The encoder remains frozen; a linear 10-to-64 action
adapter and the pretrained predictor are trained. Embodiment slot 2 and camera
slot 0 are explicit transfer initializations, not native PushT semantics. State
tokens receive zeros during training and all imagined rollouts.

PushT observations are resized to 256 x 256 with bilinear antialiasing, without
random crop or flip. RGB floats in [0,1] receive ImageNet channel normalization.
Each image is encoded separately; tokens retain row-major spatial order on a
16 x 16 grid, followed by per-token layer normalization across 1664 channels.
The same preprocessing is used for video training frames and live simulator
observations. The released RoboJEPA convenience video utility quantizes resized
frames to uint8; this transfer pipeline explicitly uses float preprocessing
throughout and verifies its numerical consistency rather than silently mixing
those two paths.

## Execution

Use the committed uv lock with Python 3.11 and `uv sync --frozen --extra pusht`.
All tensor work, simulator tests, downloads, and training run on compute nodes.
The preparation allocation remains available for shell debugging after a failure.

```bash
python scripts/pusht/prepare.py
python -m pytest -q tests/latent/test_pusht_contract.py
python scripts/pusht/verify.py --assets assets --output verification.json

python scripts/pusht/evaluate.py \
  --config configs/pusht/shared_benchmark.yaml --assets assets \
  --goals goals/pusht-h3-seed42.pt --output results/nanowm-smoke --limit 2

python scripts/pusht/evaluate.py \
  --config configs/pusht/paper_reproduction.yaml --assets assets \
  --goals goals/pusht-h3-seed42.pt --output results/nanowm-paper

python scripts/pusht/train.py --config configs/pusht/robojepa22m_finetune.yaml \
  --assets assets --output results/robojepa-smoke --smoke
python scripts/pusht/train.py --config configs/pusht/robojepa22m_finetune.yaml \
  --assets assets --output results/robojepa-train

python scripts/pusht/evaluate.py \
  --config configs/pusht/shared_benchmark.yaml --assets assets \
  --goals goals/pusht-h3-seed42.pt --output results/robojepa-shared \
  --model robojepa --checkpoint results/robojepa-train/best.pt

python scripts/pusht/summarize.py --nanowm results/nanowm-shared \
  --robojepa results/robojepa-shared --output results/comparison
```

Paths to RoboJEPA/V-JEPA source and weights can be supplied through the explicit
CLI flags. Diagnostic checkpoints never initialize the formal transfer run.

The initial training budget is one seed, at most 3,000 updates of batch size four
and at most four hours of timed training, without an automatic extension or sweep.
The internal validation set is a fixed 5% of the official training episodes;
official validation goals never select checkpoints. Frozen encodings are computed
online to avoid an unbounded multi-hundred-GB feature cache. Only trainable model
weights, optimizer, batch order, RNG, and run identity are checkpointed. The frozen
encoder is restored from the separately pinned pretrained artifact.

Training enables deterministic PyTorch algorithms and the cuBLAS workspace policy.
The actual 256-resolution pretrained gate found a maximum 6.23e-7 parameter
difference after resume with the default nondeterministic GPU backward kernels,
despite identical loss. Enabling deterministic algorithms restored bitwise-identical
next-update parameters; the acceptance criterion was not relaxed.

Training and evaluation return 85 when a persistence boundary is reached. The
worker must archive and persist outputs before propagating that code. Evaluation
resumes at episode boundaries with per-episode RNG reset; every completed episode
is saved atomically. Training resumes `latest.pt`, including optimizer and data
position, under the same configuration and W&B run ID.
Batch submissions set a persistent `PUSHT_RUN_ID` before first execution, so an
early eviction before the first checkpoint transfer cannot create a second W&B
run. Updates since the last durable checkpoint may need deterministic replay;
only transferred checkpoints provide durable optimizer and data-position state.

`scripts/pusht/chtc_worker.sh` is the batch worker used for the bounded transfer
and subsequent shared evaluations. Its input archives contain the locked runtime,
assets, pinned external sources, repository source with `SOURCE_VERSION.json`, and
the fixed goals. It archives both checkpoints and reports before exit 85; the
submit description must transfer these archives as checkpoint files. Unexpected
failures retain the debug allocation for one hour and stop automatic progress.
The `.netrc` input is encrypted by HTCondor and is excluded from every output.
Pin the GPU model after the first match so that restored training and timed
evaluation preserve their recorded hardware. Resume rejects changed provenance.
The table generator requires complete shared-protocol evaluations, identical
targets, and matching GPU, runtime, dependencies and asset versions.

When the two evaluation configs have identical numerical settings, use
`scripts/pusht/reuse_nanowm.py` to reuse the complete paper-protocol evaluation in
the shared table. It compares every setting except the name and explanatory
assumptions, preserves the original evidence and writes an explicit reuse record.
RoboJEPA is then evaluated on the same GPU model and runtime as that NanoWM run;
its training GPU may differ. This avoids repeating an identical baseline solely
to change its protocol label. Batch phases `PUSHT_PHASE=train` and
`PUSHT_PHASE=evaluate` support these separate allocations; the evaluation phase
requires the completed training checkpoint and never initializes W&B.

## Initial reported outcomes

The first complete released-checkpoint evaluation is recorded in
[`reports/pusht/nanowm-official`](../../reports/pusht/nanowm-official): 31/64
successes (48.4375%), mean original state distance 156.758985, and median/P95
planning latency 31.409165/31.438681 seconds on an A100-SXM4-40GB. This did not
reproduce the paper's reported 25%; the missing protocol settings remain explicit
assumptions. No parameters were changed to match that number. All 64 episode
records, target states, actual weight/config loading, and runtime versions are
included.

The real RoboJEPA closed-loop smoke also completed after fixing an external
namespace collision exposed by importing the NanoWM evaluator first. Full
regression passed 39 tests with one optional RT-1 skip. Its eight-update diagnostic
checkpoint reached 0/2 goals, which verifies execution only and is not a benchmark
result. Formal transfer uses a separate initialization from the released weights.

Only success rate, final goal distance, and planning latency are comparison metrics.
Success is the original environment criterion: L2 distance over agent and block
XY coordinates below 20, and wrapped block-angle difference below pi/9. Goal
distance retains the original seven-dimensional state L2, whose components have
mixed units; it is never replaced by incomparable latent-space distances.

Latency includes image encoding and full CEM, with CUDA synchronization and a
discarded full-CEM warmup. Simulator execution, rendering and model loading are
excluded. Report mean, median and P95 per replan on the same GPU. Both paths use
bf16 predictor inference, with the released VAE encoding kept in fp32.

Each result includes protocol, source revision, dependency versions, GPU, weight
loading report, fixed targets, per-episode outcomes and individual replan times.
The two models use different representations and pretraining data; this is a
system comparison, with PushT training budget reported separately. Initial
single-seed transfer results are not a multi-seed architecture ablation.
