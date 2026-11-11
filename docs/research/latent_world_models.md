# Latent world model research guide

The new package is additive. Existing `src/main.py`, Hydra experiments,
checkpoints, diffusion/flow losses and video applications remain supported.
See [the inspection and architecture decisions](architecture.md).

## Environment and smoke checks

The default `./nanowm sync` retains the complete original environment. For the
decoder-free path, use a separate environment with the same committed uv lock:

```bash
UV_PROJECT_ENVIRONMENT=.venv-latent uv sync --locked --only-group latent --only-group dev
.venv-latent/bin/python scripts/latent/run.py train \
  --config configs/latent/smoke.yaml --smoke --device cpu --output outputs/latent-smoke
.venv-latent/bin/python scripts/latent/run.py evaluate \
  --config configs/latent/smoke.yaml --smoke --device cpu \
  --checkpoint outputs/latent-smoke/latest.pt --output outputs/latent-smoke-eval
.venv-latent/bin/python -m pytest tests/latent -q
./nanowm python -m pytest -q
python3 scripts/check_repository_language.py
```

Run tensor checks on compute nodes when required by your cluster. The supported
platform remains Linux x86_64/Python 3.11/Torch 2.7.1/CUDA 12.8. The latent group
has no diffusers, LPIPS, FVD, VAE or RGB decoder. Torchvision/timm support the
backbones. W&B is imported only by non-smoke training. Tests, model loading,
evaluation and planning never create online or offline W&B runs. `--smoke` limits
diagnostics to 16 updates. Simulator dependencies remain optional.

The source integration test skips unless `ROBOJEPA_REPO` is set. The real-weight
CUDA check below is separate: synthetic tests do not establish pretrained
checkpoint compatibility or downstream task performance.

## Stage pretrained assets

Keep external sources and their licenses outside the NanoWM tree:

```bash
git clone https://github.com/facebookresearch/robo_jepa.git /path/to/robo_jepa
git -C /path/to/robo_jepa checkout f93106f5f4a451fd526d51778aa8ab7c852eb824
git clone https://github.com/facebookresearch/vjepa2.git /path/to/vjepa2
git -C /path/to/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
export ROBOJEPA_REPO=/path/to/robo_jepa
export VJEPA2_REPO=/path/to/vjepa2
export ROBOJEPA_CONFIG=$ROBOJEPA_REPO/configs/released/wm-22m/config.json
export ROBOJEPA_CHECKPOINT=/path/to/weights/cooldown_22m_150k.pth.tar
export VJEPA2_CHECKPOINT=/path/to/weights/vjepa2_1_vitG_384.pt
export LATENT_MANIFEST=/path/to/data/manifest.json
```

Official weights: [22M predictor, 150k](https://dl.fbaipublicfiles.com/robojepa/wm/wm-22m/cooldown_22m_150k.pth.tar)
and [V-JEPA 2.1-G encoder](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt).
The encoder download is about 30 GB and includes additional training state.
Only `target_encoder` is needed: a compute preparation job can save that
dictionary alone, recording the source URL, original size and retained key.
No decoder weights are needed. Model construction does not download assets.

The adapter strictly loads the official eager SDPA predictor and V-JEPA image
backbone. Missing/unexpected weights, feature mismatches and unspecified robot
identities fail; there is no random fallback. Imports of generic upstream
`app`/`src` packages are isolated during construction. Compilation and distributed
execution are not promised by this initial adapter. Original upstream source and
weight licenses apply; they are not vendored or relicensed as MIT here.

The configs use one 256x256 camera, a 16x16 feature grid, 1664 channels, bilinear
antialiased resize, ImageNet normalization and per-token layer norm. This is an
explicit downstream adaptation of a multi-camera pretrained robotic model.
`action_mode: linear` learns a task-to-64-dimensional action adapter. Its chosen
embodiment/view slots are adaptation initializations, not a claim that arbitrary
actions have native robot semantics. Native robotics must use the released
task's action/state preprocessing and correct identities.

## Data and comparison matrix

Choose one domain with actual actions, held-out episodes and an executable
environment before formal experiments. No dataset is implicitly downloaded.
Fill `protocol.dataset_id`, `protocol.action_id` and `protocol.evaluation_id` in
the shared config. Use the [manifest template](../../configs/latent/manifest.example.json).
Each episode file is a `torch.save` tensor dictionary:

| Field | Shape and semantics |
| --- | --- |
| observations | `[T,V,3,H,W]`, float RGB in `[0,1]`, encoder inputs only |
| latents | `[T,N,D]`, alternative to observations, fixed encoder features |
| actions | `[T-1,A]`, action t moves observation t to t+1 |
| states | Optional `[T,S]`, state at observation t |
| embodiment_id | Optional scalar long tensor |
| view_ids | Optional `[V]` long tensor of indices in the pretrained view map |

Exactly one of observations/latents is required. Windows stay within episodes;
duplicate IDs/files across splits fail. Data is loaded with `weights_only=True`.
The manifest's representation/action IDs must match the protocol. Prepare
train-only action statistics once and reuse them for validation, test and control.
Do not split independently per model. `from_nanowm_batch` adapts existing
`WorldModelDataset` batches, preserving their split/statistics, converting pixels
from `[-1,1]`, and removing the final unused action. Start with frame interval 1;
changes must preserve action chunking and physical time units.

| Config | Dynamics | Representation |
| --- | --- | --- |
| `robojepa_frozen.yaml` | Pretrained predictor + task adapter | Shared frozen V-JEPA-G |
| `deterministic.yaml` | Random frame-causal next-latent transformer | Same frozen V-JEPA-G |
| `nanowm_diffusion.yaml` | Original NanoWM v prediction + DFoT | Same frozen V-JEPA-G |
| `nanowm_flow.yaml` | Original NanoWM flow matching | Same frozen V-JEPA-G |
| `robojepa_finetune.yaml` | Pretrained predictor + encoder fine-tuning | Separate representation ablation |

`assert_comparable` checks dataset, representation, actions, evaluation and
planning budgets. It rejects joint encoder fine-tuning in the primary comparison.
Parameter counts and predictor pretraining differ and must be reported. Compare
shared held-out prediction/control metrics, not incompatible raw training losses.
The deterministic model follows the DINO-WM paradigm, not an exact reproduction.
NanoWM currently requires a single square grid. Multi-view extensions of the
other predictors are not a four-method matched comparison.

The frozen config trains predictor and task adapter. `predictor.frozen=true`
trains only the adapter. The fine-tuning config also trains the encoder with an
EMA target; `target_decay=1` keeps the target fixed. Targets are checkpointed.
Cached features are rejected when the encoder is trainable.

## Train, restore and evaluate

```bash
.venv-latent/bin/python scripts/latent/run.py train \
  --config configs/latent/robojepa_frozen.yaml --device cuda --output outputs/robojepa
.venv-latent/bin/python scripts/latent/run.py train \
  --config configs/latent/robojepa_frozen.yaml --device cuda --output outputs/robojepa \
  --checkpoint outputs/robojepa/latest.pt
.venv-latent/bin/python scripts/latent/run.py evaluate \
  --config configs/latent/robojepa_frozen.yaml --device cuda --output outputs/robojepa-test \
  --checkpoint outputs/robojepa/best.pt
```

`latest.pt` is overwritten; `best.pt` is independent. Checkpoints include model,
target encoder, optimizer, Python/NumPy/Torch/CUDA RNG, data permutation/position,
epoch, step and W&B run ID. Resume requires the same scientific config. Config,
provenance and metrics are written locally. The runner exits 85 at the configured
checkpoint interval; a batch worker must durably transfer checkpoint artifacts
before propagating 85. Scratch alone is not persistent. Diagnostics do not start
formal training.

Held-out evaluation reports per-horizon MSE/MAE, persistence error and paired
zero-action sensitivity with shared sampler RNG. Sensitivity demonstrates action
dependence, not counterfactual correctness. Encoder/rollout latency, parameter
counts, device and allocated GPU memory are recorded separately.

## Planning

All predictors use `LatentCEMPlanner` and `MPCController`. CEM has bounded actions,
finite-cost checks, a variance floor, candidate chunks, separately seeded
candidate sampling and optional stochastic replicates. Match these budgets and
report native diffusion sampling steps. Different CEM candidates do not share
diffusion noise in this initial implementation; paired sensitivity evaluations
do. MPC executes a prefix and replans from actual observations.

For real task evaluation, implement the `evaluate_mpc_episode` adapter contract
and set `evaluation.environment_factory=module:function`. The factory receives
config and a fixed episode entry, sets the goal, and converts model-space actions
to physical units. Reset returns observations `[1,1,V,3,H,W]` and `Conditioning`.
Step returns those plus actual `success`, `goal_distance` and `terminated`.
Latent distance must not replace environment success. Supply a JSON episode list
with `episode_id`, `seed`, `goal_path`; goal files are `[1,1,V,3,H,W]` tensors.

```bash
export PLANNING_EPISODES=/path/to/fixed-planning-episodes.json
.venv-latent/bin/python scripts/latent/run.py plan \
  --config configs/latent/robojepa_frozen.yaml --device cuda --output outputs/planning \
  --checkpoint outputs/robojepa/best.pt
```

The analytic planning test verifies mechanics, not PushT/PointMaze/robot success.
Those require selected data, task-specific environment adapters and actual runs.

## Extension interfaces

An encoder is an `nn.Module` with `spec`, `frozen`, and differentiable forward.
A predictor exposes `spec`, `max_context`, and
`predict_next(context, outgoing_actions, conditioning) -> [B,1,N,D]`.
An objective takes `(model,batch,online_features,target_features)` and returns
`(loss,metrics)`. A rollout follows `AutoregressiveRollout`'s contract. Register
factories in `ENCODERS`, `PREDICTORS`, `OBJECTIVES` and `ROLLOUTS`; loops stay shared.
Action modules live in `conditioning.py`; replace `action_conditioner` with a
module using `(tokens,actions) -> tokens` for architecture research. RoboJEPA
retains its pretrained conditioning and exposes a separate task adapter.
No future ground-truth state enters rollout. Zero-state and same-coordinate
delta-state mechanisms are provided; pass a custom `state_transition(states, action)`
callable to the rollout for other semantics.

Actual pretrained CUDA check:

```bash
.venv-latent/bin/python scripts/latent/verify_pretrained.py \
  --robojepa-repo "$ROBOJEPA_REPO" --vjepa-repo "$VJEPA2_REPO" \
  --weights /path/to/weights --output pretrained-check.json
```

It uses real weights at a tiny 32x32 resolution to check strict loading,
gradient flow, native equivalence, causal actions and planning. It does not
establish long-horizon stability, task success or research-resolution throughput.
