# Action-conditioned latent world models

## Repository inspection and scope

The starting point is NanoWM commit `cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0`.
Its Gaussian diffusion, flow matching, action injection and DFoT sampler are
already reusable. The Lightning experiment, codec factory and video metrics are
coupled to the video pipeline. Semantic codecs are not registered trainable
modules and their encode methods unconditionally disable gradients. Existing
planning repeatedly encodes observations for candidate actions.

RoboJEPA commit `f93106f5f4a451fd526d51778aa8ab7c852eb824` exposes a pretrained
action-conditioned predictor over V-JEPA 2.1-G features (1664 channels), with
embodiment-specific action/state projections and view identities. NanoWM's
existing V-JEPA 2.1-L codec is a different representation. Neither random
initialization nor a dimension adapter is a pretrained RoboJEPA predictor.

The additive `latent_world` package separates encoders, action conditioning,
predictors, objectives, rollout, planning and evaluation. Legacy entry points,
model parameter names and experiment configs remain supported. The new path
does not import an RGB decoder or compute reconstruction/pixel metrics.

## Contracts

- Features have shape `[batch, time, tokens, channels]`. `LatentSpec` records
  view/height/width layout, with tokens ordered view, row, column.
- A sequence contains T observations and T-1 outgoing actions: action t moves
  observation t to t+1. Missing history actions are an error for multi-frame
  context. Dataset-specific action normalization/chunking happens once, before
  the model. No automatic interpretation of robot actions is attempted.
- Encoders are registered modules. Frozen encoders stay in eval mode. Joint
  fine-tuning uses a separate frozen/EMA target encoder; cached features cannot
  update an encoder. EMA updates follow successful optimizer steps.
- Predictors receive only available context, outgoing actions, and explicit
  conditioning. Rollouts never receive future ground-truth observations/states.
  State propagation is a separately selected mechanism.
- NanoWM adapters reuse the original model, diffusion/flow losses and DFoT
  sampler, translating token layout and outgoing-action alignment at the boundary.
- CEM consumes encoded context and a rollout callable. The goal is encoded once.
  MPC replans from actual observations, with explicit environment success and
  goal-distance callbacks. Latent distance alone is not task success.

## Comparison protocol

Primary controlled comparisons use the same frozen V-JEPA 2.1-G weights,
preprocessing, feature normalization, dataset version, episode split, action
units/statistics, context/horizon, evaluation episode IDs, seeds and CEM budget.
Compare pretrained RoboJEPA, a randomly initialized deterministic predictor,
NanoWM diffusion and NanoWM flow in this shared feature space. Pretraining
advantage and parameter counts must be reported; this is not a compute-matched
pretraining comparison. Encoder fine-tuning is a separate representation-changing
ablation. Cross-representation latent distances are not directly comparable.

Report actual environment success, environment and latent goal distance,
per-horizon prediction error against held-out features and persistence,
paired action interventions with shared sampler randomness, latency (including
encoder and planner separately), parameter counts, peak memory, and device.
Action sensitivity demonstrates dependence, not counterfactual correctness.
Synthetic smoke results demonstrate engineering behavior, not robot performance.

## Upstream integration

RoboJEPA and V-JEPA source checkouts and weights remain external, explicitly
versioned inputs. Loading is strict and local; no silent random fallback or
automatic multi-gigabyte download occurs during model construction. Their
original licenses apply (see their LICENSE files); their source/weights are not
relicensed or vendored into NanoWM. Start with eager SDPA. Custom kernels,
compilation, distributed training and simulator installation are separate opt-ins.

References: [NanoWM](https://github.com/simchowitzlabpublic/nano-world-model),
[RoboJEPA](https://github.com/facebookresearch/robo_jepa),
[V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[DINO-WM](https://github.com/gaoyuezhou/dino_wm). The deterministic baseline uses
the next-latent prediction paradigm; it is not an exact DINO-WM reproduction.
