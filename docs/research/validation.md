# Initial validation (2026-11-11)

This is an engineering validation, not a downstream world-model benchmark.
No formal training, W&B runs, RGB decoder or reconstruction objective was used.
All tensor work ran in HTCondor allocations, not on the access point.

## Results

| Check | Result |
| --- | --- |
| Full repository regression, CPU | 33 passed, 1 skipped |
| Full regression after device-cache fix, GPU host | 33 passed, 1 skipped |
| Skipped check | Optional RT-1 extra was not installed |
| Native diffusion and flow loss equality | Exact equality against original process loss |
| Full training resume | Next loss and complete model/EMA state exactly match uninterrupted updates |
| Decoder-free environment | No diffusers, LPIPS, PIQA, Lightning or transformers installed in the latent group |
| Decoder-free imports | Import blocker test passes for latent entry points and original NanoWM backbone |
| Official 22M predictor and V-JEPA 2.1-G | Strict real checkpoint loading passes |
| Frozen encoder | No encoder gradients; predictor and task adapter receive finite gradients |
| Encoder fine-tuning | Actual pretrained encoder receives finite, nonzero gradients |
| Upstream predictor equivalence | Exact output agreement with direct upstream invocation |
| Causal actions | Future action changes do not affect earlier predictions |
| Latent CEM/MPC | Shared planner tests and analytic environment success pass |
| CUDA backends | Deterministic, diffusion and flow: finite bf16 loss/gradients, rollout and CEM |

The complete tests and exact versions are recorded in
[`reports/latent`](../../reports/latent). The original source is NanoWM
`cc1f151`; tested core changes end at `d96e74f`, with the extra CUDA backend probe
added in `3ad4f16`. Python 3.11.18 from the existing runtime was used for compute
validation, within the project's declared 3.11 range; the upstream preferred
interpreter remains 3.11.13. uv 0.12.0 and 0.13.6 both accept the lock.

Real-weight checks ran on NVIDIA A100-SXM4-40GB with Torch 2.7.1+cu128, at 32x32
input resolution. They took about 30 seconds excluding environment transfer and
setup. Peak allocated memory including the separate encoder gradient check was
about 14.0 GiB. These numbers do not estimate 256/384-pixel training or control.

The same-device synthetic backend probe used batch 2, three frames, four tokens,
eight channels, hidden width 64, two layers and two sampler steps. Mean two-step
rollout times were approximately 2.12 ms (deterministic), 18.62 ms (diffusion),
and 13.39 ms (flow). Parameter counts differ (105,928 vs 216,392), and these tiny
models received only three diagnostic updates. These are compatibility/timing
checks, not a research architecture ranking. Full measurements remain in JSON.

## Issues found and fixed

The official encoder archive is 30.24 GB. An initial 65 GB scratch request was
insufficient when packaging the untrimmed file and environment together. The
job was stopped, then corrected to extract only `target_encoder` (7.38 GB) on
the compute node and use a 100 GB scratch budget. Preparation then exited 0;
only the compact weights and one reusable runtime archive were retained.

RoboJEPA prewarms a plain attention-mask cache on CUDA whenever CUDA is present,
including when its parameters remain on CPU. Its cache is not an nn.Module
buffer and therefore does not follow `.to()`. The adapter now clears runtime
masks on construction/device changes and lets upstream forward rebuild them on
the input device, preserving the causal-mask enable flag. The CPU-on-GPU-host
test exposed this issue; the corrected adapter and real CUDA weights both pass.

## Remaining experiment work

No real dataset, task-specific environment adapter, fixed goal episodes or
formal training budget has been selected for this new codebase. Consequently,
there is no measured PushT/PointMaze/robot success rate or long-horizon quality
claim. Joint encoder fine-tuning, new embodiments, other GPU families, compilation,
distributed execution, large grids and optional simulator stacks still require
their corresponding experiment checks. The core supports those extensions
through explicit interfaces; it does not silently assume compatibility.
