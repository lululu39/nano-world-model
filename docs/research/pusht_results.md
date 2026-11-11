# Initial PushT pretrained transfer results

Both end-to-end pipelines completed on November 11, 2026. This is a single-seed
pretrained-system transfer comparison, not an architecture ablation or an exact
reproduction of the NanoWM paper's unpublished evaluation settings.

| Method | Planning success | Mean goal distance | Planning median / P95 (seconds) |
|---|---:|---:|---:|
| Official NanoWM PushT checkpoint | 31/64 (48.44%) | 156.759 | 31.409 / 31.439 |
| RoboJEPA-22M, frozen encoder, 3,000 PushT updates | 32/64 (50.00%) | 224.400 | 0.814 / 0.828 |

Goal distance is the original environment's seven-dimensional state L2, mixing
position, angle and velocity units. It is not a latent-space distance or a
distance in meters. Both methods used the same A100-SXM4-40GB GPU, fixed 64 goals,
episode seeds, simulator, action normalization, 100-physical-action budget, and
64-candidate / 10-elite / 5-iteration CEM. Horizon and replanning interval are
three five-action chunks. Latency includes image encoding and the complete CEM
call, with CUDA synchronization and a discarded full-CEM warmup; simulator
execution, rendering and observation-tensor preparation are outside this timer.

RoboJEPA's median planning time was 38.60 times lower in these implementations.
Its success count exceeded NanoWM's by only one episode, while its mean goal
distance was worse. These results do not establish superior planning accuracy.
The models also differ in representation, pretraining data, parameter count and
PushT training budget. Additional training seeds and controlled ablations would
be needed for broader scientific conclusions; none were run in this milestone.

## NanoWM reproduction status

The checkpoint was constructed from its actual downloaded config and strictly
loaded weights. It uses SD-VAE encoder latents with shape 4 x 32 x 32, despite the
model repository name and card suggesting DINO. No VAE decoder was instantiated.

The observed 48.44% does **not** reproduce the 25% reported in
[NanoWM section 4.4 / Table 4](https://arxiv.org/html/2605.23993v1#S4.SS4).
The paper does not disclose the complete planning protocol. We retained the
released planning entrypoint's 20-step full-sequence sampler and the documented
assumptions in [paper_reproduction.yaml](../../configs/pusht/paper_reproduction.yaml),
without adjusting settings to obtain 25%.

The numerical settings of the paper-attempt and shared configs are identical.
The complete NanoWM run was reused only after checking every setting except the
protocol name and explanatory assumptions. The original report and explicit
reuse record are preserved; the baseline was not rerun to change its label.

## RoboJEPA training and verification

The official 22M cooldown-150k predictor and V-JEPA 2.1-G encoder were loaded.
All 106 pretrained predictor tensors were compared with their released values
before fine-tuning. The encoder stayed frozen; the predictor and linear 10-to-64
action adapter were updated. There was no RGB decoder, reconstruction training,
new baseline, or additional formal run.

The single seed-42 run completed 3,000 batch-four updates in 1,144.60 seconds on
an H200. The internal split held out 935 of the 18,685 official training episodes;
the official validation goals were never used for checkpoint selection. Best and
latest both correspond to step 3,000 in this run. The best checkpoint was loaded
strictly for the shared evaluation, including action-statistic compatibility.

Validation covered relative-action units and temporal order, physical action
replay, image preprocessing, spatial token layout, native predictor equivalence,
frozen-encoder gradient isolation, finite predictor/adapter gradients, and
bitwise-identical next-update checkpoint recovery on A100 and H200. The real
closed-loop smoke exposed an upstream namespace collision after NanoWM imports;
the fix includes normal-path and exception-path restoration tests. Final full
regression: 39 passed, one optional RT-1 dependency test skipped.

## Reproducibility artifacts

- [Complete shared comparison and raw records](../../reports/pusht/shared-benchmark)
- [Comparison JSON](../../reports/pusht/shared-benchmark/comparison/comparison.json)
  and [CSV](../../reports/pusht/shared-benchmark/comparison/comparison.csv)
- [Independent aggregate checks](../../reports/pusht/shared-benchmark/comparison/independent-check.json)
- [Original NanoWM paper-attempt records](../../reports/pusht/nanowm-official)
- [Training config, all 3,000 update logs and provenance](../../reports/pusht/shared-benchmark/training)
- [Experiment jobs, GPU and source versions](../../reports/pusht/shared-benchmark/experiment.json)
- [Formal W&B run](https://wandb.ai/yibozhong657-none/nanochat/runs/6804e39e)

The goal tensor artifact and latest/best checkpoints remain in workspace/staging
storage rather than Git. The pinned uv lock, model/data revisions, actual package
versions, resolved configs, per-episode seeds, target states and individual replan
times accompany the results. The table generator rejects incomplete runs or
mismatched goals, protocols, GPUs, dependencies and data versions.
