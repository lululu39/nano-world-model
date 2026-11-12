# PointMaze pretrained-system pilot

The maintainer authorized one RoboJEPA-22M MAE transfer run: seed 42,
3,000 updates, batch four, frozen V-JEPA 2.1-G encoder, pretrained predictor
and the existing 10-to-64 linear action adapter. Both teacher forcing and free
rollout use MAE, equally weighted, with one real observation and up to three
predicted history frames. The training cap is four hours. This budget is
12,000 window presentations; the PushT exposure percentage does not carry
over to PointMaze. No RGB decoder or reconstruction objective is used.

## Official assets and compatibility

NanoWM uses [the official PointMaze checkpoint](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k),
revision `b84303b7151cbd1836841c3a87bf095e1015b150`. Its actual configuration
specifies NanoWM-B/2, four training frames, one observed context, frame
interval five, two-dimensional raw controls, normalization, action scale one,
and diffusion forcing with velocity prediction. Weight loading is strict and
the numerical checks inspect the actual tensors. The existing SD encoder-only
artifact is reused; no decoder is downloaded or instantiated for this task.

The official [DINO-WM PointMaze dataset](https://osf.io/download/vr5gy/?view_only=a56a296ce3b24cceaf408383a175ce28)
contains 2,000 trajectories and 200,000 observations. The original 90/10
episode split uses seed 42: 1,800 official training trajectories, 200 planning
test trajectories. RoboJEPA reserves 90 of those 1,800 for internal dev
evaluation (seed 43), leaving 1,710 for updates. The official NanoWM checkpoint
was trained on its original training pool; these are pretrained-system
comparisons with different representations and pretraining histories.

Both models use action statistics computed only from the original 1,800
training trajectories, matching NanoWM normalization. Each latent transition
uses five chronological controls `[ax0, ay0, ..., ax4, ay4]`. These are direct
controls, not PushT displacements; there is no division by 100. The simulator
clips each physical control to `[-1, 1]`, and each control advances five
MuJoCo integration steps of 0.01 seconds. The data frame interval is a
separate five-control chunk. Checkpoints record the environment and action
representation and cannot silently load as PushT models.

Images are stored as lossless uint8 NPY arrays for memory-mapped window reads.
The transfer encoder retains the verified 256-pixel preprocessing and
row-major 16-by-16 token layout. CPU preflight replay on three training
trajectories has maximum state error below 2.5e-6, mean image error below
0.000106 on the [0,1] scale, and agent-marker center error below 0.06 pixels.
GPU preprocessing, gradient, causal attention and full recovery gates must
also pass before formal training.

## Shared planning protocol

[Shared configuration](../../configs/pointmaze/shared_benchmark.yaml):
64 fixed goals sampled only from the 200 held-out trajectories, seed 42,
five-chunk lookahead, replan every five chunks, maximum ten chunks
(50 physical control actions), CEM 64 candidates / 10 elites / five updates,
20 full-sequence DDIM steps for NanoWM, and deterministic rollout for RoboJEPA.
The same goal file and evaluation loop serve both models. Goal observations
come from replaying source actions; no simulator reset occurs inside live MPC.

This is a shared pilot protocol, not a claim to reproduce a paper table.
The upstream NanoWM PointMaze script uses 50 tasks and CEM 100/10/30; the
[DINO-WM PointMaze configuration](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/plan_point_maze.yaml)
uses random-state goals. We explicitly use held-out dataset goals and a
smaller CEM budget. Report success (position error below 0.5), original
four-dimensional state L2 goal distance, and synchronized planning latency
(online encoding plus CEM, excluding warmup and simulation/rendering).

## Execution and reproducibility

[Training configuration](../../configs/pointmaze/robojepa22m_mae_finetune.yaml)
retains the PushT pilot optimizer and learning rates. Always evaluate the
final step-3,000 checkpoint, not a separately selected best step.

`uv sync --frozen --extra pointmaze` installs a separate legacy simulator
environment: Gym 0.23.1, mujoco-py 2.1.2.14, MuJoCo engine 2.1.0 and
OSMesa rendering. The `pusht` and `pointmaze` extras are mutually exclusive;
the existing PushT runtime and running experiment remain untouched. D4RL
also depends on dm-control and its newer MuJoCo Python package, but the
PointMaze simulator uses mujoco-py and the staged `libmujoco210.so`.
Native library package versions are recorded with the runtime payload.

CPU preparation job `8413665.0` stages the environment and dataset. An initial
missing-PATH failure for the already installed `patchelf` executable was fixed
inside the retained allocation. Preparation and real-data alignment then
passed; the initial shell's failure status and manual repair evidence are
preserved rather than reported as an originally successful execution.

The GPU worker runs unit/real-data gates, an eight-update exact exit-85
recovery test, two-goal smoke checks, NanoWM evaluation, fresh formal MAE
training, and RoboJEPA evaluation in one HTCondor job. It does not release
the GPU and submit another job between those stages. Checkpoint exits retain
stage outputs, goal artifacts, model/optimizer/RNG and data position; scheduler
eviction remains possible. Actual device allocations are recorded, and resumed
allocations retain the same GPU model. Diagnostics create no W&B run and their
weights are removed after validation. Formal training starts from the official
pretrained weights and creates one `nanowm` run with a descriptive name.

Results are pending. The controller saves small reports and a separate durable
checkpoint archive, stops on failure, and never extends the authorized budget.
The execution record is `world-models/jobs/pointmaze/state.json`.

Startup correction: the first GPU allocation of job `8414049.0` stopped before
smoke tests because its container did not expose the optional `nvidia-smi`
command. Its debug window expired before intervention; no model updates or
formal W&B run were created. Device metadata now comes from PyTorch with an
optional UUID field. A reviewed `debug-resume.request` can restart the worker
inside the same live allocation after a fix, preserving the failed-attempt
record. It never retries errors automatically. Environment preparation and
unverified changes use bounded debug checks; applicable validation is reused
for later training submissions. Continuous foreground monitoring through
100 formal updates was subsequently withdrawn by the maintainer. The existing
background controller continues the authorized pipeline and stops on failure.
The failed allocation is retained in the experiment history; recovery uses
the same job and reserved run identity.

The next GPU allocation passed 52 tests, then exposed a legacy mujoco-py
compatibility issue: OSMesa initialization tried to parse the scheduler's GPU
UUID as an integer. CPU rendering now supplies an explicit device index without
changing CUDA visibility; the default GPU rendering path remains unchanged.
The fix was validated on the retained H200 allocation: rendering and replay
passed, all 106 pretrained predictor tensors matched, and preprocessing, action
alignment, causal dependence, gradients and exact next-update recovery passed.
Maximum dataset state replay error was 2.4438e-6. The original worker process
resumed in the same allocation, with no new job or formal run. Evidence is in
`reports/pointmaze/startup-debug`.

Actual exit-85 transfer and eight-update recovery subsequently passed with
bitwise-equal training state and update metrics. The first planning smoke also
exposed an undefined angle metric from the shared planner: PointMaze has no
angle, so its episode records now omit that field. Strict nonfinite checks for
measured metrics remain enabled, and four regression tests cover this behavior
and PushT compatibility. That repair also reused the retained allocation.
Both models then completed their two-goal closed-loop smoke checks. These are
engineering checks, not benchmark results; the RoboJEPA smoke checkpoint has
only eight diagnostic updates and never initializes formal training.

The worker has entered the full 64-goal NanoWM evaluation. Fresh 3,000-update
RoboJEPA training and its shared evaluation follow automatically in the same
job. The execution source is `a6a9ae7`; the earlier recovery evidence remains
applicable because training and recovery code did not change. Small smoke
records and the timestamped status are in `reports/pointmaze/startup-debug`.
