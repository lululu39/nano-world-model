"""Train/evaluate latent models with local reports and complete resumable state."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
from omegaconf import OmegaConf
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import EpisodeWindowDataset, StatefulBatcher, collate_sequences
from .evaluation import benchmark, evaluate_mpc_episode, evaluate_prediction, evaluating
from .factory import build_model, load_config
from .planning import CEMConfig, LatentCEMPlanner, MPCController
from .protocol import comparison_contract


class SmokeDataset:
    """Small deterministic fixture, explicitly excluded from research results."""

    def __init__(self, spec, frames, action_dim, seed):
        self.spec, self.frames, self.action_dim, self.seed = spec, frames, action_dim, seed

    def __len__(self):
        return 12

    def __getitem__(self, index):
        rng = torch.Generator().manual_seed(self.seed + index)
        return {"observations": torch.rand(self.frames, self.spec.views, 3,
                    self.spec.height * 4, self.spec.width * 4, generator=rng),
                "actions": torch.randn(self.frames - 1, self.action_dim, generator=rng)}


def provenance(cfg):
    def commit(path):
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    root = Path(__file__).resolve().parents[2]
    code = commit(root)
    if code is None and (root / "SOURCE_VERSION.json").is_file():
        code = json.loads((root / "SOURCE_VERSION.json").read_text())["commit"]
    actual_sources = {}
    for component in ("encoder", "predictor"):
        path = cfg[component].get("repo")
        if path:
            revision = commit(path)
            metadata = Path(path) / "SOURCE_VERSION.json"
            if revision is None and metadata.is_file():
                revision = json.loads(metadata.read_text())["commit"]
            actual_sources[component] = revision
    return {"code_commit": code, "torch": torch.__version__, "actual_source_revisions": actual_sources,
            "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "upstream_revisions": cfg.get("upstream_revisions", {}),
            "comparison_contract": comparison_contract(cfg)}


def dataset(cfg, spec, split, smoke):
    if smoke:
        return SmokeDataset(spec, cfg["data"]["frames"], cfg["action_dim"], cfg["seed"] + (0 if split == "train" else 10000))
    return EpisodeWindowDataset(cfg["data"]["manifest"], split, cfg["data"]["frames"],
        dataset_id=cfg["protocol"]["dataset_id"], representation_id=cfg["representation"]["id"],
        action_id=cfg["protocol"]["action_id"], stride=cfg["data"]["stride"])


def train(cfg, model, args, output):
    training = cfg["training"]
    if args.smoke and training["steps"] > 16:
        raise ValueError("Smoke training is limited to 16 diagnostic updates")
    train_data = dataset(cfg, model.spec, "train", args.smoke)
    val_data = dataset(cfg, model.spec, "val", args.smoke)
    batcher = StatefulBatcher(train_data, training["batch_size"], cfg["seed"])
    groups = [{"params": [p for p in model.predictor.parameters() if p.requires_grad], "lr": training["lr"]}]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": training["encoder_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    state = {"step": 0, "run_id": None if args.smoke else uuid.uuid4().hex[:8], "best": float("inf"), "elapsed": 0.}
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, model=model, optimizer=optimizer, batcher=batcher, config=cfg)
    elif (output / "latest.pt").exists():
        raise FileExistsError("Existing training state: use --checkpoint to resume or choose a new output directory")
    if args.smoke != (state["run_id"] is None):
        raise ValueError("Diagnostic and formal checkpoints cannot be interchanged")
    tracker = None
    if not args.smoke:
        import wandb
        tracker = wandb.init(project=cfg["logging"]["project"], entity=cfg["logging"]["entity"],
            id=state["run_id"], resume="must" if args.checkpoint else "never", config=cfg,
            mode=cfg["logging"].get("mode", "online"), dir=str(output))
    start = time.monotonic()
    base_elapsed = state["elapsed"]
    metrics = []
    def save(name):
        save_checkpoint(output / name, model=model, optimizer=optimizer, batcher=batcher,
            step=state["step"], run_id=state["run_id"], config=cfg, best=state["best"],
            elapsed=base_elapsed + time.monotonic() - start)
    try:
        while state["step"] < training["steps"]:
            model.train()
            batch = batcher.next().to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=training.get("bf16", False)):
                loss, components = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                                                   training["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            model.update_target()
            state["step"] += 1
            record = {"step": state["step"], "loss": float(loss.detach()), "grad_norm": float(norm),
                      **{key: float(value) for key, value in components.items()}}
            if state["step"] % training["eval_every"] == 0 or state["step"] == training["steps"]:
                # Fixed held-out windows and fixed RNG; no effect on training RNG.
                from .planning import sampling_seed
                total, count = 0., 0
                with evaluating(model), sampling_seed(cfg["seed"], args.device):
                    for offset in range(0, len(val_data), training["batch_size"]):
                        vb = collate_sequences([val_data[i] for i in range(offset, min(offset + training["batch_size"], len(val_data)))]).to(args.device)
                        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                            enabled=training.get("bf16", False)):
                            val_loss, _ = model.loss(vb)
                        total += float(val_loss) * vb.actions.shape[0]
                        count += vb.actions.shape[0]
                record["val_loss"] = total / count
                if not np.isfinite(record["val_loss"]):
                    raise FloatingPointError("Nonfinite validation loss")
                if record["val_loss"] < state["best"]:
                    state["best"] = record["val_loss"]
                    save("best.pt")
                save("latest.pt")
            metrics.append(record)
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            if tracker is not None:
                tracker.log(record, step=state["step"])
            if time.monotonic() - start >= training.get("checkpoint_seconds", 1800):
                save("latest.pt")
                return 85  # worker must persist artifacts successfully before propagating 85
        save("latest.pt")
    finally:
        if tracker is not None:
            tracker.finish()
    (output / "train-summary.json").write_text(json.dumps({"smoke": args.smoke,
        "step": state["step"], "best_val_loss": state["best"], "run_id": state["run_id"],
        "updates_this_invocation": len(metrics)}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "plan"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = OmegaConf.to_container(load_config(args.config, args.overrides), resolve=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(args.device)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "provenance.json").write_text(json.dumps(provenance(cfg), indent=2))
    if args.command == "train":
        return train(cfg, model, args, output)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model, config=cfg, restore_rng=False)
    model.eval()
    if args.command == "evaluate":
        data = dataset(cfg, model.spec, "test", args.smoke)
        reports, sums = [], {}
        first = None
        for offset in range(0, len(data), cfg["training"]["batch_size"]):
            batch = collate_sequences([data[i] for i in range(offset, min(offset + cfg["training"]["batch_size"], len(data)))]).to(args.device)
            if first is None:
                first = batch
            report = evaluate_prediction(model, batch, seed=cfg["seed"] + offset,
                                         context_frames=cfg["protocol"]["context_frames"])
            reports.append(report)
            for row in report["per_horizon"]:
                total = sums.setdefault(row["horizon"], {"count": 0})
                total["count"] += row["count"]
                for key, value in row.items():
                    if key not in {"count", "horizon"}:
                        total[key] = total.get(key, 0.) + value * row["count"]
        aggregate = [{"horizon": h, **{k: v if k == "count" else v / row["count"] for k, v in row.items()}}
                     for h, row in sorted(sums.items())]
        result = {"smoke": args.smoke, "per_horizon": aggregate, "batches": reports,
                  "efficiency": benchmark(model, first, seed=cfg["seed"], context_frames=cfg["protocol"]["context_frames"])}
    else:
        if args.smoke:
            raise ValueError("Use the analytic planning smoke tests; real planning requires an environment adapter")
        module, name = cfg["evaluation"]["environment_factory"].split(":")
        env_factory = getattr(importlib.import_module(module), name)
        planner = LatentCEMPlanner(model, cfg["planning"]["action_low"], cfg["planning"]["action_high"],
                                  CEMConfig(**cfg["planning"]["cem"]))
        controller = MPCController(planner, cfg["planning"]["execute_steps"])
        episodes = json.loads(Path(cfg["evaluation"]["episodes"]).read_text())
        if not episodes:
            raise ValueError("Fixed planning episode manifest is empty")
        results = []
        for episode in episodes:
            env = env_factory(cfg, episode)
            try:
                goal = torch.load(episode["goal_path"], map_location="cpu", weights_only=True)
                results.append(evaluate_mpc_episode(model, controller, env, goal,
                    seed=episode["seed"], episode_id=episode["episode_id"], max_steps=cfg["evaluation"]["max_steps"]))
            finally:
                if hasattr(env, "close"):
                    env.close()
        result = {"episodes": results, "success_rate": sum(x["success"] for x in results) / len(results)}
    (output / f"{args.command}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
