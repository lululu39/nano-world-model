"""Resumable, decoder-free PushT evaluation using the original NanoWM CEM/MPC."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from omegaconf import OmegaConf
import torch
from experiments.planning_experiment import PlanningExperiment, PlanningGoal
from latent_world.factory import load_config
from latent_world.pusht import load_nanowm, load_robojepa, RoboPlanningModel
from planning import CEMPlanner, Preprocessor, create_objective_fn
from planning.envs import create_planning_env


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class TimedPlanner:
    def __init__(self, planner, precision):
        self.planner, self.precision, self.seconds = planner, precision, []

    def plan(self, obs_0, obs_g, actions=None):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            if hasattr(self.planner.world_model, "prepare_context"):
                self.planner.world_model.prepare_context(obs_0)
            result = self.planner.plan(obs_0, obs_g, actions=actions)
        torch.cuda.synchronize()
        self.seconds.append(time.perf_counter() - start)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--goals", required=True)
    p.add_argument("--model", choices=["nanowm", "robojepa"], default="nanowm")
    p.add_argument("--weights", default="weights")
    p.add_argument("--robo-repo", default="robo_jepa")
    p.add_argument("--vjepa-repo", default="vjepa2")
    p.add_argument("--checkpoint")
    p.add_argument("--limit", type=int, help="Diagnostic subset; full protocol remains in saved config")
    p.add_argument("--max-seconds", type=float, default=1700)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_all(cfg.seed)
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    config_file = out / "protocol.json"
    if config_file.exists() and json.loads(config_file.read_text()) != resolved:
        raise ValueError("Cannot resume evaluation under a different protocol")
    write_json(config_file, resolved)
    assets = Path(args.assets).resolve()
    train_cfg = OmegaConf.load(assets / "nanowm/config.yaml")
    train_cfg.dataset_dir = str(assets / "data")
    experiment = PlanningExperiment(OmegaConf.create({"experiment": {"tasks": ["planning"]}}))
    env = create_planning_env("pusht")
    assert env.relative and env.action_scale == 100 and cfg.frame_interval == 5
    goals_path = Path(args.goals)
    if goals_path.exists():
        goal_data = torch.load(goals_path, weights_only=False)
        if goal_data["seed"] != cfg.seed or goal_data["goal_H"] != cfg.goal_H or len(goal_data["goals"]) != cfg.n_evals:
            raise ValueError("Goal artifact disagrees with the evaluation protocol")
        goals = [PlanningGoal(**g) for g in goal_data["goals"]]
    else:
        goals = experiment._sample_dset_goals(env, train_cfg, cfg.n_evals, cfg.goal_H, 5, cfg.seed)
        goals_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seed":cfg.seed, "goal_H":cfg.goal_H, "goals":[asdict(g) for g in goals]}, goals_path)
    write_json(out / "targets.json", experiment._serialize_goals(goals))
    if args.limit:
        goals = goals[:args.limit]
    stats = torch.load(assets / "action_stats.pt", weights_only=True)
    preprocessor = Preprocessor(action_mean=stats["mean"].repeat(5), action_std=stats["std"].repeat(5))
    if args.model == "nanowm":
        wm, _, load_report = load_nanowm(assets, cfg, "cuda")
    else:
        if not args.checkpoint:
            raise ValueError("RoboJEPA benchmark requires the fine-tuned checkpoint")
        encoder, predictor, load_report = load_robojepa(args.weights, args.robo_repo, args.vjepa_repo,
                                                       "cuda", args.checkpoint)
        wm = RoboPlanningModel(encoder, predictor)
    write_json(out / "weight-loading.json", load_report)
    root = Path(__file__).resolve().parents[2]
    write_json(out / "environment.json", {"gpu":torch.cuda.get_device_name(), "torch":torch.__version__,
        "source":json.loads((root / "SOURCE_VERSION.json").read_text()) if (root / "SOURCE_VERSION.json").exists() else None,
        "dependencies":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        "assets":json.loads((assets / "provenance.json").read_text())})
    planner = TimedPlanner(CEMPlanner(wm, create_objective_fn(**cfg.objective), action_dim=10,
        horizon=cfg.horizon, device="cuda", **cfg.cem), cfg.precision)
    # Warm up actual full CEM on a fixed observation. Never execute these actions.
    obs = {"visual":experiment._obs_to_tensor(goals[0].init_obs, 256, "cuda")}
    goal_obs = {"visual":experiment._obs_to_tensor(goals[0].goal_obs, 256, "cuda")}
    planner.plan(obs, goal_obs)
    write_json(out / "warmup.json", {"seconds":planner.seconds[-1], "excluded_from_latency":True})
    start = time.monotonic()
    for i, goal in enumerate(goals):
        path = out / f"episode-{i:04d}.json"
        if path.exists():
            continue
        seed_all(cfg.seed + i)
        planner.seconds.clear()
        single_cfg = OmegaConf.merge(cfg, {"seed":cfg.seed + i, "output_dir":str(out), "n_plot_samples":0})
        result = experiment._run_mpc(env, planner, preprocessor, [goal], single_cfg, 256, 5, torch.device("cuda"))
        record = result["per_episode"][0]
        record["episode"] = i
        record["planning_seconds"] = planner.seconds.copy()
        write_json(path, record)
        if time.monotonic() - start >= args.max_seconds:
            break
    records = [json.loads((out / f"episode-{i:04d}.json").read_text()) for i in range(len(goals))
               if (out / f"episode-{i:04d}.json").exists()]
    times = [t for r in records for t in r["planning_seconds"]]
    count = len(records); success = sum(r["success"] for r in records)
    # Wilson interval describes episode sampling uncertainty, not training-seed variation.
    z = 1.96; center = (success/count+z*z/(2*count))/(1+z*z/count)
    radius = z * np.sqrt(success/count*(1-success/count)/count+z*z/(4*count*count))/(1+z*z/count)
    summary = {"model":args.model, "protocol":cfg.name, "diagnostic_subset":args.limit is not None,
        "completed":count == len(goals), "episodes":count, "requested_episodes":len(goals),
        "success_rate":success/count, "success_count":int(success), "success_wilson95":[center-radius,center+radius],
        "goal_distance_mean":float(np.mean([r["state_dist"] for r in records])),
        "goal_distance_definition":"Original PushT seven-dimensional state L2 (mixed position/angle/velocity units)",
        "planning_latency_seconds":{"median":float(np.median(times)), "p95":float(np.percentile(times,95)),
                                    "mean":float(np.mean(times)), "replans":len(times)},
        "latency_scope":"CUDA-synchronized online image encoding plus complete CEM; excludes warmup, simulator/rendering",
        "paper_protocol_complete":cfg.paper_protocol_complete}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    env.close()
    return 0 if summary["completed"] else 85


if __name__ == "__main__":
    raise SystemExit(main())
