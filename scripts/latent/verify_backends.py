#!/usr/bin/env python3
"""Same-device synthetic CUDA smoke for deterministic, diffusion and flow backends."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch

from latent_world.contracts import LatentSpec, SequenceBatch
from latent_world.encoders import CachedEncoder
from latent_world.evaluation import benchmark, evaluate_prediction
from latent_world.factory import _nanowm
from latent_world.model import LatentWorldModel
from latent_world.objectives import NanoWMObjective, NextLatentObjective
from latent_world.planning import CEMConfig, LatentCEMPlanner
from latent_world.predictors import DeterministicPredictor
from latent_world.rollout import AutoregressiveRollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA compute allocation required")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    spec = LatentSpec(8, 2, 2)
    batch = SequenceBatch(latents=torch.randn(2, 3, 4, 8, device="cuda"),
                          actions=torch.randn(2, 2, 2, device="cuda"))
    reports = {}
    for name in ("deterministic", "diffusion", "flow"):
        torch.manual_seed(42)
        if name == "deterministic":
            predictor = DeterministicPredictor(spec, 2, hidden_dim=64, depth=2, heads=4, max_context=2)
            objective = NextLatentObjective("mse", rollout_weight=.5)
        else:
            predictor = _nanowm(spec, 2, hidden_dim=64, depth=2, heads=4, frames=3,
                context_frames=2, pred_name="v" if name == "diffusion" else "flow",
                diffusion_steps=20, sampling_steps=2)
            objective = NanoWMObjective()
        model = LatentWorldModel(CachedEncoder(spec), predictor, objective, AutoregressiveRollout(2)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        losses = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.loss(batch)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        model.eval()
        planner = LatentCEMPlanner(model, [-1, -1], [1, 1], CEMConfig(horizon=2, samples=4,
                                  elites=2, iterations=2, chunk_size=2))
        plan, info = planner.plan(batch.latents[:, :1], batch.latents[:, -1:])
        assert torch.isfinite(plan).all()
        reports[name] = {"diagnostic_losses": losses, "three_update_seconds": elapsed,
                         "prediction": evaluate_prediction(model, batch),
                         "efficiency": benchmark(model, batch), "planning": info}
        del model, optimizer, predictor, objective
        torch.cuda.empty_cache()
    report = {"kind": "synthetic_backend_engineering_check", "formal_training": False,
              "seed": 42, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "shared_shape": [2, 3, 4, 8], "hidden_dim": 64, "depth": 2,
              "sampling_steps": 2, "bf16_training": True, "methods": reports}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed_backends": list(reports), "gpu": report["gpu"]}))


if __name__ == "__main__":
    main()
