#!/usr/bin/env python3
"""Bounded CUDA integration check with actual released weights; no W&B or training run."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch

from latent_world.contracts import LatentSpec, SequenceBatch
from latent_world.encoders import robojepa_encoder
from latent_world.evaluation import benchmark, evaluate_prediction
from latent_world.model import LatentWorldModel
from latent_world.objectives import NextLatentObjective
from latent_world.planning import CEMConfig, LatentCEMPlanner
from latent_world.predictors import RoboJEPAPredictor
from latent_world.rollout import AutoregressiveRollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robojepa-repo", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires a compute allocation with CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    start = time.perf_counter()
    weights = Path(args.weights)
    # A small spatial grid tests the actual architecture/weights affordably.
    # It is not a benchmark at the 256/384-pixel research resolutions.
    spec = LatentSpec(1664, 2, 2)
    encoder = robojepa_encoder(args.vjepa_repo, weights / "vjepa2_1_vitG_384.pt", spec,
                              input_size=(32, 32), frozen=True)
    predictor = RoboJEPAPredictor.from_pretrained(args.robojepa_repo, weights / "config.json",
        weights / "cooldown_22m_150k.pth.tar", spec, 2, action_mode="linear", state_policy="zeros",
        embodiment_id=2, view_ids=[0])
    model = LatentWorldModel(encoder, predictor, NextLatentObjective(rollout_weight=.5),
                            AutoregressiveRollout(2)).cuda()
    batch = SequenceBatch(observations=torch.rand(1, 3, 1, 3, 32, 32, device="cuda"),
                          actions=torch.randn(1, 2, 2, device="cuda"))
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.action_adapter.parameters())
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode(batch.observations)
        adapted = predictor(z[:, :2], batch.actions)
        direct, _, _, _ = predictor.predictor(spec.to_robojepa(z[:, :2]),
            actions=predictor.action_adapter(batch.actions), states=z.new_zeros(1, 2, predictor.predictor.state_dim),
            embodiment_ids=torch.tensor([2], device="cuda"), view_map=torch.tensor([[0]], device="cuda"))
        torch.testing.assert_close(adapted, direct.reshape_as(adapted), rtol=0, atol=0)
        changed = batch.actions.clone()
        changed[:, 1] += 1
        causal = predictor(z[:, :2], changed)
        torch.testing.assert_close(adapted[:, 0], causal[:, 0], rtol=0, atol=0)
        assert not torch.equal(adapted[:, 1], causal[:, 1])
        prediction_report = evaluate_prediction(model, batch)
        efficiency = benchmark(model, batch, repeats=2)
        planner = LatentCEMPlanner(model, [-1, -1], [1, 1], CEMConfig(horizon=2, samples=4,
                                  elites=2, iterations=2, chunk_size=2))
        plan, plan_info = planner.plan(z[:, :1], z[:, -1:], seed=42)
        assert torch.isfinite(plan).all()
    # Prove the actual pretrained encoder can receive gradients. No optimizer
    # update or diagnostic checkpoint is produced; the large frozen target is
    # tested with tiny modules in the exact-resume suite.
    encoder.frozen = False
    encoder.backbone.requires_grad_(True)
    encoder.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fine_features = encoder(batch.observations[:, :1])
        fine_loss = fine_features[..., :8].mean()
    fine_loss.backward()
    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert encoder_grads and any(g.abs().sum() > 0 for g in encoder_grads)
    assert all(torch.isfinite(g).all() for g in encoder_grads)
    report = {"kind": "pretrained_engineering_check", "formal_training": False,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "resolution": [32, 32],
              "frozen_encoder_gradient_isolation": True, "pretrained_predictor_gradient": True,
              "pretrained_encoder_gradient": True, "strict_checkpoint_load": True,
              "native_predictor_equivalence": True, "future_action_causality": True,
              "frozen_loss": float(loss), "prediction": prediction_report, "efficiency": efficiency,
              "planning": plan_info, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "elapsed_seconds": time.perf_counter() - start}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
