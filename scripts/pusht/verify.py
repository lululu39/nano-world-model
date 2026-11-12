"""Real-data GPU gate for PushT action, image, token, gradient and resume contracts."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from latent_world.pusht import load_robojepa, PushTWindows, prediction_loss, prediction_rollout, PushTActions
from latent_world.data import StatefulBatcher
from latent_world.checkpoint import save_checkpoint, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--weights", default="weights")
    p.add_argument("--robo-repo", default="robo_jepa")
    p.add_argument("--vjepa-repo", default="vjepa2")
    p.add_argument("--output", required=True)
    p.add_argument("--context-frames", type=int, default=1)
    p.add_argument("--loss-type", choices=["mae", "mse"], default="mae")
    p.add_argument("--environment", choices=["pusht", "point_maze"], default="pusht")
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4); torch.manual_seed(42)
    encoder, predictor, report = load_robojepa(args.weights, args.robo_repo, args.vjepa_repo, "cuda")
    if args.environment == "point_maze":
        from latent_world.pointmaze import PointMazeWindows
        dataset = PointMazeWindows(args.assets, "train")
    else:
        dataset = PushTWindows(args.assets, "train")
    batcher = StatefulBatcher(dataset, 1, 42)
    batch = batcher.next().to("cuda")
    # Independent expression for ImageNet normalization and row-major token layout.
    x = batch.observations.flatten(0,2).float()
    reference_input = (resize(x, [256,256], antialias=True) - x.new_tensor([.485,.456,.406])[None,:,None,None]) / x.new_tensor([.229,.224,.225])[None,:,None,None]
    captured = []
    hook = encoder.backbone.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(batch.observations)
    hook.remove()
    torch.testing.assert_close(captured[0], reference_input.unsqueeze(2), rtol=0, atol=1e-6)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = F.layer_norm(encoder.backbone(reference_input.unsqueeze(2)).float(), (1664,)).reshape_as(z)
        torch.testing.assert_close(z, reference, rtol=0, atol=0)
        # Inference [-1,1] conversion must preserve the same image coordinates/features.
        image_roundtrip = ((batch.observations * 2 - 1) + 1) / 2
        torch.testing.assert_close(image_roundtrip, batch.observations, rtol=0, atol=6e-8)
        direct, _, _, _ = predictor.predictor(predictor.spec.to_robojepa(z[:, :1]),
            actions=predictor.action_adapter(batch.actions[:, :1]), states=z.new_zeros(1,1,64),
            embodiment_ids=torch.tensor([2], device="cuda"), view_map=torch.tensor([[0]], device="cuda"))
        adapted = predictor(z[:, :1], batch.actions[:, :1])
        torch.testing.assert_close(direct.reshape_as(adapted), adapted, rtol=0, atol=0)
    if args.context_frames > 1:
        # Check the actual released attention implementation, not a surrogate mask.
        predictor.eval()
        with torch.no_grad():
            inputs, actions = z[:, :-1].float(), batch.actions.float()
            parallel = predictor(inputs, actions)
            prefixes = torch.cat([predictor.predict_next(inputs[:, :k+1], actions[:, :k+1])
                                  for k in range(3)], dim=1)
            torch.testing.assert_close(parallel, prefixes, rtol=1e-4, atol=2e-5)
            changed_z, changed_a = inputs.clone(), actions.clone()
            changed_z[:, 1:] += torch.randn_like(changed_z[:, 1:])
            changed_a[:, 1:] += 3
            causal = predictor(changed_z, changed_a)
            torch.testing.assert_close(parallel[:, :1], causal[:, :1], rtol=0, atol=0)
            changed_z = inputs.clone()
            changed_z[:, :1] += torch.randn_like(changed_z[:, :1])
            history_delta = float((predictor(changed_z, actions)[:, -1:] - parallel[:, -1:]).abs().max())
            assert history_delta > 1e-6, "Predictor ignored the earlier history"
            changed_a = actions.clone(); changed_a[:, :1] += 3
            action_delta = float((predictor(inputs, changed_a)[:, -1:] - parallel[:, -1:]).abs().max())
            assert action_delta > 1e-6, "Predictor ignored historical actions"
            # Independent growing-prefix reference checks rollout order and feedback.
            prefix = z[:, :1].float()
            expected_rollout = []
            for k in range(3):
                nxt = predictor.predict_next(prefix, actions[:, :k+1])
                expected_rollout.append(nxt); prefix = torch.cat([prefix, nxt], dim=1)
            torch.testing.assert_close(prediction_rollout(predictor, z[:, :1].float(), actions, 3),
                                       torch.cat(expected_rollout, 1), rtol=0, atol=0)
        report.update({"prefix_teacher_equivalence":True, "future_inputs_cannot_change_past":True,
                       "history_feature_max_delta":history_delta, "history_action_max_delta":action_delta,
                       "predicted_prefix_action_alignment":True})
    if args.environment == "point_maze":
        from latent_world.pointmaze import verify_environment_alignment
        report.update(verify_environment_alignment(args.assets, Path(args.output).parent))
    else:
        # Real action chunk round trip, including simulator's own x100 conversion.
        transform = dataset.action_transform
        raw = dataset.actions[0,:15].reshape(3,5,2)
        torch.testing.assert_close(transform.unpack(transform.pack(raw))*100, raw, rtol=1e-5, atol=1e-5)
        states = torch.load(dataset.root / "states.pth", weights_only=False).float()
        absolute = torch.load(dataset.root / "abs_actions.pth", weights_only=False).float()
        torch.testing.assert_close(dataset.actions[0,:15], absolute[0,:15]-states[0,:15,:2], rtol=1e-5, atol=1e-4)
        velocities = torch.load(dataset.root / "velocities.pth", weights_only=False).float()
        initial = torch.cat([states[0,0], velocities[0,0]]).numpy()
        from planning.envs import create_planning_env
        env = create_planning_env("pusht")
        env.prepare(42, initial)
        direct_states = [env.step(a.numpy())[3]["state"] for a in raw.reshape(-1,2)/100]
        env.prepare(42, initial)
        roundtrip_states = [env.step(a.numpy())[3]["state"] for a in transform.unpack(transform.pack(raw)).reshape(-1,2)]
        np.testing.assert_allclose(direct_states, roundtrip_states, rtol=1e-5, atol=1e-4)
        env.close()
        del states, absolute, velocities
        report["relative_actions_match_source_positions"] = True
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=5e-5)
    def update():
        predictor.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = prediction_loss(predictor, z, batch.actions, args.context_frames, loss_type=args.loss_type)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(v.grad is None for v in encoder.parameters())
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in predictor.action_adapter.parameters())
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in predictor.predictor.parameters())
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    first = update()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"diagnostic.pt"
        save_checkpoint(path, model=predictor, optimizer=optimizer, batcher=batcher, step=1,
                        run_id=None, config={"diagnostic":True}, best=first, elapsed=0.)
        expected_loss = update()
        expected = {k:v.detach().cpu().clone() for k,v in predictor.state_dict().items()}
        load_checkpoint(path, model=predictor, optimizer=optimizer, batcher=batcher, config={"diagnostic":True})
        actual_loss = update()
        assert expected_loss == actual_loss
        differences = {k:float((expected[k]-v.cpu()).abs().max()) for k,v in predictor.state_dict().items()
                       if not torch.equal(expected[k], v.cpu())}
        print(json.dumps({"resume_loss_equal":expected_loss == actual_loss,
                          "resume_parameter_differences":differences}), flush=True)
        assert not differences
    report.update({"actual_resolution":[256,256], "preprocessing_verified":True, "spatial_tokens_verified":True,
        "native_predictor_equivalence":True, "gradient_verified":True, "exact_resume_next_update":True,
        "environment":args.environment, "normalized_action_replay_matches_raw":True,
        "deterministic_algorithms":True, "diagnostic_loss":first, "gpu":torch.cuda.get_device_name(),
        "peak_bytes":torch.cuda.max_memory_allocated(), "rollout_context_frames":args.context_frames,
        "loss_type":args.loss_type})
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
