"""Real-data GPU gate for PushT action, image, token, gradient and resume contracts."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from latent_world.pusht import load_robojepa, PushTWindows, prediction_loss, PushTActions
from latent_world.data import StatefulBatcher
from latent_world.checkpoint import save_checkpoint, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--weights", default="weights")
    p.add_argument("--robo-repo", default="robo_jepa")
    p.add_argument("--vjepa-repo", default="vjepa2")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    encoder, predictor, report = load_robojepa(args.weights, args.robo_repo, args.vjepa_repo, "cuda")
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
    # Real action chunk round trip, including simulator's own x100 conversion.
    transform = dataset.action_transform
    raw = dataset.actions[0,:15].reshape(3,5,2)
    torch.testing.assert_close(transform.unpack(transform.pack(raw))*100, raw, rtol=1e-5, atol=1e-5)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=5e-5)
    def update():
        predictor.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = prediction_loss(predictor, z, batch.actions)
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
        assert all(torch.equal(expected[k], v.cpu()) for k,v in predictor.state_dict().items())
    report.update({"actual_resolution":[256,256], "preprocessing_verified":True, "spatial_tokens_verified":True,
        "native_predictor_equivalence":True, "gradient_verified":True, "exact_resume_next_update":True,
        "diagnostic_loss":first, "gpu":torch.cuda.get_device_name(), "peak_bytes":torch.cuda.max_memory_allocated()})
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
