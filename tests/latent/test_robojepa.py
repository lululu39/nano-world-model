import json
import os
from pathlib import Path
import sys

import pytest
import torch

from latent_world.contracts import LatentSpec
from latent_world.external import load_predictor_constructor
from latent_world.predictors import RoboJEPAPredictor


@pytest.fixture
def source():
    path = os.environ.get("ROBOJEPA_REPO")
    if not path:
        pytest.skip("Set ROBOJEPA_REPO for real upstream source integration tests")
    return path


def test_real_upstream_predictor_load_and_gradient(source, tmp_path):
    params = dict(grid_height=2, grid_width=2, num_frames=3, embed_dim=8, predictor_embed_dim=128,
        depth=1, num_heads=1, num_views=1, view_map=[0], action_tokens=1, action_dim=4,
        state_tokens=1, state_dim=4, local_attn_window=[3, -1, -1], use_rope=True,
        attn_impl="sdpa", embodiment_list=["franka_omron"], use_multiembodiment_proj=True,
        encoder_layernorm=True, use_activation_checkpointing=False, norm_type="rmsnorm")
    preexisting = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    constructor = load_predictor_constructor(source)
    assert all(sys.modules.get(k) is v for k, v in preexisting.items())
    native = constructor(**params)
    native.attn_mask_cache.clear()  # upstream's eager prewarm follows CUDA availability, not model device
    checkpoint = tmp_path / "tiny.pth.tar"
    config = tmp_path / "config.json"
    torch.save({"predictor": {"module." + k: v for k, v in native.state_dict().items()}}, checkpoint)
    config.write_text(json.dumps({**params, "encoder_version": "v21"}))
    adapter = RoboJEPAPredictor.from_pretrained(source, config, checkpoint, LatentSpec(8, 2, 2), 2,
        state_policy="zeros", embodiment_id=0, view_ids=[0])
    z = torch.randn(2, 2, 4, 8)
    a = torch.randn(2, 2, 2, requires_grad=True)
    output = adapter(z, a)
    direct, _, _, _ = native(z.reshape(2, 2, 1, 2, 2, 8),
        torch.nn.functional.pad(a, (0, 2)), states=torch.zeros(2, 2, 4),
        embodiment_ids=torch.zeros(2, dtype=torch.long), view_map=torch.zeros(2, 1, dtype=torch.long))
    torch.testing.assert_close(output, direct.reshape(2, 2, 4, 8), rtol=0, atol=0)
    output[..., 0].sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="identity"):
        RoboJEPAPredictor(native, LatentSpec(8, 2, 2), 2, state_policy="zeros", view_ids=[0])(z, a)
    broken = dict(native.state_dict())
    broken.pop(next(iter(broken)))
    torch.save({"predictor": broken}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        RoboJEPAPredictor.from_pretrained(source, config, checkpoint, LatentSpec(8, 2, 2), 2)
