import pytest
import torch

from latent_world.contracts import LatentSpec
from latent_world.pusht import PushTActions
from latent_world.factory import load_config


def test_physical_action_roundtrip_and_temporal_order():
    adapter = PushTActions([.1, -.2], [.3, .7])
    pixels = torch.tensor([[3., -7.], [20., 9.], [-18., 1.], [1., 2.], [43., -31.]])
    packed = adapter.pack(pixels)
    torch.testing.assert_close(adapter.unpack(packed) * 100, pixels)
    torch.testing.assert_close(packed[:2], (pixels[0] / 100 - adapter.mean) / adapter.std)
    assert not torch.equal(adapter.pack(pixels.flip(0)), packed)
    with pytest.raises(ValueError):
        adapter.pack(pixels[:1])


def test_spatial_token_grid_preserves_rows_columns_channels():
    spec = LatentSpec(3, 16, 16)
    x = torch.arange(16*16*3).reshape(1,1,256,3)
    grid = spec.to_robojepa(x)
    torch.testing.assert_close(grid[0,0,0,3,7], x[0,0,3*16+7])
    torch.testing.assert_close(spec.from_nanowm(spec.to_nanowm(x)), x)


def test_shared_goal_and_action_protocol():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    paper = load_config(root / "configs/pusht/paper_reproduction.yaml")
    shared = load_config(root / "configs/pusht/shared_benchmark.yaml")
    for key in ["goal_H", "horizon", "replan_every", "max_episode_steps", "n_evals", "seed", "cem"]:
        assert paper[key] == shared[key]
    assert paper.goal_H == 3 and paper.cem.num_samples == 64 and paper.cem.opt_steps == 5
    assert not paper.paper_protocol_complete
