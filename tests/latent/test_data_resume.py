import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from latent_world.checkpoint import load_checkpoint, save_checkpoint
from latent_world.cli import SmokeDataset, main
from latent_world.contracts import LatentSpec
from latent_world.data import EpisodeWindowDataset, StatefulBatcher, from_nanowm_batch
from latent_world.factory import build_model, load_config
from latent_world.protocol import assert_comparable


ROOT = Path(__file__).resolve().parents[2]


def test_complete_resume_matches_next_update(tmp_path):
    cfg = OmegaConf.to_container(load_config(ROOT / "configs/latent/smoke.yaml", ["encoder.frozen=false"]), resolve=True)
    model = build_model(cfg)
    data = SmokeDataset(model.spec, 3, 2, 42)
    batcher = StatefulBatcher(data, 2, 4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    def update(m, opt, loader):
        opt.zero_grad(set_to_none=True)
        loss, _ = m.loss(loader.next())
        loss.backward()
        opt.step()
        m.update_target()
        return loss.detach()
    update(model, optimizer, batcher)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, batcher=batcher, step=1,
                    run_id="same-run", config=cfg, best=1., elapsed=2.)
    expected_random = torch.randn(3)
    expected_loss = update(model, optimizer, batcher)
    restored = build_model(cfg)
    opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.001)
    batcher2 = StatefulBatcher(data, 2, 0)
    state = load_checkpoint(path, model=restored, optimizer=opt2, batcher=batcher2, config=cfg)
    assert state["run_id"] == "same-run" and state["step"] == 1
    torch.testing.assert_close(torch.randn(3), expected_random, rtol=0, atol=0)
    torch.testing.assert_close(update(restored, opt2, batcher2), expected_loss, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(path, model=restored, config={**cfg, "seed": 99})


def test_manifest_split_and_action_alignment(tmp_path):
    episode = {"latents": torch.arange(5.).reshape(5, 1, 1), "actions": torch.arange(4.).reshape(4, 1)}
    torch.save(episode, tmp_path / "episode.pt")
    manifest = {"schema_version": 1, "dataset_id": "d1", "representation_id": "r1", "action_id": "a1",
                "episodes": [{"episode_id": "one", "path": "episode.pt", "split": "train", "frames": 5}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = EpisodeWindowDataset(path, "train", 3, dataset_id="d1", representation_id="r1", action_id="a1")
    assert len(dataset) == 3
    assert dataset[1]["actions"].flatten().tolist() == [1., 2.]
    assert dataset[1]["latents"].flatten().tolist() == [1., 2., 3.]
    manifest["episodes"].append({**manifest["episodes"][0], "split": "test"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        EpisodeWindowDataset(path, "train", 3, dataset_id="d1", representation_id="r1", action_id="a1")


def test_legacy_dataset_conversion_preserves_outgoing_actions():
    source = {"video": torch.full((2, 3, 3, 8, 8), -1.), "action": torch.arange(12.).reshape(2, 3, 2)}
    converted = from_nanowm_batch(source)
    assert converted.observations.shape == (2, 3, 1, 3, 8, 8)
    assert torch.count_nonzero(converted.observations) == 0
    torch.testing.assert_close(converted.actions, source["action"][:, :-1])


def test_shared_configs_and_reject_changed_representations(monkeypatch):
    for key in ("VJEPA2_REPO", "VJEPA2_CHECKPOINT", "LATENT_MANIFEST", "ROBOJEPA_REPO", "ROBOJEPA_CONFIG", "ROBOJEPA_CHECKPOINT"):
        monkeypatch.setenv(key, "/staged/" + key)
    configs = [OmegaConf.to_container(load_config(ROOT / f"configs/latent/{name}.yaml"), resolve=True)
               for name in ("robojepa_frozen", "deterministic", "nanowm_diffusion", "nanowm_flow")]
    assert_comparable(configs)
    fine = OmegaConf.to_container(load_config(ROOT / "configs/latent/robojepa_finetune.yaml"), resolve=True)
    with pytest.raises(ValueError, match="separate"):
        assert_comparable([configs[0], fine])
    configs[1]["planning"]["cem"]["samples"] += 1
    with pytest.raises(ValueError, match="budgets differ"):
        assert_comparable(configs)


def test_cli_smoke_train_evaluate_without_wandb(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "wandb", None)
    config = ROOT / "configs/latent/smoke.yaml"
    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    common = ["--config", str(config), "--device", "cpu", "--smoke"]
    assert main(["train", *common, "--output", str(train_dir)]) == 0
    assert (train_dir / "latest.pt").is_file() and (train_dir / "best.pt").is_file()
    assert json.loads((train_dir / "train-summary.json").read_text())["run_id"] is None
    assert main(["evaluate", *common, "--output", str(eval_dir), "--checkpoint", str(train_dir / "latest.pt")]) == 0
    report = json.loads((eval_dir / "evaluate.json").read_text())
    assert len(report["per_horizon"]) == 2
    assert report["per_horizon"][0]["count"] == 12
