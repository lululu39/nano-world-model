from dataclasses import replace
from pathlib import Path

import pytest
import torch

from latent_world.cli import SmokeDataset
from latent_world.contracts import Conditioning, LatentSpec, SequenceBatch
from latent_world.data import collate_sequences
from latent_world.evaluation import evaluate_prediction
from latent_world.factory import build_model, load_config
from latent_world.planning import sampling_seed
from latent_world.predictors import DeterministicPredictor
from latent_world.rollout import AutoregressiveRollout


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/latent/smoke.yaml"


def tiny(frozen=True):
    cfg = load_config(CONFIG, [f"encoder.frozen={str(frozen).lower()}"])
    model = build_model(cfg)
    data = SmokeDataset(model.spec, 3, 2, 42)
    return model, collate_sequences([data[0], data[1]])


@pytest.mark.parametrize("frozen", [True, False])
def test_encoder_gradient_routing_and_target(frozen):
    model, batch = tiny(frozen)
    model.train()
    assert model.encoder.backbone.training is (not frozen)
    loss, _ = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    encoder_grads = [p.grad for p in model.encoder.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in encoder_grads) is (not frozen)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    if not frozen:
        assert not model.target_encoder.training
        assert all(p.grad is None for p in model.target_encoder.parameters())
        before = [p.clone() for p in model.target_encoder.parameters()]
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        optimizer.step()
        model.update_target()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.target_encoder.parameters()))
        with pytest.raises(ValueError, match="Cached features"):
            model.loss(SequenceBatch(actions=batch.actions, latents=torch.zeros(2, 3, 4, 8)))


def test_deterministic_predictor_is_frame_causal_and_action_sensitive():
    spec = LatentSpec(8, 2, 2)
    for mode in ("additive", "film"):
        model = DeterministicPredictor(spec, 2, hidden_dim=32, depth=2, heads=4, max_context=3, action_mode=mode).eval()
        z, a = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 2, requires_grad=True)
        output = model(z, a)
        perturbed_z, perturbed_a = z.clone(), a.detach().clone()
        perturbed_z[:, 2] += 100
        perturbed_a[:, 2] -= 100
        torch.testing.assert_close(output[:, :2], model(perturbed_z, perturbed_a)[:, :2], rtol=0, atol=0)
        grad = torch.autograd.grad(output[:, 1].square().sum(), a)[0]
        assert grad[:, 1].abs().sum() > 0
        assert torch.count_nonzero(grad[:, 2]) == 0
        torch.testing.assert_close(model(z, a), model(z, a), rtol=0, atol=0)


class Integrator(torch.nn.Module):
    max_context = 3
    stochastic = False

    def predict_next(self, context, actions, conditioning=None):
        return context[:, -1:] + actions[:, -1:, None, :1]


def test_rollout_alignment_and_no_future_ground_truth():
    rollout = AutoregressiveRollout(3)
    context = torch.tensor([0., 2.]).reshape(1, 2, 1, 1)
    history = torch.tensor([2.]).reshape(1, 1, 1)
    actions = torch.tensor([3., 5., 7.]).reshape(1, 3, 1).requires_grad_()
    prediction = rollout(Integrator(), context, actions, history_actions=history)
    assert prediction.flatten().tolist() == [5., 10., 17.]
    prediction[:, -1].sum().backward()
    assert actions.grad.flatten().tolist() == [1., 1., 1.]
    with pytest.raises(ValueError, match="history_actions"):
        rollout(Integrator(), context, actions)
    empty = rollout(Integrator(), context[:, :1], actions[:, :0])
    assert empty.shape == (1, 0, 1, 1)


def test_state_propagation_never_uses_future_labels():
    class UsesState(Integrator):
        def predict_next(self, context, actions, conditioning=None):
            return conditioning.states[:, -1:, None, :1] + actions[:, -1:, None, :1]
    z = torch.zeros(1, 1, 1, 1)
    actions = torch.ones(1, 3, 1)
    c = Conditioning(states=torch.zeros(1, 1, 1))
    prediction = AutoregressiveRollout(2, state_policy="delta")(UsesState(), z, actions, conditioning=c)
    assert prediction.flatten().tolist() == [1., 2., 3.]
    with pytest.raises(ValueError, match="explicit state"):
        AutoregressiveRollout(2)(UsesState(), z, actions, conditioning=c)


@pytest.mark.parametrize("pred_name", ["v", "flow"])
def test_native_nanowm_loss_and_rollout_compatibility(pred_name):
    from diffusion import sample_training_timesteps
    from latent_world.factory import _nanowm
    from latent_world.encoders import CachedEncoder
    from latent_world.model import LatentWorldModel
    from latent_world.objectives import NanoWMObjective

    spec = LatentSpec(4, 2, 2)
    predictor = _nanowm(spec, 2, hidden_dim=32, depth=2, heads=4, frames=3,
                        pred_name=pred_name, diffusion_steps=20, sampling_steps=2, context_frames=2)
    model = LatentWorldModel(CachedEncoder(spec), predictor, NanoWMObjective(), AutoregressiveRollout(2))
    z, a = torch.randn(2, 3, 4, 4), torch.randn(2, 2, 2)
    batch = SequenceBatch(actions=a, latents=z)
    with sampling_seed(11, "cpu"):
        adapted, _ = model.loss(batch)
    with sampling_seed(11, "cpu"):
        t = sample_training_timesteps((2, 3), 20, strategy="uniform", logit_normal_mean=0., logit_normal_std=1.)
        native = predictor.process.training_losses(predictor.backbone, spec.to_nanowm(z), t,
                                                    {"action": torch.nn.functional.pad(a, (0, 0, 0, 1))})["loss"].mean()
    torch.testing.assert_close(adapted, native, rtol=0, atol=0)
    adapted.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    # Load an upstream-style backbone state without key rewriting.
    predictor.backbone.load_state_dict(predictor.backbone.state_dict(), strict=True)
    with torch.no_grad(), sampling_seed(12, "cpu"):
        prediction = model.rollout(z[:, :2], a, history_actions=a[:, :1])
    assert prediction.shape == (2, 2, 4, 4) and torch.isfinite(prediction).all()


def test_native_sampler_receives_real_history_actions(monkeypatch):
    from latent_world.factory import _nanowm
    import diffusion.df_sample
    spec = LatentSpec(4, 2, 2)
    predictor = _nanowm(spec, 1, hidden_dim=32, depth=2, heads=4, frames=4, context_frames=3)
    seen = []
    def sampler(*args, **kwargs):
        seen.append(kwargs["model_kwargs"]["action"])
        return torch.zeros(kwargs["shape"])
    monkeypatch.setattr(diffusion.df_sample, "dfot_sample", sampler)
    predictor.predict_next(torch.zeros(1, 2, 4, 4), torch.tensor([[[2.], [3.]]]))
    assert seen[0].flatten().tolist() == [0., 2., 3., 0.]


def test_paired_sensitivity_rng_and_masked_prediction_metrics():
    model, batch = tiny()
    batch.valid = torch.tensor([[True, True], [True, False]])
    report = evaluate_prediction(model, batch)
    assert [x["count"] for x in report["per_horizon"]] == [2, 1]
    assert all(x["zero_action_sensitivity_mse"] > 0 for x in report["per_horizon"])
    zero = evaluate_prediction(model, replace(batch, actions=torch.zeros_like(batch.actions)))
    assert all(x["zero_action_sensitivity_mse"] == 0 for x in zero["per_horizon"])
    # Invalid action/frame and non-prefix padding fail before training.
    with pytest.raises(ValueError, match="T-1"):
        model.loss(replace(batch, actions=batch.actions[:, :1]))
    with pytest.raises(ValueError, match="holes"):
        model.loss(replace(batch, valid=torch.tensor([[False, True], [True, True]])))
