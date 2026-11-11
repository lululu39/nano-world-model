import pytest
import torch
from torch import nn

from latent_world.pusht import (prediction_loss, prediction_rollout, RoboPlanningModel,
                                transfer_context_frames, validate_transfer_checkpoint)


class RecordingPredictor(nn.Module):
    max_context = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.25))
        self.calls = []

    def forward(self, z, actions, conditioning=None):
        self.calls.append((z.detach().clone(), actions.detach().clone()))
        # A causal function with nonzero dependence on both older frames and actions.
        return self.scale * (z + actions[..., :1, None]).cumsum(1)

    def predict_next(self, z, actions, conditioning=None):
        return self(z, actions, conditioning)[:, -1:]


def test_rollout_keeps_initial_observation_and_predicted_prefix_with_aligned_actions():
    model = RecordingPredictor()
    initial = torch.ones(1, 1, 2, 1)
    actions = torch.arange(30.).reshape(1, 3, 10)
    predicted = prediction_rollout(model, initial, actions, 3)
    assert [z.shape[1] for z, _ in model.calls] == [1, 2, 3]
    for k, (z, a) in enumerate(model.calls):
        torch.testing.assert_close(z[:, :1], initial)
        torch.testing.assert_close(z[:, 1:], predicted[:, :k])
        torch.testing.assert_close(a, actions[:, :k+1])
    # Gradients through the predicted feedback remain enabled, as in the old recipe.
    predicted[:, -1:].sum().backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


def test_teacher_forcing_and_rollout_share_history_without_future_target_leakage():
    model = RecordingPredictor()
    z = torch.arange(8.).reshape(1, 4, 2, 1)
    actions = torch.zeros(1, 3, 10)
    loss, _ = prediction_loss(model, z, actions, 3)
    assert [x.shape[1] for x, _ in model.calls] == [3, 1, 2, 3]
    rollout_calls = model.calls[1:]
    model.calls.clear()
    changed = z.clone(); changed[:, 1:] += 1000
    changed_loss, _ = prediction_loss(model, changed, actions, 3)
    assert changed_loss > loss
    for original, perturbed in zip(rollout_calls, model.calls[1:]):
        torch.testing.assert_close(original[0], perturbed[0])
        torch.testing.assert_close(original[1], perturbed[1])


@pytest.mark.parametrize("context", [1, 3])
def test_planning_and_training_rollout_match(context):
    predictor = RecordingPredictor()
    initial = torch.ones(1, 1, 2, 1)
    actions = torch.randn(2, 3, 10)
    expected = prediction_rollout(predictor, initial.expand(2,-1,-1,-1), actions, context)
    wm = RoboPlanningModel(nn.Identity(), predictor, context_frames=context)
    wm.context = initial
    actual, _ = wm.rollout(None, actions)
    torch.testing.assert_close(actual["visual"][:, 1:], expected.flatten(2))


def test_legacy_default_and_checkpoint_history_contract():
    assert transfer_context_frames({}) == 1
    assert transfer_context_frames({"frames":4,"rollout_context_frames":3}) == 3
    for invalid in [0, 4, 9, True, 1.5]:
        with pytest.raises(ValueError, match="rollout_context_frames"):
            transfer_context_frames({"rollout_context_frames":invalid})
    cfg = {"representation":"vjepa21g-256-ln-rowmajor-v1", "context_frames":1,
           "frame_interval":5, "action_dim":10, "state_policy":"zeros", "embodiment_id":2,
           "view_ids":[0], "action_statistics":{"mean":[0.,0.],"std":[1.,1.]}}
    stats = {"mean":torch.zeros(2), "std":torch.ones(2)}
    assert validate_transfer_checkpoint({"config":cfg}, stats) == 1
    assert validate_transfer_checkpoint({"config":dict(cfg, rollout_context_frames=3)}, stats) == 3
    predictor = RecordingPredictor()
    initial = torch.ones(1,1,2,1)
    actions = torch.randn(1,3,10)
    actual = prediction_rollout(predictor, initial, actions)
    previous = initial; expected = []
    for k in range(3):
        previous = predictor.predict_next(previous, actions[:, k:k+1]); expected.append(previous)
    torch.testing.assert_close(actual, torch.cat(expected,1), rtol=0, atol=0)
