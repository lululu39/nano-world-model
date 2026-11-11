import pytest
import torch
from torch import nn

from latent_world.checkpoint import load_checkpoint, save_checkpoint
from latent_world.pusht import prediction_loss, transfer_loss_type


class ScalePredictor(nn.Module):
    max_context = 3

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))

    def forward(self, z, actions):
        return self.scale * z

    def predict_next(self, z, actions, conditioning=None):
        return self(z, actions)[:, -1:]


@pytest.mark.parametrize('context', [1, 3])
def test_mae_default_preserves_known_teacher_and_rollout_errors(context):
    model = ScalePredictor()
    z = torch.tensor([1., 2., 4., 8.], dtype=torch.float64).reshape(1, 4, 1, 1)
    actions = torch.zeros(1, 3, 10)
    loss, metrics = prediction_loss(model, z, actions, context)
    explicit, explicit_metrics = prediction_loss(model, z, actions, context, loss_type='mae')
    # Teacher errors: .5, 1, 2; free-rollout errors: .5, 1.75, 4.625.
    assert metrics == {'teacher_mae': 3.5/3, 'rollout_mae': 6.875/3}
    assert metrics == explicit_metrics and torch.equal(loss, explicit)
    torch.testing.assert_close(loss, torch.tensor((3.5+6.875)/6, dtype=torch.float64))


def test_mse_squares_before_averaging_and_backpropagates_through_feedback():
    model = ScalePredictor()
    z = torch.tensor([1., 2., 4., 8.], dtype=torch.float64).reshape(1, 4, 1, 1)
    loss, metrics = prediction_loss(model, z, torch.zeros(1, 3, 10), 3, loss_type='mse')
    assert metrics == {'teacher_mse': 1.75, 'rollout_mse': 8.234375}
    assert loss.item() == 4.9921875
    loss.backward()
    # Includes derivatives of w, w**2 and w**3 through autoregressive feedback.
    assert model.scale.grad.item() == pytest.approx(-15.822916666666666, rel=1e-6)


def test_unknown_loss_cannot_silently_fall_back_to_mae():
    assert transfer_loss_type({}) == 'mae'
    with pytest.raises(ValueError):
        transfer_loss_type({'loss_type': 'msee'})


def test_resume_cannot_change_loss_objective(tmp_path):
    class Batcher:
        def state_dict(self):
            return {}
    model = ScalePredictor()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path/'latest.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, batcher=Batcher(),
                    step=1, run_id=None, config={'loss_type':'mse'}, best=1., elapsed=1.)
    with pytest.raises(ValueError, match='Resume configuration differs'):
        load_checkpoint(path, model=model, config={'loss_type':'mae'}, restore_rng=False)
