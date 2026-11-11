import pytest
import torch

from latent_world.contracts import Conditioning
from latent_world.evaluation import evaluate_mpc_episode
from latent_world.planning import CEMConfig, LatentCEMPlanner, MPCController
from latent_world.rollout import AutoregressiveRollout


class PointDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.rollout_mechanism = AutoregressiveRollout(1)

    def rollout(self, context, actions, **kwargs):
        return context[:, -1:] + actions.cumsum(1).unsqueeze(2)

    def encode(self, observations):
        return observations[..., :1, :1, :1].reshape(1, 1, 1, 1)


@pytest.mark.parametrize("elites", [1, 8])
def test_cem_batched_bounded_reproducible_and_useful(elites):
    model = PointDynamics().eval()
    planner = LatentCEMPlanner(model, [-.5], [.5], CEMConfig(horizon=3, samples=64, elites=elites,
        iterations=4, chunk_size=16))
    context = torch.zeros(2, 1, 1, 1)
    goal = torch.tensor([1., -1.]).reshape(2, 1, 1, 1)
    actions, info = planner.plan(context, goal, seed=42)
    second, _ = planner.plan(context, goal, seed=42)
    torch.testing.assert_close(actions, second, rtol=0, atol=0)
    assert (actions.abs() <= .5).all()
    assert ((model.rollout(context, actions)[:, -1:] - goal).square().mean() < .05)
    assert info["candidate_rollouts"] == 2 * 64 * 4


class PointEnvironment:
    def reset(self, seed, episode_id):
        self.position = 0.
        return self.observe(), Conditioning()

    def observe(self):
        return torch.full((1, 1, 1, 3, 1, 1), self.position)

    def step(self, action):
        self.position += float(action[0])
        distance = abs(.75 - self.position)
        return self.observe(), Conditioning(), {"success": distance < .05,
            "goal_distance": distance, "terminated": False}


def test_mpc_replans_and_reports_actual_environment_success():
    model = PointDynamics().eval()
    planner = LatentCEMPlanner(model, [-.3], [.3], CEMConfig(horizon=2, samples=96, elites=8,
        iterations=4, chunk_size=32))
    controller = MPCController(planner)
    report = evaluate_mpc_episode(model, controller, PointEnvironment(), torch.full((1, 1, 1, 3, 1, 1), .75),
                                  seed=1, max_steps=10, episode_id="analytic-fixture")
    assert report["success"]
    assert report["environment_goal_distance"] < .05
    assert report["steps"] > 1 and len(report["plans"]) == report["steps"]
    assert report["latent_goal_distance"] < .05 ** 2


def test_nonfinite_plans_fail_and_single_elite_has_finite_variance():
    model = PointDynamics().eval()
    planner = LatentCEMPlanner(model, [-1], [1], CEMConfig(horizon=1, samples=4, elites=1, iterations=2))
    context = torch.zeros(1, 1, 1, 1)
    actions, _ = planner.plan(context, context)
    assert torch.isfinite(actions).all()
    with pytest.raises(FloatingPointError):
        planner.plan(context, torch.full_like(context, float("nan")))
    with pytest.raises(ValueError):
        CEMConfig(samples=2, elites=3)
