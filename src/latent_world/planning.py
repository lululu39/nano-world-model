from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .contracts import Conditioning


@contextmanager
def sampling_seed(seed, device):
    """Isolate sampler randomness from CEM candidates, training and callers."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


def goal_distance(prediction, goal, metric="mse"):
    if goal.shape[1] != 1 or prediction.shape[0] != goal.shape[0] or prediction.shape[2:] != goal.shape[2:]:
        raise ValueError("Goal must be one frame in the same token representation")
    if metric == "mse":
        return (prediction.float() - goal.float()).square().mean((-1, -2))
    if metric == "cosine":
        return (1 - torch.nn.functional.cosine_similarity(prediction.float(), goal.float(), dim=-1)).mean(-1)
    raise ValueError(f"Unknown goal metric: {metric}")


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 5
    samples: int = 128
    elites: int = 16
    iterations: int = 5
    initial_std: float = 1.
    min_std: float = .05
    chunk_size: int = 32
    metric: str = "mse"
    risk_samples: int = 1

    def __post_init__(self):
        if min(self.horizon, self.samples, self.elites, self.iterations, self.chunk_size, self.risk_samples) < 1:
            raise ValueError("CEM budgets must be positive")
        if self.elites > self.samples or self.min_std <= 0 or self.initial_std < self.min_std:
            raise ValueError("Invalid elite count or CEM standard deviation")


class LatentCEMPlanner:
    """One planner for deterministic, pretrained, diffusion and flow dynamics."""

    def __init__(self, model, action_low, action_high, config=None):
        self.model = model
        self.config = config or CEMConfig()
        self.low = torch.as_tensor(action_low, dtype=torch.float32)
        self.high = torch.as_tensor(action_high, dtype=torch.float32)
        if self.low.ndim != 1 or self.low.shape != self.high.shape or not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or not (self.low < self.high).all():
            raise ValueError("Provide finite per-dimension action bounds in model coordinates")

    @torch.no_grad()
    def plan(self, context, goal, *, history_actions=None, conditioning=None, seed=0, warm_start=None):
        if self.model.training:
            raise ValueError("Call model.eval() before planning")
        cfg = self.config
        b, a = context.shape[0], self.low.numel()
        low, high = self.low.to(context), self.high.to(context)
        shape = (b, cfg.horizon, a)
        mu = ((low + high) / 2).expand(shape).clone() if warm_start is None else warm_start.to(context).clone()
        if mu.shape != shape:
            raise ValueError("Warm start has the wrong batch/horizon/action shape")
        mu = mu.clamp(low, high)
        std = torch.full_like(mu, cfg.initial_std)
        generator = torch.Generator(device=context.device).manual_seed(seed)
        c = conditioning or Conditioning()
        best_cost = context.new_full((b,), float("inf"))
        best_actions = mu.clone()
        history = []
        for iteration in range(cfg.iterations):
            noise = torch.randn((b, cfg.samples, cfg.horizon, a), device=context.device,
                                dtype=context.dtype, generator=generator)
            samples = (mu[:, None] + std[:, None] * noise).clamp(low, high)
            samples[:, 0] = mu  # retain the current mean as an evaluated candidate
            costs = []
            for start in range(0, cfg.samples, cfg.chunk_size):
                candidates = samples[:, start:start + cfg.chunk_size]
                count = candidates.shape[1]
                z = context.repeat_interleave(count, dim=0)
                g = goal.repeat_interleave(count, dim=0)
                h = None if history_actions is None else history_actions.repeat_interleave(count, dim=0)
                candidate_cost = 0
                for replicate in range(cfg.risk_samples):
                    with sampling_seed(seed + 1000003 * iteration + 1009 * start + replicate, context.device):
                        prediction = self.model.rollout(z, candidates.flatten(0, 1), history_actions=h,
                                                        conditioning=c.repeat_interleave(count))
                    candidate_cost = candidate_cost + goal_distance(prediction[:, -1:], g, cfg.metric)[:, 0]
                costs.append((candidate_cost / cfg.risk_samples).reshape(b, count))
            costs = torch.cat(costs, dim=1)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("Nonfinite CEM costs; stop and inspect the model")
            values, indices = costs.topk(cfg.elites, dim=1, largest=False)
            elites = samples.gather(1, indices[:, :, None, None].expand(-1, -1, cfg.horizon, a))
            improved = values[:, 0] < best_cost
            best_cost = torch.where(improved, values[:, 0], best_cost)
            best_actions = torch.where(improved[:, None, None], elites[:, 0], best_actions)
            mu = elites.mean(1)
            std = elites.std(1, unbiased=False).clamp_min(cfg.min_std)
            history.append(values[:, 0].cpu().tolist())
        return best_actions, {"cost_history": history, "predicted_goal_distance": best_cost.cpu().tolist(),
                              "candidate_rollouts": b * cfg.samples * cfg.iterations * cfg.risk_samples}


class MPCController:
    """Replan from newly observed context; execute only an explicit plan prefix."""

    def __init__(self, planner, execute_steps=1):
        if not 1 <= execute_steps <= planner.config.horizon:
            raise ValueError("execute_steps must lie within the planning horizon")
        self.planner, self.execute_steps, self.previous = planner, execute_steps, None

    def reset(self):
        self.previous = None

    def plan(self, context, goal, **kwargs):
        warm = None
        if self.previous is not None:
            warm = torch.cat([self.previous[:, self.execute_steps:],
                              self.previous[:, -1:].expand(-1, self.execute_steps, -1)], dim=1)
        actions, info = self.planner.plan(context, goal, warm_start=warm, **kwargs)
        self.previous = actions
        return actions[:, :self.execute_steps], info
