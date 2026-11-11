import torch
from torch import nn
from torch.nn import functional as F


class AdditiveActions(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, tokens, actions):
        return tokens + self.projection(actions).unsqueeze(2)


class FiLMActions(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(action_dim, 2 * hidden_dim)

    def forward(self, tokens, actions):
        scale, shift = self.projection(actions).unsqueeze(2).chunk(2, dim=-1)
        return tokens * (1 + scale) + shift


class ActionAdapter(nn.Module):
    """Explicit task-to-pretrained action interface; normalization is upstream."""

    def __init__(self, input_dim, output_dim, mode="pad"):
        super().__init__()
        if mode not in {"pad", "linear"} or (mode == "pad" and input_dim > output_dim):
            raise ValueError("Use pad for native semantics or linear for a learned task adapter")
        self.input_dim, self.output_dim, self.mode = input_dim, output_dim, mode
        self.projection = nn.Linear(input_dim, output_dim) if mode == "linear" else None

    def forward(self, actions):
        if actions.shape[-1] != self.input_dim:
            raise ValueError("Action dimension does not match the configured task interface")
        return self.projection(actions) if self.projection is not None else F.pad(actions, (0, self.output_dim - self.input_dim))


def advance_state(states, action, policy):
    """No future state labels: propagate the last available state explicitly."""
    if states is None:
        return None
    if policy == "zeros":
        if torch.count_nonzero(states):
            raise ValueError("zeros state policy is valid only for action-only, zero-state inputs")
        return torch.zeros_like(states[:, -1:])
    if policy == "delta":
        if action.shape[-1] != states.shape[-1]:
            raise ValueError("delta policy requires state and action in identical coordinates/units")
        return states[:, -1:] + action
    raise ValueError("Rollout with proprio requires an explicit state transition policy")
