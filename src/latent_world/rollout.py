from dataclasses import replace

import torch

from .conditioning import advance_state
from .contracts import Conditioning


class AutoregressiveRollout:
    """Roll out H actions to H predictions; never include context in returned targets."""

    def __init__(self, context_frames=1, *, detach_feedback=False, state_policy="required", state_transition=None):
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        self.context_frames = context_frames
        self.detach_feedback = detach_feedback
        self.state_policy = state_policy
        self.state_transition = state_transition

    def __call__(self, predictor, context, actions, *, history_actions=None, conditioning=None):
        if context.ndim != 4 or context.shape[1] < 1 or actions.ndim != 3 or context.shape[0] != actions.shape[0]:
            raise ValueError("Expected nonempty [B,T,N,D] context and [B,H,A] future actions")
        if self.context_frames > predictor.max_context:
            raise ValueError("Rollout context exceeds predictor capacity")
        b, t = context.shape[:2]
        if history_actions is None:
            if t != 1:
                raise ValueError("Multiple context frames require actual history_actions")
            history_actions = actions[:, :0]
        if history_actions.shape != (b, t - 1, actions.shape[-1]):
            raise ValueError("history_actions must describe the T-1 observed transitions")
        c = conditioning or Conditioning()
        if c.states is not None and c.states.shape[:2] != (b, t):
            raise ValueError("Conditioning states must align with context")
        predictions = []
        for step in range(actions.shape[1]):
            current = actions[:, step:step + 1]
            outgoing = torch.cat([history_actions, current], dim=1)
            prediction = predictor.predict_next(context[:, -self.context_frames:],
                                                outgoing[:, -self.context_frames:],
                                                c.window(-self.context_frames))
            if prediction.shape != context[:, :1].shape:
                raise ValueError("predict_next must return exactly one frame in the same representation")
            predictions.append(prediction)
            feedback = prediction.detach() if self.detach_feedback else prediction
            context = torch.cat([context, feedback], dim=1)[:, -self.context_frames:]
            history_actions = outgoing[:, -(context.shape[1] - 1):] if context.shape[1] > 1 else outgoing[:, :0]
            if c.states is not None and step + 1 < actions.shape[1]:
                state = (self.state_transition(c.states, current) if self.state_transition is not None
                         else advance_state(c.states, current, self.state_policy))
                c = replace(c, states=torch.cat([c.states, state], dim=1)[:, -self.context_frames:])
        return torch.cat(predictions, dim=1) if predictions else context[:, :0]
