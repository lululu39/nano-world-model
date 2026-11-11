import copy

import torch
from torch import nn


class LatentWorldModel(nn.Module):
    """Composition root; architecture components own their implementation."""

    def __init__(self, encoder, predictor, objective, rollout, *, target_decay=.99):
        super().__init__()
        if not 0 <= target_decay <= 1:
            raise ValueError("target_decay must be in [0,1]; 1 keeps the pretrained target fixed")
        self.encoder, self.predictor, self.objective = encoder, predictor, objective
        self.spec = encoder.spec
        if self.spec != predictor.spec:
            raise ValueError("Encoder and predictor representations differ")
        self.rollout_mechanism = rollout
        self.target_decay = target_decay
        self.target_encoder = None if encoder.frozen else copy.deepcopy(encoder)
        if self.target_encoder is not None:
            self.target_encoder.frozen = True
            self.target_encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def encode(self, observations):
        return self.encoder(observations)

    def rollout(self, context, actions, **kwargs):
        return self.rollout_mechanism(self.predictor, context, actions, **kwargs)

    def loss(self, batch):
        batch.validate(self.spec)
        if batch.latents is not None:
            if not self.encoder.frozen:
                raise ValueError("Cached features cannot be used for encoder fine-tuning")
            online = batch.latents.detach()
            targets = online
        else:
            online = self.encode(batch.observations)
            with torch.no_grad():
                targets = online.detach() if self.target_encoder is None else self.target_encoder(batch.observations)
        return self.objective(self, batch, online, targets)

    @torch.no_grad()
    def update_target(self):
        """Call only after a successful optimizer update, never during forward."""
        if self.target_encoder is None:
            return
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online, 1 - self.target_decay)
        for target, online in zip(self.target_encoder.buffers(), self.encoder.buffers(), strict=True):
            target.copy_(online)
