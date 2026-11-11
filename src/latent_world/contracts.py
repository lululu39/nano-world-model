from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentSpec:
    channels: int
    height: int
    width: int
    views: int = 1

    def __post_init__(self):
        if min(self.channels, self.height, self.width, self.views) < 1:
            raise ValueError("Latent dimensions must be positive")

    @property
    def tokens(self):
        return self.views * self.height * self.width

    def validate(self, z: Tensor):
        if z.ndim != 4 or z.shape[-2:] != (self.tokens, self.channels):
            raise ValueError(f"Expected [B,T,{self.tokens},{self.channels}], got {tuple(z.shape)}")

    def to_robojepa(self, z: Tensor):
        self.validate(z)
        return z.reshape(*z.shape[:2], self.views, self.height, self.width, self.channels)

    def to_nanowm(self, z: Tensor):
        if self.views != 1 or self.height != self.width:
            raise ValueError("The upstream NanoWM backbone requires a single square latent grid")
        return self.to_robojepa(z).squeeze(2).permute(0, 1, 4, 2, 3).contiguous()

    def from_nanowm(self, z: Tensor):
        result = z.permute(0, 1, 3, 4, 2).flatten(2, 3).contiguous()
        self.validate(result)
        return result


@dataclass
class Conditioning:
    """States correspond to observed frames; identities are constant over time."""

    states: Tensor | None = None       # [B,T,S]
    embodiment_ids: Tensor | None = None  # [B]
    view_ids: Tensor | None = None     # [B,V], indices in the pretrained view map

    def window(self, start=None, end=None):
        return replace(self, states=None if self.states is None else self.states[:, start:end])

    def repeat_interleave(self, repeats):
        return Conditioning(**{
            name: None if value is None else value.repeat_interleave(repeats, dim=0)
            for name, value in vars(self).items()
        })

    def to(self, device):
        return Conditioning(**{
            name: None if value is None else value.to(device)
            for name, value in vars(self).items()
        })


@dataclass
class SequenceBatch:
    actions: Tensor                  # [B,T-1,A], outgoing
    observations: Tensor | None = None  # [B,T,V,3,H,W], RGB in [0,1]
    latents: Tensor | None = None     # [B,T,N,D], frozen encoder cache
    conditioning: Conditioning | None = None
    valid: Tensor | None = None      # [B,T-1], valid transitions (prefix mask)

    def validate(self, spec: LatentSpec):
        if (self.observations is None) == (self.latents is None):
            raise ValueError("Provide exactly one of observations or cached latents")
        x = self.latents if self.latents is not None else self.observations
        if self.latents is not None:
            spec.validate(x)
        elif x.ndim != 6 or x.shape[2:4] != (spec.views, 3):
            raise ValueError("Observations must be [B,T,V,3,H,W]")
        if x.shape[1] < 2 or self.actions.ndim != 3 or self.actions.shape[:2] != (x.shape[0], x.shape[1] - 1):
            raise ValueError("A sequence needs T observations and T-1 outgoing actions")
        if self.conditioning is not None and self.conditioning.states is not None:
            if self.conditioning.states.shape[:2] != x.shape[:2]:
                raise ValueError("States must align with observations")
        if self.valid is not None:
            if self.valid.shape != self.actions.shape[:2] or self.valid.dtype != torch.bool:
                raise ValueError("valid must be a boolean [B,T-1] prefix mask")
            if (~self.valid[:, :-1] & self.valid[:, 1:]).any() or not self.valid.any():
                raise ValueError("valid must contain a nonempty prefix, never holes")

    def to(self, device):
        return SequenceBatch(**{
            name: None if value is None else value.to(device)
            for name, value in vars(self).items()
        })
