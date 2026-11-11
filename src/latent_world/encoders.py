from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import LatentSpec


class FrameEncoder(nn.Module):
    """Differentiable, per-frame patch encoding with explicit preprocessing."""

    def __init__(self, backbone, spec: LatentSpec, input_size, *, frozen=True, layernorm=True):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.input_size = tuple(input_size)
        self.frozen = frozen
        self.layernorm = layernorm
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.backbone.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, observations):
        if observations.ndim != 6 or observations.shape[2:4] != (self.spec.views, 3):
            raise ValueError("Expected RGB [B,T,V,3,H,W] in [0,1]")
        b, t, v, c, h, w = observations.shape
        x = observations.reshape(b * t * v, c, h, w).float()
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            tokens = self.backbone(x.unsqueeze(2))  # official image tokenizer, one frame
        expected = (b * t * v, self.spec.height * self.spec.width, self.spec.channels)
        if tokens.shape != expected:
            raise ValueError(f"Encoder token layout {tuple(tokens.shape)} differs from {expected}")
        z = tokens.reshape(b, t, v * self.spec.height * self.spec.width, -1).float()
        self.spec.validate(z)
        return F.layer_norm(z, (z.shape[-1],)) if self.layernorm else z


class CachedEncoder(nn.Module):
    frozen = True

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, observations):
        raise ValueError("CachedEncoder needs SequenceBatch.latents; it cannot encode RGB")


class TinyPatchBackbone(nn.Module):
    """Offline smoke fixture, never presented as a pretrained encoder."""

    def __init__(self, channels, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x.squeeze(2)).flatten(2).transpose(1, 2)


def robojepa_encoder(repo, checkpoint, spec, *, frozen=True, input_size=(256, 256)):
    from .external import load_vjepa_encoder

    if spec.channels != 1664 or tuple(input_size) != (spec.height * 16, spec.width * 16):
        raise ValueError("RoboJEPA requires 1664 channels and a patch-16 V-JEPA 2.1-G grid")
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Stage the V-JEPA 2.1-G checkpoint first: {path}")
    backbone = load_vjepa_encoder(repo)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weights = {k.removeprefix("module.backbone."): v for k, v in state["target_encoder"].items()}
    backbone.load_state_dict(weights, strict=True)
    return FrameEncoder(backbone, spec, input_size, frozen=frozen, layernorm=True)
