"""Explicit component factories; register new research components without editing loops."""

from pathlib import Path

from omegaconf import OmegaConf

from .contracts import LatentSpec
from .encoders import CachedEncoder, FrameEncoder, TinyPatchBackbone, robojepa_encoder
from .model import LatentWorldModel
from .objectives import NanoWMObjective, NextLatentObjective
from .predictors import DeterministicPredictor, NanoWMPredictor, RoboJEPAPredictor
from .rollout import AutoregressiveRollout


class Registry:
    def __init__(self):
        self.factories = {}

    def register(self, name, factory):
        if name in self.factories:
            raise ValueError(f"Component already registered: {name}")
        self.factories[name] = factory

    def build(self, config, **context):
        params = dict(config)
        name = params.pop("kind")
        if name not in self.factories:
            raise ValueError(f"Unknown component {name!r}; available: {sorted(self.factories)}")
        return self.factories[name](**context, **params)


ENCODERS, PREDICTORS, OBJECTIVES, ROLLOUTS = Registry(), Registry(), Registry(), Registry()
ENCODERS.register("cached", CachedEncoder)
ENCODERS.register("robojepa", robojepa_encoder)
ENCODERS.register("tiny", lambda spec, frozen=True: FrameEncoder(
    TinyPatchBackbone(spec.channels), spec, (spec.height * 4, spec.width * 4), frozen=frozen))
PREDICTORS.register("deterministic", DeterministicPredictor)
PREDICTORS.register("robojepa", RoboJEPAPredictor.from_pretrained)
OBJECTIVES.register("next_latent", NextLatentObjective)
OBJECTIVES.register("nanowm", NanoWMObjective)
ROLLOUTS.register("autoregressive", AutoregressiveRollout)


def _nanowm(spec, action_dim, *, hidden_dim=256, depth=4, heads=8, frames=3,
            patch_size=1, action_mode="additive", pred_name="v", diffusion_steps=1000,
            zero_terminal_snr=True, **rollout):
    from diffusion import create_diffusion
    from models.nanowm import NanoWM

    backbone = NanoWM(input_size=spec.height, patch_size=patch_size, in_channels=spec.channels,
                     hidden_size=hidden_dim, depth=depth, num_heads=heads, num_frames=frames,
                     use_action=True, action_dim=action_dim, action_injection_type=action_mode, causal=True)
    process = create_diffusion(timestep_respacing="", noise_schedule="squaredcos_cap_v2", pred_name=pred_name,
                               diffusion_steps=diffusion_steps, snr_gamma=0., zero_terminal_snr=zero_terminal_snr)
    return NanoWMPredictor(backbone, process, spec, **rollout)


PREDICTORS.register("nanowm", _nanowm)


def load_config(path, overrides=(), _seen=None):
    """Small compositional YAML configs, independent of legacy Hydra globals."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError("Circular config inheritance")
    seen.add(path)
    cfg = OmegaConf.load(path)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = OmegaConf.merge(load_config(path.parent / parent, _seen=seen), cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def build_model(config):
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    spec = LatentSpec(**cfg["representation"]["shape"])
    encoder = ENCODERS.build(cfg["encoder"], spec=spec)
    predictor = PREDICTORS.build(cfg["predictor"], spec=spec, action_dim=cfg["action_dim"])
    objective = OBJECTIVES.build(cfg["objective"])
    rollout = ROLLOUTS.build(cfg["rollout"])
    if isinstance(predictor, NanoWMPredictor) != isinstance(objective, NanoWMObjective):
        raise ValueError("Native NanoWM predictors and native diffusion/flow objectives must be paired")
    if isinstance(predictor, NanoWMPredictor) and not encoder.frozen:
        raise ValueError("The native NanoWM comparison currently requires a frozen encoder")
    if rollout.context_frames > predictor.max_context:
        raise ValueError("Rollout context exceeds predictor capacity")
    return LatentWorldModel(encoder, predictor, objective, rollout, target_decay=cfg.get("target_decay", .99))
