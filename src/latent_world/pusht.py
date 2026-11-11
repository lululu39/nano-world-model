"""PushT-specific adapters around the existing planning and pretrained interfaces."""
import json
import bisect
import pickle
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from omegaconf import OmegaConf


class PushTActions:
    """Ordered [dx0,dy0,...,dx4,dy4] chunks in training-standardized units."""
    def __init__(self, mean, std):
        self.mean, self.std = torch.as_tensor(mean), torch.as_tensor(std)
        if self.mean.shape != (2,) or self.std.shape != (2,) or (self.std <= 0).any():
            raise ValueError("Expected two finite positive action standard deviations")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Nonfinite action statistics")

    def pack(self, relative_pixels):
        if relative_pixels.shape[-2:] != (5, 2):
            raise ValueError("One transition requires five ordered relative 2D actions")
        x = relative_pixels / 100.0
        return ((x - self.mean.to(x)) / self.std.to(x)).flatten(-2)

    def unpack(self, normalized):
        if normalized.shape[-1] != 10:
            raise ValueError("PushT model action dimension must be 10")
        x = normalized.reshape(*normalized.shape[:-1], 5, 2)
        # PushTEnv.step multiplies by 100 and adds the CURRENT agent position.
        return x * self.std.to(x) + self.mean.to(x)


class SDEncoderOnly(nn.Module):
    """Exact released VAE encoder/quantization weights, without a decoder module."""
    kind = "sd_vae"
    precision = "fp32"
    has_decoder = False

    def __init__(self, path):
        super().__init__()
        from diffusers.models.autoencoders.vae import Encoder
        cfg = json.loads((Path(path) / "config.json").read_text())
        self.encoder = Encoder(in_channels=cfg["in_channels"], out_channels=cfg["latent_channels"],
            down_block_types=tuple(cfg["down_block_types"]), block_out_channels=tuple(cfg["block_out_channels"]),
            layers_per_block=cfg["layers_per_block"], act_fn=cfg["act_fn"],
            norm_num_groups=cfg["norm_num_groups"], double_z=True)
        self.quant_conv = nn.Conv2d(2 * cfg["latent_channels"], 2 * cfg["latent_channels"], 1)
        self.load_state_dict(load_file(str(Path(path) / "encoder.safetensors")), strict=True)
        self.scaling_factor = cfg.get("scaling_factor", 0.18215)
        self.requires_grad_(False).eval()

    def encode(self, frames):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        with torch.autocast(device_type=frames.device.type, enabled=False):
            posterior = DiagonalGaussianDistribution(self.quant_conv(self.encoder(frames.float())))
            return posterior.sample() * self.scaling_factor


def load_nanowm(assets, protocol, device):
    from models import get_models
    from diffusion import create_diffusion
    from planning.diffusion_world_model import DiffusionWorldModel
    assets = Path(assets)
    cfg = OmegaConf.load(assets / "nanowm/config.yaml")
    original = OmegaConf.to_container(cfg, resolve=False)
    # Resolve legacy absent codec fields by verifying the actual weight shapes.
    assert cfg.model.arch == "NanoWM-B/2" and cfg.model.latent_size == 32
    assert cfg.dataset.frame_interval == 5 and cfg.dataset.spec.action_dim == 2
    assert cfg.dataset.loader.use_relative_actions and cfg.dataset.loader.action_scale == 100.0
    assert cfg.dataset.loader.normalize_action and cfg.model.n_context_frames == 1
    cfg.experiment.infra.compile = False
    model = get_models(cfg)
    weights = load_file(str(assets / "nanowm/model.safetensors"))
    result = model.load_state_dict(weights, strict=True)
    report = {"strict": True, "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
              "weight_shapes": {k: list(v.shape) for k,v in weights.items()}, "published_config": original,
              "parameters": sum(p.numel() for p in model.parameters()), "decoder_instantiated": False}
    del weights
    model = model.to(device).eval().requires_grad_(False)
    codec = SDEncoderOnly(assets / "vae").to(device)
    d = cfg.experiment.diffusion
    process = create_diffusion(timestep_respacing="", noise_schedule=d.noise_schedule,
        pred_name=d.pred_name, diffusion_steps=d.diffusion_steps, snr_gamma=d.snr_gamma,
        zero_terminal_snr=d.zero_terminal_snr)
    cfg.model.num_sampling_steps = protocol.num_sampling_steps
    cfg.model.scheduling_mode = protocol.scheduling_mode
    return DiffusionWorldModel(model, codec, process, cfg), cfg, report


def load_robojepa(weights, robo_repo, vjepa_repo, device, checkpoint=None):
    from .contracts import LatentSpec
    from .encoders import robojepa_encoder
    from .predictors import RoboJEPAPredictor
    spec = LatentSpec(channels=1664, height=16, width=16, views=1)
    encoder = robojepa_encoder(vjepa_repo, Path(weights) / "vjepa2_1_vitG_384.pt", spec, frozen=True)
    predictor = RoboJEPAPredictor.from_pretrained(robo_repo, Path(weights) / "config.json",
        Path(weights) / "cooldown_22m_150k.pth.tar", spec, 10, action_mode="linear",
        state_policy="zeros", embodiment_id=2, view_ids=[0])
    source = torch.load(Path(weights) / "cooldown_22m_150k.pth.tar", map_location="cpu", weights_only=True, mmap=True)
    state = source.get("predictor", source.get("model"))
    state = {k.removeprefix("module."):v for k,v in state.items()}
    # Actual value equality, not merely a count of successfully loaded keys.
    assert all(torch.equal(v, state[k]) for k,v in predictor.predictor.state_dict().items())
    report = {"strict": True, "pretrained_predictor_exactly_equal": True, "predictor_tensor_count": len(state),
              "encoder_frozen": True, "action_dim": 10, "embodiment_id": 2, "state_policy": "zeros",
              "input_size": [256,256], "token_layout": "view,row,column,channel", "grid": [1,16,16,1664]}
    del state, source
    if checkpoint:
        trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if trained["config"]["representation"] != "vjepa21g-256-ln-rowmajor-v1":
            raise ValueError("Fine-tuned checkpoint uses an incompatible representation")
        predictor.load_state_dict(trained["model"], strict=True)
        report["finetuned_step"] = trained["step"]
    return encoder.to(device).eval(), predictor.to(device), report


class RoboPlanningModel:
    """Use the original CEM interface and a one-frame autoregressive context."""
    def __init__(self, encoder, predictor):
        self.encoder, self.predictor = encoder.eval(), predictor.eval()
        self.context = None

    @torch.no_grad()
    def prepare_context(self, obs):
        self.context = self._encode(obs["visual"])

    def _encode(self, x):
        return self.encoder(((x + 1) / 2).unsqueeze(2))

    @torch.no_grad()
    def encode_obs(self, obs):
        return {"visual": self._encode(obs["visual"]).flatten(2), "proprio": None}

    @torch.no_grad()
    def rollout(self, obs_0, act):
        if self.context is None:
            raise RuntimeError("Call prepare_context before CEM")
        z = self.context.expand(act.shape[0], -1, -1, -1)
        frames = [z]
        for t in range(act.shape[1]):
            z = self.predictor.predict_next(z, act[:, t:t+1])
            frames.append(z)
        return {"visual": torch.cat(frames, dim=1).flatten(2), "proprio": None}, None


class PushTWindows:
    """Lazy real-video windows; reserve train episodes for tuning, never planning goals."""
    def __init__(self, assets, split, frames=4, seed=42):
        import numpy as np
        self.root = Path(assets) / "data/pusht_noise/train"
        with (self.root / "seq_lengths.pkl").open("rb") as stream:
            self.lengths = list(map(int, pickle.load(stream)))
        order = np.random.RandomState(seed).permutation(len(self.lengths)).tolist()
        cutoff = int(.95 * len(order))
        self.episodes = order[:cutoff] if split == "train" else order[cutoff:]
        if split not in {"train", "val"}:
            raise ValueError("Training windows use only the official training archive")
        self.frames = frames
        self.actions = torch.load(self.root / "rel_actions.pth", weights_only=False).float()
        stats = torch.load(Path(assets) / "action_stats.pt", weights_only=True)
        self.action_transform = PushTActions(stats["mean"], stats["std"])
        self.ends = []
        total = 0
        for ep in self.episodes:
            total += max(0, self.lengths[ep] - (frames - 1) * 5)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        from decord import VideoReader, cpu
        from torchvision.transforms.functional import resize
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = bisect.bisect_right(self.ends, index)
        ep = self.episodes[pos]
        start = index - (self.ends[pos-1] if pos else 0)
        reader = VideoReader(str(self.root / f"obses/episode_{ep:03d}.mp4"), ctx=cpu(0), num_threads=1)
        images = reader.get_batch([start + i*5 for i in range(self.frames)])
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images.asnumpy())
        images = resize(images.permute(0,3,1,2).float() / 255, [256,256], antialias=True)
        actions = self.actions[ep, start:start+(self.frames-1)*5].reshape(self.frames-1,5,2)
        return {"observations":images.unsqueeze(1), "actions":self.action_transform.pack(actions)}


def prediction_loss(predictor, z, actions):
    """One-frame-context teacher forcing and three-step free rollout, feature MAE only."""
    b, t, n, d = z.shape
    teacher = predictor(z[:, :-1].reshape(b*(t-1),1,n,d), actions.reshape(b*(t-1),1,10))
    teacher = (teacher.reshape(b,t-1,n,d).float() - z[:, 1:]).abs().mean()
    current = z[:, :1]
    losses = []
    for k in range(t-1):
        current = predictor.predict_next(current, actions[:, k:k+1])
        losses.append((current.float() - z[:, k+1:k+2]).abs().mean())
    rollout = torch.stack(losses).mean()
    return (teacher + rollout) / 2, {"teacher_mae":float(teacher.detach()), "rollout_mae":float(rollout.detach())}
