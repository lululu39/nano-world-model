import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import ActionAdapter, AdditiveActions, FiLMActions
from .contracts import Conditioning


class DeterministicPredictor(nn.Module):
    """Frame-causal next-latent transformer, inspired by the DINO-WM paradigm."""

    stochastic = False

    def __init__(self, spec, action_dim, hidden_dim=256, depth=4, heads=8, max_context=8, action_mode="additive"):
        super().__init__()
        if action_mode not in {"additive", "film"}:
            raise ValueError(f"Unknown action conditioning: {action_mode}")
        self.spec, self.max_context = spec, max_context
        self.input = nn.Linear(spec.channels, hidden_dim)
        self.spatial = nn.Parameter(torch.randn(1, 1, spec.tokens, hidden_dim) * .02)
        self.temporal = nn.Parameter(torch.randn(1, max_context, 1, hidden_dim) * .02)
        cls = AdditiveActions if action_mode == "additive" else FiLMActions
        self.action_conditioner = cls(action_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout=0.,
                                       batch_first=True, norm_first=True, activation="gelu")
            for _ in range(depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, spec.channels))

    def forward(self, z, actions, conditioning=None):
        self.spec.validate(z)
        b, t, n, _ = z.shape
        if t > self.max_context or actions.shape[:2] != (b, t):
            raise ValueError("Predictor needs one outgoing action per context frame within max_context")
        if conditioning is not None and conditioning.states is not None:
            raise ValueError("This visual deterministic baseline does not consume proprio")
        x = self.action_conditioner(self.input(z), actions) + self.spatial + self.temporal[:, :t]
        x = x.flatten(1, 2)
        time = torch.arange(t, device=z.device).repeat_interleave(n)
        mask = time[:, None] < time[None, :]  # block future frames, permit all same-frame patches
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.output(x).reshape(b, t, n, -1)

    def predict_next(self, z, actions, conditioning=None):
        return self(z, actions, conditioning)[:, -1:]


class RoboJEPAPredictor(nn.Module):
    """Strict adapter to the released predictor, with explicit task semantics."""

    stochastic = False

    def __init__(self, predictor, spec, action_dim, *, action_mode="pad", state_policy="required",
                 embodiment_id=None, view_ids=None, frozen=False):
        super().__init__()
        self.predictor, self.spec = predictor, spec
        if predictor.encoder_layernorm is not True:
            raise ValueError("Released RoboJEPA adapter expects layer-normalized latent features")
        self.max_context = int(predictor.num_frames)
        self.action_adapter = ActionAdapter(action_dim, predictor.action_dim, action_mode)
        self.state_policy = state_policy
        self.embodiment_id = embodiment_id
        self.view_ids = view_ids
        self.frozen = frozen
        self.predictor.requires_grad_(not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.predictor.eval()
        return self

    def forward(self, z, actions, conditioning=None):
        self.spec.validate(z)
        b, t = z.shape[:2]
        if actions.shape[:2] != (b, t) or t > self.max_context:
            raise ValueError("RoboJEPA needs one outgoing action per available frame")
        c = conditioning or Conditioning()
        states = c.states
        if self.predictor.state_tokens:
            if states is None:
                if self.state_policy != "zeros":
                    raise ValueError("Pretrained predictor needs states or an explicit action-only zero-state policy")
                states = z.new_zeros(b, t, self.predictor.state_dim)
            if states.shape[:2] != (b, t) or states.shape[-1] > self.predictor.state_dim:
                raise ValueError("State shape does not fit the pretrained interface")
            states = F.pad(states, (0, self.predictor.state_dim - states.shape[-1]))
        emb = c.embodiment_ids
        if emb is None and self.embodiment_id is not None:
            emb = torch.full((b,), self.embodiment_id, device=z.device, dtype=torch.long)
        if self.predictor.is_multiembodiment:
            if emb is None or emb.shape != (b,) or (emb < 0).any() or (emb >= len(self.predictor.embodiment_list)).any():
                raise ValueError("Choose a valid pretrained embodiment identity explicitly")
        views = c.view_ids
        if views is None and self.view_ids is not None:
            views = torch.tensor(self.view_ids, device=z.device, dtype=torch.long)[None].expand(b, -1)
        if self.predictor.view_map is not None:
            if views is None or views.shape != (b, self.spec.views):
                raise ValueError("Provide one pretrained view index per camera")
            if (views < 0).any() or (views >= len(self.predictor.view_map)).any():
                raise ValueError("View index is outside the pretrained view map")
        output, _, _, _ = self.predictor(
            self.spec.to_robojepa(z), actions=self.action_adapter(actions), states=states,
            embodiment_ids=emb, view_map=views, use_cache=False,
        )
        return output.reshape(b, t, self.spec.tokens, self.spec.channels)

    def predict_next(self, z, actions, conditioning=None):
        return self(z, actions, conditioning)[:, -1:]

    @classmethod
    def from_pretrained(cls, repo, config, checkpoint, spec, action_dim, **kwargs):
        from .external import load_predictor_constructor

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Stage pretrained RoboJEPA weights first: {path}")
        params = json.loads(Path(config).read_text())
        if params.pop("encoder_version", "v21") != "v21" or params["embed_dim"] != spec.channels:
            raise ValueError("Predictor and representation must use V-JEPA 2.1-G features")
        params["attn_impl"] = "sdpa"
        params["use_activation_checkpointing"] = False
        predictor = load_predictor_constructor(repo)(**params)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        weights = state.get("predictor", state.get("model"))
        if weights is None:
            raise KeyError("Expected a released checkpoint with predictor or model state")
        predictor.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
        return cls(predictor, spec, action_dim, **kwargs)


class NanoWMPredictor(nn.Module):
    """The original NanoWM backbone/process in a token-space next-step interface."""

    stochastic = True

    def __init__(self, backbone, process, spec, *, context_frames=1, sampling_steps=10,
                 scheduling="sequential", history_stabilization=0.):
        super().__init__()
        if spec.views != 1 or spec.height != spec.width:
            raise ValueError("NanoWM currently requires a single square latent grid")
        if not 1 <= context_frames < backbone.num_frames:
            raise ValueError("context_frames must leave at least one future frame")
        self.backbone, self.process, self.spec = backbone, process, spec
        self.max_context = context_frames
        self.sampling_steps = sampling_steps
        self.scheduling = scheduling
        self.history_stabilization = history_stabilization

    def predict_next(self, z, actions, conditioning=None):
        from diffusion.df_sample import dfot_sample

        self.spec.validate(z)
        b, t = z.shape[:2]
        if actions.shape[:2] != (b, t) or t > self.max_context:
            raise ValueError("NanoWM needs aligned history actions and a next action")
        if conditioning is not None and conditioning.states is not None:
            raise ValueError("The NanoWM visual baseline does not consume proprio")
        missing = self.max_context - t
        context = torch.cat([z[:, :1].expand(-1, missing, -1, -1), z], dim=1)
        # Upstream shifts actions internally: a_t belongs at the source frame.
        a = F.pad(actions, (0, 0, missing, self.backbone.num_frames - self.max_context))
        result = dfot_sample(
            self.process, self.backbone,
            shape=(b, self.backbone.num_frames, self.spec.channels, self.spec.height, self.spec.width),
            context=self.spec.to_nanowm(context), n_context_frames=self.max_context,
            scheduling_mode=self.scheduling, num_sampling_steps=self.sampling_steps,
            model_kwargs={"action": a}, device=z.device, progress=False,
            n_generate_frames=1, history_stabilization_level=self.history_stabilization,
        )
        return self.spec.from_nanowm(result[:, self.max_context:self.max_context + 1])
