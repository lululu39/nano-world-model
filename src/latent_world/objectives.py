import torch
from torch import nn
from torch.nn import functional as F

from .contracts import Conditioning


def transition_loss(prediction, target, metric="mse", valid=None):
    if prediction.shape != target.shape:
        raise ValueError("Predictions and targets must have identical shapes")
    if metric == "mse":
        errors = (prediction.float() - target.float()).square().mean((-1, -2))
    elif metric == "mae":
        errors = (prediction.float() - target.float()).abs().mean((-1, -2))
    elif metric == "cosine":
        errors = (1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean(-1)
    else:
        raise ValueError(f"Unknown latent metric: {metric}")
    if valid is None:
        return errors.mean()
    weights = valid.to(errors)
    if not weights.any():
        raise ValueError("Loss requires at least one valid transition")
    return (errors * weights).sum() / weights.sum()


class NextLatentObjective(nn.Module):
    def __init__(self, metric="mae", rollout_weight=0., rollout_steps=2):
        super().__init__()
        if metric not in {"mae", "mse", "cosine"} or rollout_weight < 0 or rollout_steps < 1:
            raise ValueError("Invalid next-latent objective configuration")
        self.metric, self.rollout_weight, self.rollout_steps = metric, rollout_weight, rollout_steps

    def forward(self, model, batch, online, targets):
        conditioning = batch.conditioning or Conditioning()
        # Use the same rolling context at training and planning. This also
        # supports long clips without silently exceeding a checkpoint's window.
        predictions = []
        for t in range(batch.actions.shape[1]):
            start = max(0, t + 1 - model.rollout_mechanism.context_frames)
            predictions.append(model.predictor.predict_next(
                online[:, start:t + 1], batch.actions[:, start:t + 1], conditioning.window(start, t + 1)))
        teacher = transition_loss(torch.cat(predictions, 1), targets[:, 1:], self.metric, batch.valid)
        loss = teacher
        metrics = {"teacher_loss": teacher.detach()}
        if self.rollout_weight:
            steps = min(self.rollout_steps, batch.actions.shape[1])
            predicted = model.rollout(online[:, :1], batch.actions[:, :steps], conditioning=conditioning.window(0, 1))
            valid = None if batch.valid is None else batch.valid[:, :steps]
            ar = transition_loss(predicted, targets[:, 1:steps + 1], self.metric, valid)
            loss = loss + self.rollout_weight * ar
            metrics["rollout_loss"] = ar.detach()
        return loss, metrics


class NanoWMObjective(nn.Module):
    """Native diffusion/flow training; the baseline equations are unchanged."""

    def __init__(self, timestep_sampling="uniform", logit_mean=0., logit_std=1., mode="diffusion_forcing"):
        super().__init__()
        if mode not in {"diffusion_forcing", "full_seq_diffusion"}:
            raise ValueError("Unknown diffusion training mode")
        self.timestep_sampling = timestep_sampling
        self.logit_mean, self.logit_std, self.mode = logit_mean, logit_std, mode

    def forward(self, model, batch, online, targets):
        from diffusion import sample_training_timesteps

        if not model.encoder.frozen:
            raise ValueError("Controlled NanoWM baselines use a frozen encoder")
        if batch.valid is not None and not batch.valid.all():
            raise ValueError("Native NanoWM loss requires full clips; exclude padded clips in the shared protocol")
        predictor = model.predictor
        if online.shape[1] != predictor.backbone.num_frames:
            raise ValueError("Native NanoWM training clip length must equal backbone.num_frames")
        x = model.spec.to_nanowm(online)
        shape = x.shape[:2] if self.mode == "diffusion_forcing" else x.shape[:1]
        t = sample_training_timesteps(shape, predictor.process.num_timesteps,
            strategy=self.timestep_sampling, logit_normal_mean=self.logit_mean,
            logit_normal_std=self.logit_std, device=x.device)
        actions = F.pad(batch.actions, (0, 0, 0, 1))
        result = predictor.process.training_losses(predictor.backbone, x, t, {"action": actions})
        loss = result["loss"].mean()
        return loss, {"native_loss": loss.detach()}
