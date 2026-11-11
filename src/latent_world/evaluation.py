from contextlib import contextmanager
import time

import torch

from .contracts import Conditioning
from .planning import goal_distance, sampling_seed


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def evaluating(model):
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(mode)


def evaluate_prediction(model, batch, *, seed=0, context_frames=1):
    """Open-loop errors and paired action interventions on a held-out batch."""
    batch.validate(model.spec)
    with evaluating(model):
        if batch.latents is not None and not model.encoder.frozen:
            raise ValueError("Fine-tuned encoders require fresh observations for evaluation")
        z = batch.latents if batch.latents is not None else model.encode(batch.observations)
        if not 1 <= context_frames < z.shape[1]:
            raise ValueError("Evaluation context must leave future targets")
        context, targets = z[:, :context_frames], z[:, context_frames:]
        actions = batch.actions[:, context_frames - 1:]
        history = batch.actions[:, :context_frames - 1]
        conditioning = (batch.conditioning or Conditioning()).window(0, context_frames)
        def predict(a):
            with sampling_seed(seed, z.device):
                return model.rollout(context, a, history_actions=history, conditioning=conditioning)
        prediction = predict(actions)
        # A zero-action intervention is explicit; it is not a counterfactual label.
        counterfactual = predict(torch.zeros_like(actions))
        errors = (prediction.float() - targets.float()).square().mean((-1, -2))
        mae = (prediction.float() - targets.float()).abs().mean((-1, -2))
        persistence = (context[:, -1:].float() - targets.float()).square().mean((-1, -2))
        sensitivity = (prediction.float() - counterfactual.float()).square().mean((-1, -2))
        valid = torch.ones_like(errors, dtype=torch.bool) if batch.valid is None else batch.valid[:, context_frames - 1:]
        if not torch.isfinite(prediction).all() or not torch.isfinite(counterfactual).all():
            raise FloatingPointError("Nonfinite evaluation prediction")
        per_horizon = []
        for t in range(actions.shape[1]):
            mask = valid[:, t]
            if not mask.any():
                continue
            per_horizon.append({"horizon": t + 1, "count": int(mask.sum()),
                "mse": float(errors[mask, t].mean()), "mae": float(mae[mask, t].mean()),
                "persistence_mse": float(persistence[mask, t].mean()),
                "zero_action_sensitivity_mse": float(sensitivity[mask, t].mean())})
        return {"per_horizon": per_horizon, "seed": seed,
                "action_intervention": "zero actions; paired sampler RNG; no counterfactual ground truth"}


def benchmark(model, batch, *, warmup=1, repeats=3, seed=0, context_frames=1):
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid benchmark iteration counts")
    device = batch.actions.device
    with evaluating(model):
        def encode():
            return batch.latents if batch.latents is not None else model.encode(batch.observations)
        def rollout(z):
            if not 1 <= context_frames < z.shape[1]:
                raise ValueError("Benchmark context must leave future targets")
            return model.rollout(z[:, :context_frames], batch.actions[:, context_frames - 1:],
                history_actions=batch.actions[:, :context_frames - 1],
                conditioning=(batch.conditioning or Conditioning()).window(0, context_frames))
        for _ in range(warmup):
            z = encode()
            rollout(z)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        encode_times, rollout_times = [], []
        for i in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            z = encode()
            synchronize(device)
            encode_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            with sampling_seed(seed + i, device):
                rollout(z)
            synchronize(device)
            rollout_times.append(time.perf_counter() - start)
        return {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__, "batch_size": batch.actions.shape[0],
                "context_frames": context_frames, "horizon": batch.actions.shape[1] - context_frames + 1,
                "cached_features": batch.latents is not None, "encode_seconds": encode_times,
                "rollout_seconds": rollout_times, "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def evaluate_mpc_episode(model, controller, env, goal, *, seed, max_steps, episode_id):
    """Environment adapter contract:

    reset(seed=..., episode_id=...) -> observations [1,1,V,3,H,W], Conditioning.
    step(action [A] in configured model units) -> observations, Conditioning, info.
    info MUST contain actual task success, goal_distance and terminated booleans.
    Goal comes from the fixed evaluation episode, never from planner predictions.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    with evaluating(model):
        controller.reset()
        observation, conditioning = env.reset(seed=seed, episode_id=episode_id)
        device = next(model.parameters()).device
        context = model.encode(observation.to(device))
        conditioning = conditioning.to(device)
        goal_z = model.encode(goal.to(device))
        history = context.new_empty(1, 0, controller.planner.low.numel())
        steps, plan_seconds, plans = 0, [], []
        info = None
        while steps < max_steps:
            synchronize(device)
            start = time.perf_counter()
            actions, plan_info = controller.plan(context, goal_z, history_actions=history,
                                                 conditioning=conditioning, seed=seed + steps)
            synchronize(device)
            plan_seconds.append(time.perf_counter() - start)
            plans.append(plan_info)
            for a in actions.split(1, dim=1):
                observation, new_conditioning, info = env.step(a[0, 0].cpu())
                if not {"success", "goal_distance", "terminated"} <= info.keys():
                    raise ValueError("Environment must report actual success, goal_distance and termination")
                if not isinstance(info["success"], bool) or not isinstance(info["terminated"], bool):
                    raise ValueError("Environment success/termination must be booleans")
                if not torch.isfinite(torch.tensor(float(info["goal_distance"]))):
                    raise FloatingPointError("Nonfinite environment goal distance")
                observed_z = model.encode(observation.to(device))
                window = model.rollout_mechanism.context_frames
                context = torch.cat([context, observed_z], dim=1)[:, -window:]
                history = torch.cat([history, a], dim=1)
                history = history[:, -(context.shape[1] - 1):] if context.shape[1] > 1 else history[:, :0]
                new_conditioning = new_conditioning.to(device)
                if conditioning.states is not None:
                    if new_conditioning.states is None:
                        raise ValueError("Environment dropped required proprio")
                    new_conditioning.states = torch.cat([conditioning.states, new_conditioning.states], 1)[:, -window:]
                conditioning = new_conditioning
                steps += 1
                if info["success"] or info["terminated"] or steps >= max_steps:
                    break
            if info["success"] or info["terminated"]:
                break
        return {"episode_id": episode_id, "seed": seed, "steps": steps, "success": info["success"],
                "environment_goal_distance": float(info["goal_distance"]),
                "latent_goal_distance": float(goal_distance(context[:, -1:], goal_z).mean()),
                "plan_seconds": plan_seconds, "plans": plans}
