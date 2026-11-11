"""Full local training state; copy latest/best to persistent storage in the worker."""

import os
from pathlib import Path
import random

import numpy as np
import torch


def save_checkpoint(path, *, model, optimizer, batcher, step, run_id, config, best, elapsed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "batcher": batcher.state_dict(), "step": step, "run_id": run_id, "config": config,
             "best": best, "elapsed": elapsed, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, *, model, optimizer=None, batcher=None, config=None, restore_rng=True):
    # This is our own trusted training artifact (includes Python/NumPy RNG state),
    # unlike external pretrained model files which use weights_only=True.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or (config is not None and state["config"] != config):
        raise ValueError("Resume configuration differs; start a new experiment instead")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if batcher is not None:
        batcher.load_state_dict(state["batcher"])
    if restore_rng:
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("Exact CUDA RNG resume requires CUDA")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
