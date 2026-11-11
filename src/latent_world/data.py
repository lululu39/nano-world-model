"""Episode-preserving datasets shared by every latent dynamics method."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .contracts import Conditioning, SequenceBatch


def from_nanowm_batch(batch, *, include_states=False):
    """Adapt the existing normalized WorldModelDataset without re-splitting data."""
    video = batch["video"]
    return SequenceBatch(
        observations=((video + 1) / 2).unsqueeze(2), actions=batch["action"][:, :-1],
        conditioning=Conditioning(states=batch["state"] if include_states else None),
    )


class EpisodeWindowDataset(Dataset):
    """One local tensor file per episode, indexed by an explicit split manifest.

    Each file contains actions [T-1,A] and either latents [T,N,D] or
    observations [T,V,3,H,W] in [0,1]. Optional states [T,S], embodiment_id
    scalar and view_ids [V] are passed without reinterpretation. Files use
    torch.save dictionaries of tensors, loaded with weights_only=True.
    """

    def __init__(self, manifest, split, frames, *, dataset_id, representation_id, action_id, stride=1):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema_version") != 1 or frames < 2 or stride < 1:
            raise ValueError("Invalid manifest schema, clip length or stride")
        for name, expected in {"dataset_id": dataset_id, "representation_id": representation_id, "action_id": action_id}.items():
            if not expected or self.manifest.get(name) != expected:
                raise ValueError(f"Manifest {name} does not match the comparison protocol")
        seen, paths = set(), set()
        self.windows = []
        self.frames = frames
        for entry in self.manifest["episodes"]:
            episode_id = entry["episode_id"]
            path = (self.path.parent / entry["path"]).resolve()
            if episode_id in seen or path in paths:
                raise ValueError("Duplicate episode ID/file: possible train/evaluation leakage")
            if entry["split"] not in {"train", "val", "test"}:
                raise ValueError("Unknown episode split")
            seen.add(episode_id)
            paths.add(path)
            if entry["split"] == split:
                for start in range(0, entry["frames"] - frames + 1, stride):
                    self.windows.append((path, start, entry["frames"]))
        if not self.windows:
            raise ValueError(f"No complete {frames}-frame windows in {split}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        path, start, length = self.windows[index]
        episode = torch.load(path, map_location="cpu", weights_only=True)
        key = "latents" if "latents" in episode else "observations"
        if ("latents" in episode) == ("observations" in episode):
            raise ValueError("Episode must contain exactly one representation source")
        if episode[key].shape[0] != length or episode["actions"].shape[0] != length - 1:
            raise ValueError("Episode tensor lengths disagree with the manifest")
        end = start + self.frames
        result = {key: episode[key][start:end], "actions": episode["actions"][start:end - 1]}
        if "states" in episode:
            if episode["states"].shape[0] != length:
                raise ValueError("States must align with episode observations")
            result["states"] = episode["states"][start:end]
        for name in ("embodiment_id", "view_ids"):
            if name in episode:
                result[name] = episode[name]
        return result


def collate_sequences(items):
    if any(item.keys() != items[0].keys() for item in items):
        raise ValueError("A batch must use consistent representation and conditioning fields")
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    conditioning = Conditioning(states=batch.pop("states", None),
                                embodiment_ids=batch.pop("embodiment_id", None), view_ids=batch.pop("view_ids", None))
    return SequenceBatch(**batch, conditioning=conditioning)


class StatefulBatcher:
    """Deterministic shuffled traversal with resumable permutation and position."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 1 or len(dataset) < 1:
            raise ValueError("Dataset and batch size must be nonempty")
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.offset, self.epoch = 0, 0

    def next(self):
        if self.offset == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.offset = 0
            self.epoch += 1
        indices = self.order[self.offset:self.offset + self.batch_size].tolist()
        self.offset += len(indices)
        return collate_sequences([self.dataset[i] for i in indices])

    def state_dict(self):
        return {"order": self.order, "offset": self.offset, "epoch": self.epoch,
                "generator": self.generator.get_state(), "batch_size": self.batch_size}

    def load_state_dict(self, state):
        if len(state["order"]) != len(self.dataset) or state["batch_size"] != self.batch_size:
            raise ValueError("Cannot resume with a different dataset length or batch size")
        self.order, self.offset, self.epoch = state["order"], state["offset"], state["epoch"]
        self.generator.set_state(state["generator"])
