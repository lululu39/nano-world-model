"""Stage official PushT assets on a compute node; never scan content hashes."""
import json
import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

DATA_URL = "https://osf.io/download/k2d8w/?view_only=a56a296ce3b24cceaf408383a175ce28"
MODEL_ID = "knightnemo/nanowm-b2-dino-wm-pusht-100k"


def main():
    root = Path("assets")
    root.mkdir(exist_ok=True)
    report = {"dataset_url": DATA_URL, "dataset_bytes": 2785304515}
    archive = Path("pusht_noise.zip")
    extracted = root / "dataset-extracted.json"
    if not extracted.exists():
        if not archive.exists() or archive.stat().st_size != report["dataset_bytes"]:
            with urllib.request.urlopen(DATA_URL, timeout=180) as source, archive.open("wb") as out:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
        assert archive.stat().st_size == report["dataset_bytes"]
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive member")
                if path.name == "tokens.pth":
                    continue  # Unrelated cached features; keep only source RGB/actions/state.
                target = root / "data" / path
                if item.is_dir() or (target.exists() and target.stat().st_size == item.file_size):
                    continue
                source.extract(item, root / "data")
        extracted.write_text(json.dumps({"url":DATA_URL, "bytes":report["dataset_bytes"]}))
        archive.unlink()
    for model_id, target, patterns in [
        (MODEL_ID, "nanowm", ["config.yaml", "model.safetensors"]),
        ("stabilityai/sd-vae-ft-mse", "vae", ["config.json", "diffusion_pytorch_model.safetensors"]),
    ]:
        # The resolved revision is persisted in metadata; no remote HEAD audit.
        snapshot = Path(snapshot_download(model_id, allow_patterns=patterns))
        folder = root / target
        folder.mkdir(exist_ok=True)
        report[target] = {"model_id": model_id, "revision": snapshot.name}
        for name in patterns:
            shutil.copyfile(snapshot / name, folder / name)
    vae_path = root / "vae" / "diffusion_pytorch_model.safetensors"
    full = load_file(str(vae_path))
    encoder = {k: v for k, v in full.items() if k.startswith(("encoder.", "quant_conv."))}
    save_file(encoder, str(root / "vae" / "encoder.safetensors"))
    report["vae"]["retained_keys"] = list(encoder)
    del full, encoder
    vae_path.unlink()  # Decoder tensors are not retained or used.
    dataset = root / "data" / "pusht_noise"
    report["splits"] = {}
    for split in ("train", "val"):
        folder = dataset / split
        with (folder / "seq_lengths.pkl").open("rb") as f:
            lengths = pickle.load(f)
        actions = torch.load(folder / "rel_actions.pth", weights_only=False).float() / 100.0
        valid = torch.cat([actions[i, :int(n)] for i, n in enumerate(lengths)])
        report["splits"][split] = {"episodes": len(lengths), "frames": sum(map(int, lengths))}
        if split == "train":
            stats = {"mean": valid.mean(0), "std": valid.std(0) + 1e-6}
            torch.save(stats, root / "action_stats.pt")
            report["action_stats"] = {k: v.tolist() for k, v in stats.items()}
    (root / "provenance.json").write_text(json.dumps(report, indent=2))
    Path("results/assets.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "vae"}), flush=True)


if __name__ == "__main__":
    main()
