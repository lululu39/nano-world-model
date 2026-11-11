"""Reuse a complete NanoWM evaluation only when the numerical protocol is identical."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from omegaconf import OmegaConf
from latent_world.factory import load_config


def validate(source, destination, summary):
    metadata = {"name", "assumptions"}
    numerical = lambda cfg: {k:v for k,v in cfg.items() if k not in metadata}
    if numerical(source) != numerical(destination):
        raise ValueError("Numerical protocols differ; a new evaluation is required")
    if destination["name"] != "shared_benchmark":
        raise ValueError("Expected the shared benchmark destination")
    if (summary["model"] != "nanowm" or not summary["completed"] or
            summary["diagnostic_subset"] or summary["episodes"] != destination["n_evals"]):
        raise ValueError("Only a complete NanoWM evaluation can be reused")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source, output = Path(args.source), Path(args.output)
    original = json.loads((source/"protocol.json").read_text())
    destination = OmegaConf.to_container(load_config(args.config), resolve=True)
    summary = json.loads((source/"summary.json").read_text())
    validate(original, destination, summary)
    if output.exists():
        raise FileExistsError("Do not overwrite an existing benchmark result")
    if len(list(source.glob("episode-*.json"))) != summary["episodes"]:
        raise ValueError("Missing per-episode evidence")
    shutil.copytree(source, output)
    summary["protocol"] = destination["name"]
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"protocol.json").write_text(json.dumps(destination, indent=2)+"\n")
    (output/"reuse.json").write_text(json.dumps({
        "source_directory":str(source), "original_protocol":original,
        "reason":"Identical numerical protocol and targets; no additional model execution",
        "changes":["protocol name", "protocol explanatory assumptions"],
        "hardware":"Original recorded GPU; the other method must match it"}, indent=2)+"\n")
    print(f"Reused {summary['episodes']} NanoWM episodes with an identical numerical protocol")


if __name__ == "__main__":
    main()
