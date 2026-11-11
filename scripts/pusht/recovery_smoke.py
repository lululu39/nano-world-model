"""Compare uninterrupted training with an actual Condor exit-85 transfer/restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def equal_tree(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal_tree(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal_tree(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.output)
    reference, resumed = root/"recovery-reference", root/"recovery-resumed"
    marker = root/"recovery-first-allocation.json"
    command = [sys.executable, str(Path(__file__).with_name("train.py")),
               "--config", args.config, "--assets", args.assets, "--smoke"]
    if not marker.exists():
        if reference.exists() or resumed.exists():
            raise RuntimeError("Unexpected partial diagnostics; inspect instead of blindly restarting")
        subprocess.run(command + ["--output", str(reference)], check=True)
        partial = subprocess.run(command + ["--output", str(resumed), "--checkpoint-after-steps", "4"])
        assert partial.returncode == 85
        summary = json.loads((resumed/"summary.json").read_text())
        assert summary["step"] == 4 and not summary["completed"] and summary["run_id"] is None
        marker.write_text(json.dumps({"pid":os.getpid(), "scratch":str(Path.cwd()),
                                      "gpu":summary["gpu"], "interrupted_step":4}, indent=2))
        return 85
    first = json.loads(marker.read_text())
    subprocess.run(command + ["--output", str(resumed)], check=True)
    expected = torch.load(reference/"latest.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed/"latest.pt", map_location="cpu", weights_only=False)
    # Wall time is intentionally different; every optimization/data/RNG state must match.
    assert expected.keys() == actual.keys()
    for key in expected:
        if key != "elapsed":
            equal_tree(expected[key], actual[key], key)
    def metrics(folder):
        rows = [json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("elapsed_seconds")
        return rows
    equal_tree(metrics(reference), metrics(resumed), "metrics")
    summary = json.loads((resumed/"summary.json").read_text())
    assert summary["completed"] and summary["step"] == 8 and summary["run_id"] is None
    report = {"completed":True, "interrupted_step":4, "final_step":8,
              "full_checkpoint_bitwise_equal":True, "all_update_metrics_equal":True,
              "diagnostic":True, "wandb_run_created":False,
              "first_allocation":first, "restored_scratch":str(Path.cwd()),
              "gpu":summary["gpu"]}
    (root/"recovery-check.json").write_text(json.dumps(report, indent=2)+"\n")
    (root/"recovery.complete").touch()
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
