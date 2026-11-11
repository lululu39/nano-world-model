"""Bounded PushT transfer: frozen encoder, pretrained predictor, minimal action adapter."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from omegaconf import OmegaConf
import torch
from latent_world.checkpoint import save_checkpoint, load_checkpoint
from latent_world.data import StatefulBatcher, collate_sequences
from latent_world.pusht import load_robojepa, PushTWindows, prediction_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--robo-repo", default="robo_jepa")
    parser.add_argument("--vjepa-repo", default="vjepa2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    tcfg = cfg["training"]
    if args.smoke:
        tcfg.update(steps=8, eval_every=8, validation_windows=4)
    torch.set_num_threads(4)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    encoder, predictor, report = load_robojepa(args.weights, args.robo_repo, args.vjepa_repo, "cuda")
    (output / "pretrained-loading.json").write_text(json.dumps(report, indent=2))
    train = PushTWindows(args.assets, "train", cfg["frames"], cfg["seed"])
    val = PushTWindows(args.assets, "val", cfg["frames"], cfg["seed"])
    assert not set(train.episodes) & set(val.episodes)
    (output / "split.json").write_text(json.dumps({"train_episodes":train.episodes,
        "validation_episodes":val.episodes, "planning_split":"official val (not used for tuning)"}, indent=2))
    batcher = StatefulBatcher(train, tcfg["batch_size"], cfg["seed"])
    optimizer = torch.optim.AdamW([
        {"params":predictor.predictor.parameters(), "lr":tcfg["lr"]},
        {"params":predictor.action_adapter.parameters(), "lr":tcfg["adapter_lr"]}], weight_decay=tcfg["weight_decay"])
    state = {"step":0, "best":float("inf"), "elapsed":0., "run_id":None if args.smoke else uuid.uuid4().hex[:8]}
    latest = output / "latest.pt"
    if latest.exists():
        state = load_checkpoint(latest, model=predictor, optimizer=optimizer, batcher=batcher, config=cfg)
    if args.smoke != (state["run_id"] is None):
        raise ValueError("Diagnostic checkpoints cannot initialize formal training")
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "run-id.json").write_text(json.dumps({"run_id":state["run_id"]}))
    tracker = None
    if not args.smoke:
        import wandb
        tracker = wandb.init(**cfg["logging"], id=state["run_id"], resume="must" if latest.exists() else "never",
                             config=cfg, dir=str(output), mode="online")
        tracker.summary["gpu"] = torch.cuda.get_device_name()
    # Fixed, encoded held-out windows avoid repeated frozen-encoder work during validation.
    val_indices = np.random.RandomState(cfg["seed"]+1).choice(len(val), tcfg["validation_windows"], replace=False)
    validation = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(val_indices), tcfg["batch_size"]):
            batch = collate_sequences([val[int(i)] for i in val_indices[start:start+tcfg["batch_size"]]]).to("cuda")
            validation.append((encoder(batch.observations).cpu(), batch.actions.cpu()))
    started = time.monotonic()
    previous_elapsed = state["elapsed"]
    def save(name):
        save_checkpoint(output/name, model=predictor, optimizer=optimizer, batcher=batcher,
            step=state["step"], run_id=state["run_id"], config=cfg, best=state["best"],
            elapsed=previous_elapsed+time.monotonic()-started)
    code = 0
    try:
        print("FORMAL_TRAIN_START" if not args.smoke else "DIAGNOSTIC_START", flush=True)
        while state["step"] < tcfg["steps"]:
            predictor.train()
            batch = batcher.next().to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = encoder(batch.observations)
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1., (state["step"]+1)/tcfg["warmup_steps"])
            for group, base in zip(optimizer.param_groups, [tcfg["lr"], tcfg["adapter_lr"]]):
                group["lr"] = base * warmup
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = prediction_loss(predictor, z, batch.actions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite latent prediction loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), tcfg["grad_clip"], error_if_nonfinite=True)
            assert all(p.grad is None for p in encoder.parameters())
            optimizer.step(); state["step"] += 1
            row = {"step":state["step"], "train/loss":float(loss.detach()), "grad_norm":float(norm),
                   "elapsed_seconds":previous_elapsed+time.monotonic()-started, **parts}
            if state["step"] % tcfg["eval_every"] == 0 or state["step"] == tcfg["steps"]:
                predictor.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    values = [float(prediction_loss(predictor, z.cuda(), a.cuda())[0]) for z,a in validation]
                row["dev/loss"] = float(np.mean(values))
                if not np.isfinite(row["dev/loss"]):
                    raise FloatingPointError("Nonfinite validation loss")
                if row["dev/loss"] < state["best"]:
                    state["best"] = row["dev/loss"]; save("best.pt")
                save("latest.pt")
            with (output/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            if tracker is not None:
                tracker.log(row, step=state["step"])
            if state["step"] % 25 == 0 or args.smoke:
                print(json.dumps(row), flush=True)
            elapsed = previous_elapsed + time.monotonic()-started
            if elapsed >= tcfg["max_training_seconds"]:
                save("latest.pt"); code=0; break
            if time.monotonic()-started >= tcfg["checkpoint_seconds"] and state["step"] < tcfg["steps"]:
                save("latest.pt"); code=85; break
        save("latest.pt")
    finally:
        if tracker is not None:
            tracker.finish()
    (output/"summary.json").write_text(json.dumps({"step":state["step"], "run_id":state["run_id"],
        "elapsed_seconds":previous_elapsed+time.monotonic()-started, "best_dev_loss":state["best"],
        "diagnostic":args.smoke, "exit_code":code, "gpu":torch.cuda.get_device_name()}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
