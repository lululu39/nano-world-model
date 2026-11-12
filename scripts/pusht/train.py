"""Bounded visual-control transfer; PushT defaults remain backward compatible."""
import argparse
import json
import math
import os
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
from latent_world.pusht import load_robojepa, training_windows, prediction_loss, transfer_context_frames, transfer_loss_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--robo-repo", default="robo_jepa")
    parser.add_argument("--vjepa-repo", default="vjepa2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-after-steps", type=int,
                        help="Diagnostic-only forced exit 85 for a real process/transfer recovery check")
    parser.add_argument("--run-id", default=os.environ.get("PUSHT_RUN_ID"),
                        help="Persistent job identity, including before the first checkpoint transfer")
    args = parser.parse_args()
    if args.smoke and args.run_id:
        raise ValueError("Diagnostic training cannot attach to a formal run identity")
    if args.checkpoint_after_steps is not None and (not args.smoke or args.checkpoint_after_steps < 1):
        raise ValueError("Forced step checkpoints require diagnostic training and a positive step")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    context_frames = transfer_context_frames(cfg)
    loss_type = transfer_loss_type(cfg)
    tcfg = cfg["training"]
    if args.smoke:
        tcfg.update(steps=8, eval_every=8, validation_windows=4)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(cfg["deterministic_algorithms"])
    torch.set_num_threads(4)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    encoder, predictor, report = load_robojepa(args.weights, args.robo_repo, args.vjepa_repo, "cuda")
    report["rollout_context_frames"] = context_frames
    (output / "pretrained-loading.json").write_text(json.dumps(report, indent=2))
    train = training_windows(args.assets, "train", cfg)
    val = training_windows(args.assets, "val", cfg)
    cfg["action_statistics"] = {"mean":train.action_transform.mean.tolist(), "std":train.action_transform.std.tolist()}
    root = Path(__file__).resolve().parents[2]
    provenance = {"assets":json.loads((Path(args.assets)/"provenance.json").read_text()),
                  "torch":torch.__version__, "gpu":torch.cuda.get_device_name()}
    for name, path in [("source",root),("robojepa",Path(args.robo_repo)),("vjepa",Path(args.vjepa_repo))]:
        metadata = path/"SOURCE_VERSION.json"
        provenance[name] = json.loads(metadata.read_text()) if metadata.exists() else {"path":str(path)}
    provenance_path = output / "provenance.json"
    if (output / "latest.pt").exists():
        if not provenance_path.exists() or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Training resume requires the same GPU, sources, assets and runtime")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    assert not set(train.episodes) & set(val.episodes)
    (output / "split.json").write_text(json.dumps({"train_episodes":train.episodes,
        "validation_episodes":val.episodes, "planning_split":"official val (not used for tuning)"}, indent=2))
    batcher = StatefulBatcher(train, tcfg["batch_size"], cfg["seed"])
    optimizer = torch.optim.AdamW([
        {"params":predictor.predictor.parameters(), "lr":tcfg["lr"]},
        {"params":predictor.action_adapter.parameters(), "lr":tcfg["adapter_lr"]}], weight_decay=tcfg["weight_decay"])
    state = {"step":0, "best":float("inf"), "elapsed":0.,
             "run_id":None if args.smoke else (args.run_id or uuid.uuid4().hex[:8])}
    latest = output / "latest.pt"
    if latest.exists():
        state = load_checkpoint(latest, model=predictor, optimizer=optimizer, batcher=batcher, config=cfg)
    if args.smoke != (state["run_id"] is None):
        raise ValueError("Diagnostic checkpoints cannot initialize formal training")
    if args.run_id and state["run_id"] != args.run_id:
        raise ValueError("Checkpoint run identity disagrees with the submitted job")
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "run-id.json").write_text(json.dumps({"run_id":state["run_id"]}))
    tracker = None
    if not args.smoke:
        import wandb
        resume = "must" if latest.exists() else ("allow" if args.run_id else "never")
        tracker = wandb.init(**cfg["logging"], id=state["run_id"], resume=resume,
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
                loss, parts = prediction_loss(predictor, z, batch.actions, context_frames, loss_type=loss_type)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite latent prediction loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), tcfg["grad_clip"], error_if_nonfinite=True)
            assert all(p.grad is None for p in encoder.parameters())
            optimizer.step(); state["step"] += 1
            row = {"step":state["step"], "train/loss":float(loss.detach()), "grad_norm":float(norm),
                   "elapsed_seconds":previous_elapsed+time.monotonic()-started,
                   "windows_seen":state["step"]*tcfg["batch_size"], **parts}
            if state["step"] % tcfg["eval_every"] == 0 or state["step"] == tcfg["steps"]:
                predictor.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    values = [float(prediction_loss(predictor, z.cuda(), a.cuda(), context_frames,
                                                   loss_type=loss_type)[0]) for z,a in validation]
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
            if args.checkpoint_after_steps == state["step"] and state["step"] < tcfg["steps"]:
                save("latest.pt"); code=85; break
            if time.monotonic()-started >= tcfg["checkpoint_seconds"] and state["step"] < tcfg["steps"]:
                save("latest.pt"); code=85; break
        save("latest.pt")
    finally:
        if tracker is not None:
            tracker.finish()
    (output/"summary.json").write_text(json.dumps({"step":state["step"], "run_id":state["run_id"],
        "elapsed_seconds":previous_elapsed+time.monotonic()-started,
        "best_dev_loss":state["best"] if math.isfinite(state["best"]) else None,
        "diagnostic":args.smoke, "exit_code":code, "gpu":torch.cuda.get_device_name(),
        "loss_type":loss_type,
        "completed":state["step"] == tcfg["steps"], "target_steps":tcfg["steps"],
        "windows_seen":state["step"]*tcfg["batch_size"],
        "stop_reason":"target_reached" if state["step"] == tcfg["steps"] else
                      ("checkpoint" if code == 85 else "time_budget")}, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
