"""Compare the two completed history settings and the unchanged NanoWM baseline."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from summarize import compare, read


def verify_records(folder):
    summary = read(folder, "summary.json")
    protocol = read(folder, "protocol.json")
    records = [read(folder, f"episode-{i:04d}.json") for i in range(protocol["n_evals"])]
    assert len(records) == summary["episodes"] == 64
    for i, record in enumerate(records):
        assert record["episode"] == i and record["seed"] == protocol["seed"] + i
        assert record["success"] in (0, 1)
        assert 0 < record["n_steps"] <= protocol["max_episode_steps"] * protocol["frame_interval"]
        assert math.isfinite(record["state_dist"]) and record["state_dist"] >= 0
    assert sum(r["success"] for r in records) == summary["success_count"]
    assert summary["success_rate"] == summary["success_count"] / len(records)
    assert math.isclose(statistics.mean(r["state_dist"] for r in records),
                        summary["goal_distance_mean"], rel_tol=1e-12)
    times = [t for r in records for t in r["planning_seconds"]]
    assert all(math.isfinite(t) and t > 0 for t in times)
    assert math.isclose(statistics.median(times), summary["planning_latency_seconds"]["median"], rel_tol=1e-12)
    sorted_times = sorted(times)
    index = .95 * (len(times) - 1)
    lower = math.floor(index); upper = math.ceil(index)
    p95 = sorted_times[lower] + (index-lower)*(sorted_times[upper]-sorted_times[lower])
    assert math.isclose(p95, summary["planning_latency_seconds"]["p95"], rel_tol=1e-12)
    assert len(times) == summary["planning_latency_seconds"]["replans"]
    return {"episodes":len(records), "successes":summary["success_count"], "replans":len(times)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", required=True, help="Original shared-benchmark directory")
    p.add_argument("--current", required=True, help="Collected prefix experiment: training/ and reports/")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    previous, current, output = Path(args.previous), Path(args.current), Path(args.output)
    nano, single, prefix = previous/"nanowm-shared", previous/"robojepa-shared", current/"reports/robojepa-prefix"
    original = compare(nano, single)
    updated = compare(nano, prefix)
    old_config = read(previous/"training", "config.json")
    new_config = read(current/"training", "config.json")
    assert old_config.get("rollout_context_frames", 1) == 1
    assert new_config.pop("rollout_context_frames") == 3
    old_config.pop("rollout_context_frames", None)
    for cfg in (old_config, new_config):
        cfg["logging"].pop("group")
    assert old_config == new_config, "The experiment changed more than prediction history"
    assert read(previous/"training", "split.json") == read(current/"training", "split.json")
    summary = read(current/"training", "summary.json")
    assert summary["step"] == 3000 and not summary["diagnostic"] and summary["exit_code"] == 0
    rows = [json.loads(line) for line in (current/"training/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 3001))
    assert all(math.isfinite(r[k]) for r in rows for k in ("train/loss", "grad_norm", "teacher_mae", "rollout_mae"))
    loading = read(prefix, "weight-loading.json")
    assert loading["rollout_context_frames"] == 3 and loading["finetuned_run_id"] == summary["run_id"]
    assert loading["pretrained_predictor_exactly_equal"] and loading["encoder_frozen"]
    result = dict(updated)
    result["rows"] = [original["rows"][0], dict(original["rows"][1], method="robojepa_single_frame"),
                      dict(updated["rows"][1], method="robojepa_predicted_prefix")]
    result["comparison"] = "Matched-budget RoboJEPA history comparison; NanoWM remains a pretrained-system baseline"
    result["training"] = summary
    output.mkdir(parents=True, exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(result["rows"])
    checks = {name:verify_records(path) for name,path in [("nanowm",nano),("single_frame",single),("prefix",prefix)]}
    checks.update(same_training_recipe=True, same_episode_split=True, finite_updates=3000,
                  same_goals_protocol_gpu_and_dependencies=True)
    (output/"independent-check.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
