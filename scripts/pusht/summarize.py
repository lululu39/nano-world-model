"""Build the initial three-metric table only from complete, compatible evaluations."""
import argparse
import csv
import json
from pathlib import Path


def read(folder, name):
    return json.loads((Path(folder) / name).read_text())


def compare(nanowm, robojepa):
    folders = [nanowm, robojepa]
    summaries = [read(p, "summary.json") for p in folders]
    protocols = [read(p, "protocol.json") for p in folders]
    environments = [read(p, "environment.json") for p in folders]
    if any(not s["completed"] or s["diagnostic_subset"] for s in summaries):
        raise ValueError("A smoke subset or incomplete evaluation cannot populate the benchmark table")
    if [s["model"] for s in summaries] != ["nanowm", "robojepa"]:
        raise ValueError("Expected one NanoWM and one fine-tuned RoboJEPA evaluation")
    if protocols[0] != protocols[1] or protocols[0]["name"] != "shared_benchmark":
        raise ValueError("The two methods must use the exact shared benchmark config")
    if read(nanowm, "targets.json") != read(robojepa, "targets.json"):
        raise ValueError("The two methods used different goals")
    for field in ("gpu", "torch", "dependencies", "assets"):
        if environments[0][field] != environments[1][field]:
            raise ValueError(f"The methods differ in {field}; their latencies must not share one table")
    rows = []
    for summary in summaries:
        rows.append({"method":summary["model"], "success_rate":summary["success_rate"],
            "success_count":summary["success_count"], "episodes":summary["episodes"],
            "goal_distance":summary["goal_distance_mean"],
            "planning_median_seconds":summary["planning_latency_seconds"]["median"],
            "planning_p95_seconds":summary["planning_latency_seconds"]["p95"]})
    return {"rows":rows, "gpu":environments[0]["gpu"],
            "goal_distance_definition":summaries[0]["goal_distance_definition"],
            "comparison":"Pretrained system transfer; different representations and pretraining data"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanowm", required=True)
    parser.add_argument("--robojepa", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.nanowm, args.robojepa)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader(); writer.writerows(result["rows"])
    lines = ["| Method | Success | Goal distance | Planning median / P95 (s) |",
             "|---|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f'| {row["method"]} | {row["success_count"]}/{row["episodes"]} ({row["success_rate"]:.1%}) | '
                     f'{row["goal_distance"]:.3f} | {row["planning_median_seconds"]:.3f} / {row["planning_p95_seconds"]:.3f} |')
    lines.extend(["", f'GPU: {result["gpu"]}.', "", result["goal_distance_definition"], "", result["comparison"]])
    (output/"comparison.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
