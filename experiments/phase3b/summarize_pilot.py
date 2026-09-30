import argparse
import json
from pathlib import Path

from collect import complete_rows, validate_row


def inspect(directory):
    launch = json.loads((directory / "launch.json").read_text())
    index = int(launch["array_task_id"])
    manifest = launch["manifest"]
    config = manifest["configurations"][index // 2]
    agent = manifest["agents"][index % 2]
    path = directory / "rows.jsonl"
    rows, incomplete = complete_rows(path) if path.exists() else ([], False)
    if len(rows) > len(config["targets"]):
        raise ValueError(f"too many rows in {path}")
    for row, point in zip(rows, config["targets"]):
        validate_row(row, config, agent, manifest["pilot_seed"], launch["commit"], point)
    resource_path = directory / "resources.jsonl"
    resources, resource_tail = complete_rows(resource_path) if resource_path.exists() else ([], False)
    for entry, row in zip(resources, rows):
        if entry["nominal_step"] != row["nominal_step"]:
            raise ValueError(f"resource alignment mismatch in {directory}")
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    train_cpu = sum(entry["train_cpu_seconds_estimate"] for entry in resources)
    return {"directory": str(directory), "configuration": config["id"], "agent": agent,
            "commit": launch["commit"], "job_id": launch["job_id"], "rows": len(rows),
            "expected_rows": len(config["targets"]), "incomplete_final_line": incomplete,
            "incomplete_resource_line": resource_tail,
            "hosts": sorted({row["host"] for row in rows}),
            "cpu_models": sorted({row["cpu_model"] for row in rows}),
            "completion": completion, "last_step": rows[-1]["nominal_step"] if rows else None,
            "first_row_seconds": resources[0]["observed_elapsed_seconds"] if resources else None,
            "train_cpu_seconds_estimate": train_cpu,
            "train_steps_per_cpu_second_estimate": rows[len(resources) - 1]["actual_steps"] / train_cpu if train_cpu > 0 and len(resources) <= len(rows) else None,
            "evaluation_costs": resources,
            "population_at_points": [[row["nominal_step"], row["population_classifiers"], row["population_numerosity"]] for row in rows],
            "peak_rss_kib": max([entry["peak_rss_kib"] for entry in resources] + [completion["peak_rss_kib"] if completion else 0]),
            "floor": rows[0]["random_floor"] if rows else None,
            "ceiling": rows[0]["reachable_within_cap"] if rows else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    runs = [inspect(path.parent) for path in sorted((args.batch / "runs").glob("*/launch.json"))]
    print(json.dumps({"runs": runs, "run_count": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
