import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from grid_protocol import stop_phase


def outside_repository(path):
    resolved = Path(path).resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("results must be outside every Git repository")
    return resolved


def complete_rows(path):
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    incomplete = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete:
        lines.pop()
    return [json.loads(line) for line in lines], incomplete


def validate_row(row, config, agent, seed, commit, target):
    expected = {
        "schema": 2, "commit": commit, "source_state": "clean",
        "task": config.get("row_task", config["task"]), "cap": config["cap"], "agent": agent,
        "seed": seed, "nominal_step": target, "starts": None,
        "goal_pool": config["row_pool"], "goal_encoding": config["encoding"],
        "evaluated_policy": "greedy_change_anticipating_population",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{key}: expected {value!r}, got {row.get(key)!r}")
    for key in ("host", "cpu_model"):
        if not row.get(key) or row[key] in ("test", "unknown") or "unavailable" in row[key]:
            raise ValueError(f"missing {key}")
    steps = row["actual_steps"]
    if not target <= steps < target + config["cap"]:
        raise ValueError("invalid episode overshoot")
    if row["preset"]["replay_updates_per_step"] != 3:
        raise ValueError("unexpected replay volume")
    if config["task"].startswith("bitflip") and row["preset"]["number_of_possible_actions"] != int(config["task"][7:]):
        raise ValueError("unexpected BitFlipping width")
    if agent == "acs2":
        expected_updates = (steps, 0)
    else:
        expected_updates = (0, sum(min(i, 3) for i in range(1, min(steps, 2) + 1)) + max(steps - 2, 0) * 3)
    if (row["online_updates"], row["replay_updates"]) != expected_updates:
        raise ValueError("unexpected learning update counts")


def process_sample(pid):
    try:
        cpu = int(Path(f"/proc/{pid}/schedstat").read_text().split()[0]) / 1e9
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        peak = next((int(line.split()[1]) for line in status if line.startswith("VmHWM:")), 0)
        return cpu, peak
    except (FileNotFoundError, ProcessLookupError):
        return None


def interpolate(samples, moment):
    times = [sample[0] for sample in samples]
    index = bisect.bisect_left(times, moment)
    if index == 0:
        return samples[0][1]
    if index == len(samples):
        return samples[-1][1]
    left, right = samples[index - 1], samples[index]
    weight = (moment - left[0]) / (right[0] - left[0])
    return left[1] + weight * (right[1] - left[1])


def interval_cost(samples, observed, train_wall, eval_wall):
    eval_start = observed - eval_wall
    train_start = eval_start - train_wall
    return {
        "train_cpu_seconds_estimate": max(0.0, interpolate(samples, eval_start) - interpolate(samples, train_start)),
        "eval_cpu_seconds_estimate": max(0.0, interpolate(samples, observed) - interpolate(samples, eval_start)),
        "train_wall_seconds": train_wall,
        "eval_wall_seconds": eval_wall,
        "maximum_sample_gap_seconds": max((b[0] - a[0] for a, b in zip(samples, samples[1:])), default=0.0),
    }


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    grid = "configuration" in manifest
    if grid:
        config, agent, seed = manifest["configuration"], manifest["agent"], manifest["seed"]
    else:
        config = manifest["configurations"][args.index // 2]
        agent = manifest["agents"][args.index % 2]
        seed = manifest["pilot_seed"]
    directory = outside_repository(Path(args.output) / f"{config['id']}_{agent}_s{seed}")
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / "rows.jsonl"
    command = [str(Path(args.binary).resolve()), "--task", config["task"], "--cap", str(config["cap"]),
               "--pool", config["pool"], "--encoding", config["encoding"], "--agents", agent,
               "--seeds", str(seed), "--targets", ",".join(map(str, config["targets"])), "--out", str(raw)]
    metadata = {"command": command, "commit": args.commit, "manifest": manifest,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
                "job_id": os.environ.get("SLURM_JOB_ID"), "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"), "started_unix": time.time(),
                "sample_interval_seconds": args.sample_interval,
                "cpu_attribution": "Estimated from schedstat samples and runner wall intervals ending at observed flush; includes polling and serialization alignment error. Process CPU total and peak RSS are wait4 measurements."}
    if metadata["partition"] != "lem-cpu-normal":
        raise ValueError("runs only on lem-cpu-normal")
    (directory / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if grid:
        try:
            cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
            if cpu != manifest['expected_cpu_model'] or (manifest.get('expected_host') and socket.gethostname().split('.')[0] != manifest['expected_host']):
                raise ValueError('hardware differs from the approved run assignment')
            if metadata['binary_sha256'] != manifest['expected_binary_sha256'] or args.commit != manifest['measurement_commit']:
                raise ValueError('measurement binary identity mismatch')
        except BaseException as error:
            stop_phase(manifest['stop_path'], {'run': str(directory), 'error': str(error)})
            raise
    started = time.monotonic()
    samples = [(0.0, 0.0)]
    peak = 0
    rows_seen = 0
    previous_train = previous_eval = 0.0
    pending = b""
    offset = 0
    failure = None
    first_row = None
    with (directory / "stderr.txt").open("wb") as stderr, (directory / "stdout.txt").open("wb") as stdout, \
            (directory / "resources.jsonl").open("w", buffering=1) as resources, \
            (directory / "cpu_samples.csv").open("w", buffering=1) as trace:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        trace.write("elapsed_seconds,cpu_seconds,peak_rss_kib\n")
        try:
            while True:
                now = time.monotonic() - started
                exited = process.poll() is not None
                sample = process_sample(process.pid)
                if sample:
                    cpu, rss = sample
                    samples.append((now, cpu))
                    peak = max(peak, rss)
                    trace.write(f"{now:.9f},{cpu:.9f},{rss}\n")
                elif exited:
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    samples.append((now, usage.ru_utime + usage.ru_stime))
                if raw.exists():
                    with raw.open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read()
                    offset += len(chunk)
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        row = json.loads(line)
                        validate_row(row, config, agent, seed, args.commit, config["targets"][rows_seen])
                        if grid and (row['cpu_model'] != manifest['expected_cpu_model'] or (manifest.get('expected_host') and row['host'].split('.')[0] != manifest['expected_host']) or row['preset'] != manifest['expected_preset']):
                            raise ValueError('row hardware or preset mismatch')
                        if first_row is None and now > manifest['first_row_timeout_seconds']:
                            raise TimeoutError('first row arrived after its deadline')
                        rows_seen += 1
                        if first_row is None:
                            first_row = now
                        train = row["wall_seconds_train"]
                        evaluation = row["wall_seconds_eval"]
                        cost = interval_cost(samples, now, train - previous_train, evaluation - previous_eval)
                        cost.update({"nominal_step": row["nominal_step"], "observed_elapsed_seconds": now,
                                     "row_bytes": len(line) + 1, "peak_rss_kib": peak,
                                     "process_cpu_seconds_at_observation": samples[-1][1]})
                        resources.write(json.dumps(cost) + "\n")
                        print(json.dumps({"event": "row", "config": config["id"], "agent": agent,
                                          "step": row["nominal_step"], "elapsed": now}), flush=True)
                        previous_train, previous_eval = train, evaluation
                        samples = samples[-2:]
                if exited:
                    break
                if first_row is None and now > manifest["first_row_timeout_seconds"]:
                    raise TimeoutError("first row deadline exceeded")
                if now > manifest["process_timeout_seconds"]:
                    raise TimeoutError("process deadline exceeded")
                time.sleep(args.sample_interval)
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = {"exit_code": process.returncode, "failure": failure,
                  "rows_seen": rows_seen, "rows_expected": len(config["targets"]),
                  "incomplete_final_line": bool(pending), "first_row_seconds": first_row,
                  "elapsed_seconds": time.monotonic() - started,
                  "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
                  "peak_rss_kib": max(peak, usage.ru_maxrss), "finished_unix": time.time()}
        (directory / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    if failure or process.returncode or rows_seen != len(config["targets"]) or pending:
        if grid:
            stop_phase(manifest['stop_path'], {'run': str(directory), 'result': result})
        raise SystemExit(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    args = parser.parse_args()
    if args.sample_interval <= 0:
        parser.error("sample interval must be positive")
    run(args)


if __name__ == "__main__":
    main()
