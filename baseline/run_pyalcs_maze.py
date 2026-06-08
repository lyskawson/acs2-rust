"""Run the unmodified pyalcs ACS2 agent on a gym_maze environment.

Reproduces the pinned benchmark protocol from PROJECT_CONTEXT.md section 5:
explore for 500 trials (epsilon = 0.8) followed by three exploit phases of 200
trials each. Exploitation always uses BestAction and ignores epsilon, so the
three exploit phases are independent pure-RL evaluations of the frozen
population. The episode length cap is the per-maze TimeLimit registered by
gym_maze and is applied automatically by gym.make.
"""

import argparse
import random
import time
from statistics import fmean, pstdev
from typing import Dict, List

import numpy as np
import gym
import gym_maze  # noqa: F401  (import triggers gym_maze environment registration)

from lcs.agents.acs2 import ACS2, Configuration

EXPLORE_TRIALS = 500
EXPLOIT_TRIALS = 200
EXPLOIT_PHASES = 3
EXPLORE_EPSILON = 0.8


def build_configuration(do_ga: bool = False) -> Configuration:
    return Configuration(
        classifier_length=8,
        number_of_possible_actions=8,
        metrics_trial_frequency=1,
        user_metrics_collector_fcn=None,
        epsilon=EXPLORE_EPSILON,
        beta=0.05,
        gamma=0.95,
        theta_i=0.1,
        theta_r=0.9,
        theta_exp=20,
        theta_as=20,
        u_max=100000,
        mu=0.3,
        chi=0.8,
        do_ga=do_ga,
        do_pee=False,
        do_action_planning=False,
        do_subsumption=True,
    )


def steps_per_trial(trial_metrics: List[dict]) -> List[int]:
    return [m["steps_in_trial"] for m in trial_metrics]


def run_repeat(maze_id: str, seed: int, do_ga: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(maze_id)
    env.action_space.seed(seed)

    agent = ACS2(build_configuration(do_ga))

    explore_start = time.perf_counter()
    explore_metrics = agent.explore(env, EXPLORE_TRIALS)
    explore_time = time.perf_counter() - explore_start

    exploit_phase_steps: List[List[int]] = []
    exploit_start = time.perf_counter()
    for _ in range(EXPLOIT_PHASES):
        phase_metrics = agent.exploit(env, EXPLOIT_TRIALS)
        exploit_phase_steps.append(steps_per_trial(phase_metrics))
    exploit_time = time.perf_counter() - exploit_start

    population = agent.get_population()
    numerosity = sum(cl.num for cl in population)
    reliable = sum(1 for cl in population if cl.is_reliable())

    return {
        "seed": seed,
        "explore_steps": steps_per_trial(explore_metrics),
        "exploit_phase_steps": exploit_phase_steps,
        "explore_time": explore_time,
        "exploit_time": exploit_time,
        "macro_population": len(population),
        "numerosity": numerosity,
        "reliable": reliable,
    }


def format_steps(steps: List[int]) -> str:
    return " ".join(str(s) for s in steps)


def report_repeat(maze_id: str, repeat_index: int, result: Dict) -> None:
    print(f"=== {maze_id} | repeat {repeat_index} | seed {result['seed']} ===")

    print(f"explore per-trial steps ({EXPLORE_TRIALS}):")
    print(format_steps(result["explore_steps"]))
    for phase_index, phase_steps in enumerate(result["exploit_phase_steps"]):
        print(f"exploit phase {phase_index + 1} per-trial steps ({EXPLOIT_TRIALS}):")
        print(format_steps(phase_steps))

    final_window = result["exploit_phase_steps"][-1]
    final_mean = fmean(final_window)
    final_std = pstdev(final_window)
    total_time = result["explore_time"] + result["exploit_time"]

    print(
        f"summary: explore_time={result['explore_time']:.3f}s "
        f"exploit_time={result['exploit_time']:.3f}s "
        f"total_time={total_time:.3f}s"
    )
    print(
        f"summary: final_exploit_window_steps mean={final_mean:.3f} "
        f"std={final_std:.3f} (min={min(final_window)} max={max(final_window)})"
    )
    print(
        f"summary: macro_population={result['macro_population']} "
        f"numerosity={result['numerosity']} reliable={result['reliable']}"
    )
    print()


CSV_HEADER = (
    "maze,n_exp,exploit_steps_mean,exploit_steps_std,macro_pop_mean,numerosity_mean,"
    "reliable_mean,explore_time_total_s,exploit_time_total_s,total_time_s"
)


def aggregate_maze(maze_id: str, seed: int, repeats: int, do_ga: bool, verbose: bool) -> Dict:
    final_window_means: List[float] = []
    macro_values: List[int] = []
    numerosity_values: List[int] = []
    reliable_values: List[int] = []
    explore_total = 0.0
    exploit_total = 0.0

    for repeat_index in range(repeats):
        result = run_repeat(maze_id, seed + repeat_index, do_ga)
        if verbose:
            report_repeat(maze_id, repeat_index, result)
        final_window_means.append(fmean(result["exploit_phase_steps"][-1]))
        macro_values.append(result["macro_population"])
        numerosity_values.append(result["numerosity"])
        reliable_values.append(result["reliable"])
        explore_total += result["explore_time"]
        exploit_total += result["exploit_time"]

    return {
        "maze": maze_id,
        "n_exp": repeats,
        "exploit_steps_mean": fmean(final_window_means),
        "exploit_steps_std": pstdev(final_window_means),
        "macro_pop_mean": fmean(macro_values),
        "numerosity_mean": fmean(numerosity_values),
        "reliable_mean": fmean(reliable_values),
        "explore_time_total_s": explore_total,
        "exploit_time_total_s": exploit_total,
        "total_time_s": explore_total + exploit_total,
    }


def csv_row(summary: Dict) -> str:
    return (
        f"{summary['maze']},{summary['n_exp']},{summary['exploit_steps_mean']:.4f},"
        f"{summary['exploit_steps_std']:.4f},{summary['macro_pop_mean']:.2f},"
        f"{summary['numerosity_mean']:.2f},{summary['reliable_mean']:.2f},"
        f"{summary['explore_time_total_s']:.4f},{summary['exploit_time_total_s']:.4f},"
        f"{summary['total_time_s']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mazes", nargs="+", help="gym_maze environment ids, e.g. Maze4-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--do-ga", action="store_true")
    parser.add_argument("--csv", default=None, help="write aggregated per-maze CSV to this path")
    args = parser.parse_args()

    verbose = args.csv is None
    rows = [CSV_HEADER]
    print(f"pyalcs baseline: n_exp={args.repeats} seed={args.seed} do_ga={args.do_ga}")

    for maze_id in args.mazes:
        summary = aggregate_maze(maze_id, args.seed, args.repeats, args.do_ga, verbose)
        rows.append(csv_row(summary))
        print(
            f"{maze_id:<12} exploit_steps={summary['exploit_steps_mean']:.3f}"
            f"±{summary['exploit_steps_std']:.3f} macro={summary['macro_pop_mean']:.1f} "
            f"reliable={summary['reliable_mean']:.1f} total_time={summary['total_time_s']:.3f}s"
        )

    if args.csv is not None:
        with open(args.csv, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
