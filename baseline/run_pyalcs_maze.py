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


def build_configuration() -> Configuration:
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
        do_ga=False,
        do_pee=False,
        do_action_planning=False,
        do_subsumption=True,
    )


def steps_per_trial(trial_metrics: List[dict]) -> List[int]:
    return [m["steps_in_trial"] for m in trial_metrics]


def run_repeat(maze_id: str, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(maze_id)
    env.action_space.seed(seed)

    agent = ACS2(build_configuration())

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maze", help="gym_maze environment id, e.g. Maze4-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    overall_start = time.perf_counter()
    final_window_means: List[float] = []
    total_times: List[float] = []

    for repeat_index in range(args.repeats):
        seed = args.seed + repeat_index
        result = run_repeat(args.maze, seed)
        report_repeat(args.maze, repeat_index, result)
        final_window_means.append(fmean(result["exploit_phase_steps"][-1]))
        total_times.append(result["explore_time"] + result["exploit_time"])

    wall_clock = time.perf_counter() - overall_start

    print(f"### {args.maze}: {args.repeats} repeat(s) ###")
    print(
        f"final_exploit_window_steps over repeats: "
        f"mean={fmean(final_window_means):.3f} "
        f"std={pstdev(final_window_means):.3f}"
    )
    print(
        f"protocol_time per repeat: mean={fmean(total_times):.3f}s "
        f"total_wall_clock={wall_clock:.3f}s"
    )


if __name__ == "__main__":
    main()
