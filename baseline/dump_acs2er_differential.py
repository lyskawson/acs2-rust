"""Differential dumper for the ACS2ER (experience replay) agent.

Runs the unmodified pyalcs ``ACS2ER`` explore loop on a deterministic
environment and records every source of randomness so the Rust port can be
replayed against it step for step.

With ``do_ga=False``, ``do_pee=False`` and the default ``u_max`` the pyalcs
learning path draws from ``random`` in exactly two places: the replay index
draw in ``ACS2ER._run_trial_explore`` and the ``get_differences`` position pick
in ``PMark``. Both are wrapped here and appended to a single chronological
event log, so the Rust side can serve them back through its injected
``RandomSource`` in the same order. Action selection is scripted on both sides.

The pyalcs mid-iteration deletion bug documented in ``docs/ARCHITECTURE.md``
would make an end-to-end population comparison invalid. The only trigger for it
is the ALP deletion of an inadequate classifier, so the gate configuration sets
``theta_i = 0`` to remove that path (``q < 0`` is unreachable); deletions are
counted anyway and the emitted fixture records the count, so the Rust gate can
refuse to compare when it is non-zero. ALP deletion itself is already covered by
the P8 differential.

Never modifies pyalcs: all instrumentation is in-process monkeypatching.
"""

import json
import random
import sys
from pathlib import Path

from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2er import ACS2ER, Configuration

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
from fixtures_common import cl_to_dict, sort_population  # noqa: E402

PERCEPTION_LENGTH = 4
NUMBER_OF_ACTIONS = 2
GOAL_STATE = 5
MAX_EPISODE_STEPS = 15
TRIALS = 60
ER_BUFFER_SIZE = 30
ER_MIN_SAMPLES = 10
ER_SAMPLES_NUMBER = 3
GOAL_REWARD = 1000
ACTION_SCRIPT = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
THETA_I = 0.0


class ActionSpaceMock:
    def sample(self):
        return 0


class CorridorEnv:
    """Deterministic 4-bit counter world. No randomness at all."""

    def __init__(self):
        self.state = 0
        self.elapsed = 0
        self.action_space = ActionSpaceMock()

    @staticmethod
    def encode(state):
        return [c for c in format(state, "04b")]

    def reset(self):
        self.state = 0
        self.elapsed = 0
        return self.encode(self.state)

    def step(self, action):
        self.elapsed += 1
        if action == 1:
            self.state = min(self.state + 1, 15)
        else:
            self.state = max(self.state - 1, 0)

        terminated = self.state == GOAL_STATE
        truncated = self.elapsed >= MAX_EPISODE_STEPS
        reward = GOAL_REWARD if terminated else 0
        return self.encode(self.state), reward, terminated or truncated, None


class ScriptedActionSelector:
    def __init__(self, script, recorded):
        self.script = script
        self.recorded = recorded
        self.cursor = 0

    def __call__(self, match_set):
        action = self.script[self.cursor % len(self.script)]
        self.cursor += 1
        self.recorded.append(action)
        return action


def main():
    events = []
    actions = []
    deletions = {"count": 0}

    real_sample = random.sample
    real_choice = random.choice
    real_safe_remove = ClassifiersList.safe_remove

    def recording_sample(population, k):
        pool = list(population)
        drawn = real_sample(pool, k)
        events.append({"kind": "sample", "bound": len(pool), "values": list(drawn)})
        return drawn

    def recording_choice(sequence):
        pool = list(sequence)
        picked = real_choice(pool)
        events.append({"kind": "choice", "bound": len(pool), "values": [pool.index(picked)]})
        return picked

    def counting_safe_remove(self, item):
        deletions["count"] += 1
        return real_safe_remove(self, item)

    random.sample = recording_sample
    random.choice = recording_choice
    ClassifiersList.safe_remove = counting_safe_remove

    try:
        cfg = Configuration(
            classifier_length=PERCEPTION_LENGTH,
            number_of_possible_actions=NUMBER_OF_ACTIONS,
            do_ga=False,
            do_pee=False,
            do_subsumption=True,
            theta_i=THETA_I,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_MIN_SAMPLES,
            er_samples_number=ER_SAMPLES_NUMBER,
        )
        cfg.action_selector = ScriptedActionSelector(ACTION_SCRIPT, actions)

        agent = ACS2ER(cfg)
        env = CorridorEnv()

        random.seed(20260828)

        time = 0
        trial_steps = []
        for trial in range(TRIALS):
            steps, _reward = agent._run_trial_explore(env, time, trial)
            trial_steps.append(steps)
            time += steps
    finally:
        random.sample = real_sample
        random.choice = real_choice
        ClassifiersList.safe_remove = real_safe_remove

    population = sort_population([cl_to_dict(cl) for cl in agent.get_population()])

    fixture = {
        "config": {
            "classifier_length": PERCEPTION_LENGTH,
            "number_of_possible_actions": NUMBER_OF_ACTIONS,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "theta_i": cfg.theta_i,
            "theta_r": cfg.theta_r,
            "theta_exp": cfg.theta_exp,
            "theta_as": cfg.theta_as,
            "u_max": cfg.u_max,
            "initial_q": cfg.initial_q,
            "do_ga": cfg.do_ga,
            "do_subsumption": cfg.do_subsumption,
            "er_buffer_size": cfg.er_buffer_size,
            "er_min_samples": cfg.er_min_samples,
            "er_samples_number": cfg.er_samples_number,
        },
        "environment": {
            "goal_state": GOAL_STATE,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "goal_reward": GOAL_REWARD,
        },
        "trials": TRIALS,
        "trial_steps": trial_steps,
        "total_steps": time,
        "actions": actions,
        "rng_events": events,
        "alp_deletions": deletions["count"],
        "replay_memory_size": len(agent.replay_memory),
        "population_after": population,
    }

    out_path = FIXTURES_DIR / "acs2er_differential.json"
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")

    sample_events = sum(1 for e in events if e["kind"] == "sample")
    choice_events = sum(1 for e in events if e["kind"] == "choice")
    print(f"wrote {out_path}")
    print(f"  trials={TRIALS} total_steps={time} population={len(population)}")
    print(f"  rng events: sample={sample_events} choice={choice_events}")
    print(f"  alp deletions (pyalcs skip hazard): {deletions['count']}")


if __name__ == "__main__":
    main()
