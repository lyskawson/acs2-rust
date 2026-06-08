"""Differential dumper: emit deterministic-path learning-step cases and one
full-episode metric snapshot from the unmodified pyalcs ACS2 core, for P8.

Part A (the deterministic gate). Random (population, p0, action, p1, time,
reward) cases are generated. For each, the match set, action set, next-state
match set, the RL bootstrap (max q*r over change-anticipating next-state
matchers), and the population after one learning step (apply_alp +
apply_reinforcement_learning) are recorded. Only RNG-free cases are emitted:
the sole randomness in apply_alp is expected_case's get_differences position
pick, which is non-deterministic iff some correctly-anticipating action-set
classifier has >= 2 differing marked positions. This is filtered analytically
and cross-checked empirically by running the step under many seeds.

Part B (the RNG path). pyalcs ACS2.explore is run for one fixed-seed episode on
two mazes; end-of-episode metrics are recorded. Exact agreement with the Rust
loop is NOT expected here: the two implementations draw from different RNG
streams (reset cell, epsilon-greedy selection, ALP position pick), so this is a
reported, explained divergence, not an assertion.
"""

import json
import random
from pathlib import Path

import numpy as np
import gym
import gym_maze  # noqa: F401  (registers the maze environments)

from lcs import Perception
from lcs.agents.acs2 import ACS2, Classifier, ClassifiersList, Configuration

import sys
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
from fixtures_common import (  # noqa: E402
    canonical_key,
    cl_to_dict,
    get_differences_candidates,
    make_cfg,
    sort_population,
)

ALPHABET = ["0", "1"]
GENERATED_CASES = 800
DETERMINISM_SEEDS = 12
CFG = make_cfg(length=8)


def rand_condition():
    return ["#" if random.random() < 0.5 else random.choice(ALPHABET) for _ in range(8)]


def rand_effect():
    return ["#" if random.random() < 0.6 else random.choice(ALPHABET) for _ in range(8)]


def rand_perception():
    return [random.choice(ALPHABET) for _ in range(8)]


def mutate_perception(p0):
    p1 = list(p0)
    for index in range(8):
        if random.random() < 0.25:
            others = [s for s in ALPHABET if s != p0[index]]
            p1[index] = random.choice(others)
    return p1


def cover_effect(p0, p1):
    return [p1[index] if p0[index] != p1[index] else "#" for index in range(8)]


def marks_on_wildcards(condition):
    mark = [[] for _ in range(8)]
    for index in range(8):
        if condition[index] == "#" and random.random() < 0.4:
            count = random.choice([1, 2])
            mark[index] = sorted(random.sample(ALPHABET, count))
    return mark


def base_stats(time):
    return {
        "q": round(random.uniform(0.05, 1.0), 6),
        "r": round(random.uniform(0.0, 5.0), 6),
        "ir": round(random.uniform(0.0, 2.0), 6),
        "num": 1,
        "exp": random.randint(1, 30),
        "talp": random.randint(0, time),
        "tga": random.randint(0, time),
        "tav": round(random.uniform(0.0, 10.0), 6),
    }


def rand_classifier_spec(time):
    condition = rand_condition()
    spec = {"condition": condition, "action": random.randint(0, 7),
            "effect": rand_effect(), "mark": marks_on_wildcards(condition)}
    spec.update(base_stats(time))
    return spec


def matcher_spec(p0, p1, case_action, time):
    condition = [p0[index] if random.random() < 0.6 else "#" for index in range(8)]
    action = case_action if random.random() < 0.7 else random.randint(0, 7)
    if random.random() < 0.5:
        effect = cover_effect(p0, p1)
        quality = round(random.uniform(0.05, 1.0), 6)
    else:
        effect = rand_effect()
        quality = round(random.uniform(0.05, 0.6), 6)
    spec = {"condition": condition, "action": action, "effect": effect,
            "mark": marks_on_wildcards(condition)}
    spec.update(base_stats(time))
    spec["q"] = quality
    return spec


def gen_population_specs(p0, p1, case_action, time):
    specs = []
    for _ in range(random.randint(0, 5)):
        if random.random() < 0.55:
            specs.append(matcher_spec(p0, p1, case_action, time))
        else:
            specs.append(rand_classifier_spec(time))
    return unique_specs(specs)


def spec_to_classifier(spec):
    cl = Classifier(
        condition=list(spec["condition"]),
        action=spec["action"],
        effect=list(spec["effect"]),
        quality=spec["q"],
        reward=spec["r"],
        immediate_reward=spec["ir"],
        numerosity=spec["num"],
        experience=spec["exp"],
        talp=spec["talp"],
        tga=spec["tga"],
        tav=spec["tav"],
        cfg=CFG,
    )
    for index, values in enumerate(spec["mark"]):
        for value in values:
            cl.mark[index].add(str(value))
    return cl


def build_population(specs):
    return ClassifiersList(*[spec_to_classifier(spec) for spec in specs])


def unique_specs(specs):
    seen = set()
    unique = []
    for spec in specs:
        key = (
            "".join(spec["condition"]),
            spec["action"],
            "".join(spec["effect"]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def indices(population, predicate):
    return [index for index, cl in enumerate(population) if predicate(cl)]


def is_deterministic(specs, p0, p1):
    p0p, p1p = Perception(p0), Perception(p1)
    for spec in specs:
        cl = spec_to_classifier(spec)
        if cl.action is None or not cl.does_match(p0p):
            continue
        if not cl.does_anticipate_correctly(p0p, p1p):
            continue
        candidates = get_differences_candidates(spec["mark"], p0)
        if len(candidates) >= 2:
            return False
    return True


def run_learning_step(specs, p0, p1, action, time, reward, bootstrap):
    population = build_population(specs)
    p0p, p1p = Perception(p0), Perception(p1)
    match_set_next = ClassifiersList(*[cl for cl in population if cl.does_match(p1p)])
    action_set = ClassifiersList(
        *[cl for cl in population if cl.action == action and cl.does_match(p0p)]
    )
    ClassifiersList.apply_alp(
        population, match_set_next, action_set, p0p, action, p1p, time, CFG.theta_exp, CFG
    )
    ClassifiersList.apply_reinforcement_learning(
        action_set, reward, bootstrap, CFG.beta, CFG.gamma
    )
    return sort_population([cl_to_dict(cl) for cl in population])


def spec_key(spec):
    return ("".join(spec["condition"]), spec["action"], "".join(spec["effect"]))


def dict_key(cl_dict):
    return ("".join(cl_dict["condition"]), cl_dict["action"], "".join(cl_dict["effect"]))


def is_skip_tainted(specs, action_set_indices, reference):
    survivor_exp = {dict_key(cl): cl["exp"] for cl in reference}
    for index in action_set_indices:
        spec = specs[index]
        key = spec_key(spec)
        if key in survivor_exp and survivor_exp[key] == spec["exp"]:
            return True
    return False


def build_case(case_id, time):
    p0 = rand_perception()
    p1 = mutate_perception(p0)
    action = random.randint(0, 7)
    reward = random.choice([0.0, 0.0, 1000.0])

    specs = gen_population_specs(p0, p1, action, time)

    if not is_deterministic(specs, p0, p1):
        return "rng", None

    population = build_population(specs)
    p0p, p1p = Perception(p0), Perception(p1)
    match_set = indices(population, lambda cl: cl.does_match(p0p))
    action_set = [i for i in match_set if population[i].action == action]
    match_set_next = indices(population, lambda cl: cl.does_match(p1p))
    bootstrap = ClassifiersList(
        *[population[i] for i in match_set_next]
    ).get_maximum_fitness()

    saved_state = random.getstate()
    reference = run_learning_step(specs, p0, p1, action, time, reward, bootstrap)
    seed_invariant = True
    for seed in range(DETERMINISM_SEEDS):
        random.seed(seed)
        if run_learning_step(specs, p0, p1, action, time, reward, bootstrap) != reference:
            seed_invariant = False
            break
    random.setstate(saved_state)
    if not seed_invariant:
        return "rng", None

    if is_skip_tainted(specs, action_set, reference):
        return "pyalcs_bug", None

    action_set_classifiers = [population[i] for i in action_set]
    any_correct = any(cl.does_anticipate_correctly(p0p, p1p) for cl in action_set_classifiers)
    any_deletion = any(
        (not cl.does_anticipate_correctly(p0p, p1p)) and cl.q < CFG.theta_i
        for cl in action_set_classifiers
    )
    coverage = {
        "has_action_set": len(action_set) > 0,
        "expected_case": any_correct,
        "covering": not any_correct,
        "unexpected_case": len(action_set) > 0 and not any_correct,
        "deletion": any_deletion,
    }

    return "kept", {
        "_coverage": coverage,
        "id": case_id,
        "input": {
            "population": [cl_to_dict(spec_to_classifier(spec)) for spec in specs],
            "p0": p0,
            "p1": p1,
            "action": action,
            "time": time,
            "reward": reward,
            "bootstrap": bootstrap,
        },
        "expected": {
            "match_set": match_set,
            "action_set": action_set,
            "match_set_next": match_set_next,
            "bootstrap": bootstrap,
            "population_after": reference,
        },
    }


def build_part_a():
    random.seed(20240607)
    cases = []
    generated = 0
    rng_excluded = 0
    bug_excluded = 0
    coverage = {"has_action_set": 0, "expected_case": 0, "covering": 0,
                "unexpected_case": 0, "deletion": 0}
    while generated < GENERATED_CASES:
        time = random.randint(1, 2000)
        status, case = build_case(f"diff_{generated:04d}", time)
        generated += 1
        if status == "rng":
            rng_excluded += 1
        elif status == "pyalcs_bug":
            bug_excluded += 1
        else:
            for key, value in case.pop("_coverage").items():
                coverage[key] += int(value)
            cases.append(case)
    return cases, generated, rng_excluded, bug_excluded, coverage


def build_part_b():
    snapshots = []
    for maze_id in ["Woods1-v0", "Maze4-v0"]:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        env = gym.make(maze_id)
        env.action_space.seed(seed)
        agent = ACS2(Configuration(
            classifier_length=8, number_of_possible_actions=8,
            metrics_trial_frequency=1, user_metrics_collector_fcn=None,
            epsilon=0.8, beta=0.05, gamma=0.95, theta_i=0.1, theta_r=0.9,
            theta_exp=20, theta_as=20, u_max=100000, mu=0.3, chi=0.8,
            do_ga=False, do_pee=False, do_action_planning=False, do_subsumption=True,
        ))
        metrics = agent.explore(env, 1)
        population = agent.get_population()
        snapshots.append({
            "maze": maze_id,
            "seed": seed,
            "trials": 1,
            "steps": metrics[-1]["steps_in_trial"],
            "macro_population": len(population),
            "numerosity": sum(cl.num for cl in population),
            "reliable": sum(1 for cl in population if cl.is_reliable()),
        })
    return snapshots


def main():
    cases, generated, rng_excluded, bug_excluded, coverage = build_part_a()
    for case in cases:
        keys = [canonical_key(cl) for cl in case["expected"]["population_after"]]
        assert len(set(keys)) == len(keys), case["id"]
    part_a_path = FIXTURES_DIR / "differential_cases.json"
    part_a_path.write_text(json.dumps({
        "generated": generated,
        "deterministic_kept": len(cases),
        "rng_excluded": rng_excluded,
        "pyalcs_iteration_bug_excluded": bug_excluded,
        "coverage": coverage,
        "cases": cases,
    }, indent=2))

    snapshots = build_part_b()
    part_b_path = FIXTURES_DIR / "differential_episode.json"
    part_b_path.write_text(json.dumps({"episodes": snapshots}, indent=2))

    print(f"Part A: {len(cases)} deterministic cases kept, {rng_excluded} RNG-excluded, "
          f"{bug_excluded} pyalcs-iteration-bug-excluded (of {generated} generated) "
          f"-> {part_a_path.name}")
    print(f"Part A coverage: {coverage}")
    for snapshot in snapshots:
        print(f"Part B: {snapshot['maze']} seed={snapshot['seed']} "
              f"steps={snapshot['steps']} macro={snapshot['macro_population']} "
              f"numerosity={snapshot['numerosity']} reliable={snapshot['reliable']}")


if __name__ == "__main__":
    main()
