import glob
import json
import os
import sys

import lcs.strategies.reinforcement_learning as rl
from lcs.agents.acs import Condition
from lcs.agents.acs2 import Classifier, ClassifiersList
from lcs.agents.acs2.alp import cover, expected_case, unexpected_case
from lcs.strategies.anticipatory_learning_process import add_classifier
from lcs.strategies.subsumption import does_subsume, is_subsumer

from fixtures_common import (
    cl_to_dict,
    dict_to_cl,
    get_differences_candidates,
    make_cfg,
    mark_to_list,
    perception,
    sample_expected_case,
    sample_get_differences,
    seq_to_list,
    sort_population,
)

TOL = 1e-9
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_KEYS = {"condition", "action", "effect", "mark", "q", "r", "ir",
                   "num", "exp", "talp", "tga", "tav"}


def validate_classifier_keys(node) -> None:
    if isinstance(node, dict):
        if {"condition", "effect", "mark", "q"} <= node.keys():
            assert set(node.keys()) == CLASSIFIER_KEYS, sorted(node.keys())
            for key, value in node.items():
                assert value is not None or key in ("talp", "action"), key
        for value in node.values():
            validate_classifier_keys(value)
    elif isinstance(node, list):
        for value in node:
            validate_classifier_keys(value)


def approx_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOL
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(approx_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(approx_equal(a[k], b[k]) for k in a)
    return a == b


def cfg_for(fixture):
    inp = fixture["input"]
    if "config" in inp:
        c = inp["config"]
        return make_cfg(length=c["classifier_length"], beta=c["beta"], gamma=c["gamma"],
                        theta_exp=c["theta_exp"], theta_r=c["theta_r"], theta_i=c["theta_i"])
    for key in ("p0", "condition"):
        if key in inp:
            return make_cfg(length=len(inp[key]))
    if "classifier" in inp:
        return make_cfg(length=len(inp["classifier"]["condition"]))
    if "cl" in inp:
        return make_cfg(length=len(inp["cl"]["condition"]))
    return make_cfg()


def check(fixture) -> None:
    op = fixture["operation"]
    inp = fixture["input"]
    exp = fixture["expected"]

    if op == "Condition.does_match":
        actual = Condition(list(inp["condition"])).does_match(Condition(list(inp["other"])))
        assert actual == exp["result"]

    elif op == "Condition.subsumes":
        actual = Condition(list(inp["condition"])).subsumes(Condition(list(inp["other"])))
        assert actual == exp["result"]

    elif op == "Effect.anticipates_correctly":
        cfg = cfg_for(fixture)
        cl = Classifier(effect=list(inp["effect"]), cfg=cfg)
        actual = cl.does_anticipate_correctly(perception(inp["p0"]), perception(inp["p1"]))
        assert actual == exp["result"]

    elif op == "Effect.is_specializable":
        cfg = cfg_for(fixture)
        cl = Classifier(effect=list(inp["effect"]), cfg=cfg)
        actual = cl.effect.is_specializable(perception(inp["p0"]), perception(inp["p1"]))
        assert actual == exp["result"]

    elif op == "PMark.get_differences":
        cfg = cfg_for(fixture)

        def mark_factory():
            cl = Classifier(cfg=cfg)
            for idx, values in enumerate(inp["mark"]):
                for v in values:
                    cl.mark[idx].add(str(v))
            return cl.mark

        if exp["mode"] == "exact":
            diff = mark_factory().get_differences(perception(inp["p0"]))
            assert seq_to_list(diff) == exp["diff"]
        else:
            sampled = sample_get_differences(mark_factory, list(inp["p0"]))
            assert sampled["allowed_specialized_indices"] == exp["allowed_specialized_indices"]
            assert sampled["allowed_specificity"] == exp["allowed_specificity"]
            analytic_indices = get_differences_candidates(inp["mark"], inp["p0"])
            assert exp["allowed_specialized_indices"] == analytic_indices
            assert exp["allowed_specificity"] == [1]

    elif op == "alp.expected_case":
        cfg = cfg_for(fixture)

        def cl_factory():
            return dict_to_cl(inp["classifier"], cfg)

        if exp["mode"] == "exact":
            cl = cl_factory()
            child = expected_case(cl, perception(inp["p0"]), inp["time"])
            assert child is None
            assert approx_equal(cl.q, exp["input_after"]["q"])
            assert mark_to_list(cl.mark) == exp["input_after"]["mark"]
        else:
            sampled = sample_expected_case(cl_factory, list(inp["p0"]), inp["time"])
            assert sampled["allowed_specificity"] == exp["child"]["allowed_specificity"]
            candidates = get_differences_candidates(inp["classifier"]["mark"], inp["p0"])
            condition = inp["classifier"]["condition"]
            orig_spec = sum(1 for a in condition if a != "#")
            analytic_spec = sorted({orig_spec + (1 if condition[pos] == "#" else 0)
                                    for pos in candidates})
            assert exp["child"]["allowed_specificity"] == analytic_spec
            assert list(sampled["effect"]) == exp["child"]["effect"]
            assert sampled["action"] == exp["child"]["action"]
            assert approx_equal(sampled["q"], exp["child"]["q"])
            assert sampled["is_marked"] == exp["child"]["is_marked"]
            cl = cl_factory()
            expected_case(cl, perception(inp["p0"]), inp["time"])
            assert approx_equal(cl.q, exp["input_after"]["q"])

    elif op == "alp.unexpected_case":
        cfg = cfg_for(fixture)
        cl = dict_to_cl(inp["classifier"], cfg)
        child = unexpected_case(cl, perception(inp["p0"]), perception(inp["p1"]), inp["time"])
        assert (child is not None) == exp["returns_child"]
        assert approx_equal(cl.q, exp["input_after"]["q"])
        assert mark_to_list(cl.mark) == exp["input_after"]["mark"]
        if child is None:
            assert exp["child"] is None
        else:
            assert approx_equal(cl_to_dict(child), exp["child"])

    elif op == "alp.cover":
        cfg = cfg_for(fixture)
        cl = cover(perception(inp["p0"]), inp["action"], perception(inp["p1"]), inp["time"], cfg)
        assert approx_equal(cl_to_dict(cl), exp["classifier"])

    elif op in ("Classifier.increase_quality", "Classifier.decrease_quality"):
        cfg = make_cfg(beta=inp["beta"])
        cl = Classifier(quality=inp["q"], cfg=cfg)
        getattr(cl, op.split(".")[1])()
        assert approx_equal(cl.q, exp["q"])

    elif op == "Classifier.update_application_average":
        cfg = make_cfg(beta=inp["beta"])
        cl = Classifier(experience=inp["exp"], talp=inp["talp"], tav=inp["tav"], cfg=cfg)
        cl.update_application_average(inp["time"])
        assert approx_equal(cl.tav, exp["tav"])
        assert cl.talp == exp["talp"]

    elif op == "rl.update_classifier":
        cfg = make_cfg(beta=inp["beta"], gamma=inp["gamma"])
        cl = Classifier(reward=inp["r"], immediate_reward=inp["ir"], cfg=cfg)
        rl.update_classifier(cl, inp["reward"], inp["max_fitness"], inp["beta"], inp["gamma"])
        assert approx_equal(cl.r, exp["r"])
        assert approx_equal(cl.ir, exp["ir"])

    elif op == "subsumption.does_subsume":
        cfg = make_cfg(length=len(inp["cl"]["condition"]),
                       theta_r=inp["theta_r"], theta_i=inp["theta_i"])
        cl = dict_to_cl(inp["cl"], cfg)
        other = dict_to_cl(inp["other"], cfg)
        assert does_subsume(cl, other, inp["theta_exp"]) == exp["result"]

    elif op == "subsumption.is_subsumer":
        cfg = make_cfg(length=len(inp["cl"]["condition"]), theta_r=inp["theta_r"])
        cl = dict_to_cl(inp["cl"], cfg)
        assert is_subsumer(cl, inp["theta_exp"]) == exp["result"]

    elif op == "Classifier.__eq__":
        cfg = make_cfg(length=len(inp["cl"]["condition"]))
        assert (dict_to_cl(inp["cl"], cfg) == dict_to_cl(inp["other"], cfg)) == exp["result"]

    elif op == "alp.add_classifier":
        cfg = make_cfg(length=len(inp["child"]["condition"]))
        population = ClassifiersList(*[dict_to_cl(d, cfg) for d in inp["population"]])
        new_list = ClassifiersList()
        child = dict_to_cl(inp["child"], cfg)
        add_classifier(child, population, new_list, inp["theta_exp"])
        assert len(new_list) == exp["new_list_size"]
        assert len(population) == exp["population_size"]
        assert (len(new_list) == 0) == exp["merged_existing"]
        assert approx_equal(sort_population([cl_to_dict(cl) for cl in population]), exp["population"])

    elif op == "ClassifiersList.apply_alp+apply_reinforcement_learning":
        cfg = cfg_for(fixture)
        population = ClassifiersList(*[dict_to_cl(d, cfg) for d in inp["population"]])
        p0p, p1p = perception(inp["p0"]), perception(inp["p1"])
        action = inp["action"]
        action_set = ClassifiersList(*[cl for cl in population
                                       if cl.action == action and cl.does_match(p0p)])
        match_set = ClassifiersList(*[cl for cl in population if cl.does_match(p1p)])
        ClassifiersList.apply_alp(population, match_set, action_set, p0p, action, p1p,
                                  inp["time"], cfg.theta_exp, cfg)
        ClassifiersList.apply_reinforcement_learning(action_set, inp["reward"], inp["max_fitness"],
                                                     cfg.beta, cfg.gamma)
        assert approx_equal(sort_population([cl_to_dict(cl) for cl in population]),
                            exp["population_after"])

    else:
        raise AssertionError(f"unknown operation: {op}")


def main() -> None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    total = 0
    failures = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for fx in data["fixtures"]:
            total += 1
            try:
                validate_classifier_keys(fx["input"])
                validate_classifier_keys(fx["expected"])
                check(fx)
            except AssertionError as err:
                failures.append((fx["id"], fx["operation"], repr(err)))

    print(f"verified {total} fixtures across {len(files)} files")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for fid, op, err in failures:
            print(f"  - {fid} [{op}]: {err}")
        sys.exit(1)
    print("ALL FIXTURES VERIFIED")


if __name__ == "__main__":
    main()
