import json
import os
import random
from typing import Dict, List

import lcs.strategies.reinforcement_learning as rl
from lcs.agents.acs import Condition
from lcs.agents.acs2 import Classifier, ClassifiersList
from lcs.agents.acs2.alp import cover, expected_case, unexpected_case
from lcs.strategies.anticipatory_learning_process import add_classifier
from lcs.strategies.subsumption import does_subsume, is_subsumer

from fixtures_common import (
    cl_to_dict,
    dict_to_cl,
    make_cfg,
    mark_to_list,
    perception,
    sample_expected_case,
    sample_get_differences,
    seq_to_list,
    sort_population,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fixture(id_, operation, source, description, input_, expected) -> Dict:
    return {
        "id": id_,
        "operation": operation,
        "source": source,
        "description": description,
        "wildcard": "#",
        "input": input_,
        "expected": expected,
    }


def build_condition_matching() -> List[Dict]:
    out = []

    def does_match(id_, cond, other, source, desc):
        result = Condition(list(cond)).does_match(Condition(list(other)))
        return fixture(
            id_, "Condition.does_match", source, desc,
            {"condition": list(cond), "other": list(other)},
            {"mode": "exact", "result": result},
        )

    def subsumes(id_, cond, other, source, desc, expect):
        result = Condition(list(cond)).subsumes(Condition(list(other)))
        assert result is expect, (id_, result, expect)
        return fixture(
            id_, "Condition.subsumes", source, desc,
            {"condition": list(cond), "other": list(other)},
            {"mode": "exact", "result": result},
        )

    out.append(does_match("cond_match_full", "1010", "1010",
                          "generated: deterministic", "identical perception matches"))
    out.append(does_match("cond_match_no", "1010", "1011",
                          "generated: deterministic", "single mismatch fails"))
    out.append(does_match("cond_match_wildcard_in_condition", "#0#0", "1000",
                          "generated: deterministic", "wildcard in condition matches anything"))
    out.append(does_match("cond_match_all_wildcard", "####", "1010",
                          "generated: deterministic", "fully general condition matches"))
    out.append(does_match("cond_match_wildcard_other_side", "1010", "#0#0",
                          "generated: deterministic", "wildcard on the other side also matches"))
    out.append(does_match("cond_match_maze_symbol_9", "####9###", "11111111",
                          "generated: deterministic", "specified reward symbol vs wall fails"))

    out.append(subsumes("cond_subsumes_more_general", "###0####", "1##0####",
                        "test: tests/lcs/strategies/test_subsumption.py::test_should_subsume_another_classifier",
                        "general condition subsumes specific compatible one", True))
    out.append(subsumes("cond_subsumes_conflict_false", "10######", "11######",
                        "generated: deterministic", "conflicting specified attributes do not subsume", False))
    out.append(subsumes("cond_subsumes_equal", "1##0####", "1##0####",
                        "generated: deterministic", "identical conditions subsume", True))

    return out


def build_effect_anticipation() -> List[Dict]:
    out = []

    def anticipates(id_, effect, p0, p1, source, desc):
        cfg = make_cfg(length=len(p0))
        cl = Classifier(effect=list(effect), cfg=cfg)
        result = cl.does_anticipate_correctly(perception(p0), perception(p1))
        return fixture(
            id_, "Effect.anticipates_correctly", source, desc,
            {"effect": list(effect), "p0": list(p0), "p1": list(p1)},
            {"mode": "exact", "result": result},
        )

    def specializable(id_, effect, p0, p1, source, desc):
        cfg = make_cfg(length=len(p0))
        cl = Classifier(effect=list(effect), cfg=cfg)
        result = cl.effect.is_specializable(perception(p0), perception(p1))
        return fixture(
            id_, "Effect.is_specializable", source, desc,
            {"effect": list(effect), "p0": list(p0), "p1": list(p1)},
            {"mode": "exact", "result": result},
        )

    out.append(anticipates("effect_passthrough_correct", "####", "1010", "1010",
                           "generated: deterministic", "wildcard pass-through with no change is correct"))
    out.append(anticipates("effect_passthrough_unexpected_change", "####", "1010", "1011",
                           "generated: deterministic", "wildcard pass-through but value changed is incorrect"))
    out.append(anticipates("effect_specified_change_correct", "#0##", "1110", "1010",
                           "generated: deterministic", "specified change to the observed value is correct"))
    out.append(anticipates("effect_specified_no_change", "#0##", "1010", "1010",
                           "generated: deterministic", "specified change but value stayed is incorrect"))
    out.append(anticipates("effect_specified_wrong_value", "#0##", "1110", "1190",
                           "generated: deterministic", "specified change to a different value is incorrect"))

    out.append(specializable("effect_specializable_all_wildcard", "####", "1010", "1011",
                             "generated: deterministic", "all-wildcard effect is always specializable"))
    out.append(specializable("effect_not_specializable_wrong_target", "#1##", "1010", "1000",
                             "generated: deterministic", "specified attr not matching p1 is not specializable"))
    out.append(specializable("effect_not_specializable_no_change", "#1##", "1110", "1110",
                             "generated: deterministic", "specified attr where p0==p1 is not specializable"))

    return out


def build_mark_differences() -> List[Dict]:
    out = []

    def exact(id_, mark, p0, source, desc):
        cfg = make_cfg(length=len(p0))

        def factory():
            cl = Classifier(cfg=cfg)
            for idx, values in enumerate(mark):
                for v in values:
                    cl.mark[idx].add(str(v))
            return cl.mark

        random.seed(0)
        diff = factory().get_differences(perception(p0))
        return fixture(
            id_, "PMark.get_differences", source, desc,
            {"mark": [[str(v) for v in s] for s in mark], "p0": list(p0)},
            {"mode": "exact", "diff": seq_to_list(diff)},
        )

    def prop(id_, mark, p0, source, desc):
        cfg = make_cfg(length=len(p0))

        def factory():
            cl = Classifier(cfg=cfg)
            for idx, values in enumerate(mark):
                for v in values:
                    cl.mark[idx].add(str(v))
            return cl.mark

        sampled = sample_get_differences(factory, list(p0))
        return fixture(
            id_, "PMark.get_differences", source, desc,
            {"mark": [[str(v) for v in s] for s in mark], "p0": list(p0)},
            {
                "mode": "property",
                "allowed_specialized_indices": sampled["allowed_specialized_indices"],
                "allowed_specificity": sampled["allowed_specificity"],
                "specialized_value_equals_p0": True,
            },
        )

    empty = [[] for _ in range(8)]
    out.append(exact("mark_diff_empty", empty, "01100011",
                     "test: tests/lcs/agents/acs/test_PMark.py::test_should_get_differences_1",
                     "empty mark yields fully general difference"))

    nr2_mark = [["0", "1"], ["0", "1"], [], ["0", "1"], ["0", "1"], [], ["0", "1"], ["0"]]
    out.append(exact("mark_diff_all_positions", nr2_mark, "11111010",
                     "test: tests/lcs/agents/acs/test_PMark.py::test_should_get_differences_4",
                     "nr2 branch specializes every multi-valued position (deterministic)"))

    single = [[], [], ["1"], [], [], [], [], []]
    out.append(exact("mark_diff_single_candidate", single, "00000000",
                     "generated: deterministic (single nr1 candidate)",
                     "exactly one differing position is specialized deterministically"))

    prop2 = [["1"], ["1"], ["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]
    out.append(prop("mark_diff_property_specificity1", prop2, "11011101",
                    "test: tests/lcs/agents/acs/test_PMark.py::test_should_get_differences_2",
                    "nr1 branch specializes one random differing position"))

    prop3 = [["0", "1"], ["1"], ["0", "1"], ["1"], ["0", "1"], ["1"], ["0", "1"], ["1"]]
    out.append(prop("mark_diff_property_mixed", prop3, "01100000",
                    "test: tests/lcs/agents/acs/test_PMark.py::test_should_get_differences_3",
                    "nr1 dominates nr2: one random differing position specialized"))

    prop5 = [[], [], [], ["0"], [], [], ["0"], []]
    out.append(prop("mark_diff_property_two_candidates", prop5, "00211010",
                    "test: tests/lcs/agents/acs/test_PMark.py::test_should_get_differences_5",
                    "two nr1 candidates, one chosen at random"))

    return out


def build_expected_case() -> List[Dict]:
    out = []

    def none_case(id_, cond, q, p0, time, source, desc, q_literal):
        cfg = make_cfg(length=len(p0))
        cl = Classifier(condition=list(cond), quality=q, cfg=cfg)
        child = expected_case(cl, perception(p0), time)
        assert child is None
        assert abs(q_literal - cl.q) < 0.01, (id_, cl.q)
        return fixture(
            id_, "alp.expected_case", source, desc,
            {"classifier": cl_to_dict(Classifier(condition=list(cond), quality=q, cfg=cfg)),
             "p0": list(p0), "time": time},
            {"mode": "exact", "returns_child": False,
             "input_after": {"q": cl.q, "mark": mark_to_list(cl.mark)}},
        )

    def child_case(id_, cond, action, effect, q, mark, p0, time, source, desc):
        cfg = make_cfg(length=len(p0))

        def factory():
            cl = Classifier(condition=list(cond) if cond else None,
                            action=action,
                            effect=list(effect) if effect else None,
                            quality=q, cfg=cfg)
            for idx, values in enumerate(mark):
                for v in values:
                    cl.mark[idx].add(str(v))
            return cl

        sampled = sample_expected_case(factory, list(p0), time)
        assert sampled["effect_invariant"], (id_, sampled["effect"])
        baseline = factory()
        input_dict = cl_to_dict(baseline)
        return fixture(
            id_, "alp.expected_case", source, desc,
            {"classifier": input_dict, "p0": list(p0), "time": time},
            {
                "mode": "property",
                "returns_child": True,
                "input_after": {"q": baseline.q, "mark": mark_to_list(baseline.mark)},
                "child": {
                    "effect": list(sampled["effect"]),
                    "action": sampled["action"],
                    "q": sampled["q"],
                    "is_marked": sampled["is_marked"],
                    "allowed_specificity": sampled["allowed_specificity"],
                },
            },
        )

    out.append(none_case("expected_case_none_1", "#######0", 0.525, "11111010", 47,
                         "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_expected_case_1",
                         "no mark difference: quality increased, no child", 0.54))
    out.append(none_case("expected_case_none_2", "#0######", 0.521, "10101001", 59,
                         "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_expected_case_2",
                         "no mark difference: quality increased, no child", 0.54))

    mark3 = [["0"], ["1"], ["0"], ["1"], ["0"], ["1"], ["1"], ["1"]]
    out.append(child_case("expected_case_child_1", None, 5, None, 0.46, mark3, "00110000", 26,
                          "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_expected_case_3",
                          "one random attribute specialized; child quality floored to 0.5"))

    mark4 = [[], ["0", "2"], ["1"], [], [], ["0", "1"], [], ["1"]]
    out.append(child_case("expected_case_child_2", "1##01#0#", 7, "0##10#1#", 0.47, mark4, "11101101", 703,
                          "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_expected_case_4",
                          "one random attribute specialized; specificity in {4,5}"))

    return out


def build_unexpected_case() -> List[Dict]:
    out = []

    def build(id_, kwargs, mark, p0, p1, time, source, desc, expect_cond=None, expect_effect=None,
              expect_q_literal=None, expect_child_stats=None):
        cfg = make_cfg(length=len(p0))

        def make_cl():
            cl = Classifier(cfg=cfg, **kwargs)
            for idx, values in enumerate(mark):
                for v in values:
                    cl.mark[idx].add(str(v))
            return cl

        input_cl = make_cl()
        input_dict = cl_to_dict(input_cl)

        working = make_cl()
        child = unexpected_case(working, perception(p0), perception(p1), time)

        if expect_q_literal is not None:
            assert abs(expect_q_literal - working.q) < 0.01, (id_, working.q)

        child_dict = None
        if child is not None:
            if expect_cond is not None:
                assert seq_to_list(child.condition) == list(expect_cond), (id_, seq_to_list(child.condition))
            if expect_effect is not None:
                assert seq_to_list(child.effect) == list(expect_effect), (id_, seq_to_list(child.effect))
            if expect_child_stats is not None:
                for field, literal in expect_child_stats.items():
                    assert abs(getattr(child, field) - literal) < 0.1, (id_, field, getattr(child, field))
            child_dict = cl_to_dict(child)

        return fixture(
            id_, "alp.unexpected_case", source, desc,
            {"classifier": input_dict, "p0": list(p0), "p1": list(p1), "time": time},
            {
                "mode": "exact",
                "returns_child": child is not None,
                "input_after": {"q": working.q, "mark": mark_to_list(working.mark)},
                "child": child_dict,
            },
        )

    out.append(build(
        "unexpected_case_1", {"action": 2}, [[] for _ in range(8)],
        "01100000", "10100010", 14,
        "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_unexpected_case_1",
        "general classifier specialized; original quality decreased and marked",
        "01####0#", "10####1#", 0.475))

    mark2 = [["0", "1"], ["0", "1"], ["0", "1"], ["0", "1"], ["1"], ["0", "1"], ["0", "1"], []]
    out.append(build(
        "unexpected_case_2", {"condition": "#######0", "action": 4, "quality": 0.4}, mark2,
        "11101010", "10011101", 94,
        "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_unexpected_case_2",
        "specialized child built from changing positions",
        "#110#010", "#001#101", 0.38))

    mark3 = [["1"], ["1"], ["0"], ["1"], [], ["1"], [], ["1"]]
    out.append(build(
        "unexpected_case_3_none", {"condition": "#####1#0", "effect": "#####0#1", "quality": 0.475}, mark3,
        "11001110", "01110000", 20,
        "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_unexpected_case_3",
        "effect not specializable: quality decreased, no child returned",
        None, None, 0.45125))

    mark5 = [[], [], ["2"], ["1"], ["1"], ["0"], [], ["0"]]
    out.append(build(
        "unexpected_case_5",
        {"condition": "00####1#", "action": 2, "effect": "########", "quality": 0.129,
         "reward": 341.967, "immediate_reward": 130.369, "experience": 201, "tga": 129,
         "talp": 9628, "tav": 25.08}, mark5,
        "00211010", "00001110", 9628,
        "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_unexpected_case_5",
        "child inherits reward statistics from parent",
        "0021#01#", "##00#1##", None,
        {"r": 341.967, "ir": 130.369, "tav": 25.08, "exp": 1, "num": 1}))

    mark6 = [[], ["1"], [], ["1"], ["0", "1"], ["1"], ["0", "1"], []]
    out.append(build(
        "unexpected_case_6",
        {"condition": "0#1####1", "action": 2, "effect": "1#0####0", "quality": 0.38505,
         "reward": 1.20898, "immediate_reward": 0, "experience": 11, "tga": 95,
         "talp": 873, "tav": 71.3967}, mark6,
        "01111101", "11011110", 873,
        "test: tests/lcs/agents/acs2/test_alp.py::test_should_handle_unexpected_case_6",
        "child specializes additional changing positions",
        "0#1###01", "1#0###10", None,
        {"r": 1.20898, "ir": 0.0, "tav": 71.3967, "exp": 1, "num": 1}))

    return out


def build_covering() -> List[Dict]:
    out = []

    def build(id_, p0, p1, action, time, source, desc, expect_cond=None, expect_effect=None):
        cfg = make_cfg(length=len(p0))
        cl = cover(perception(p0), action, perception(p1), time, cfg)
        if expect_cond is not None:
            assert seq_to_list(cl.condition) == list(expect_cond), (id_, seq_to_list(cl.condition))
        if expect_effect is not None:
            assert seq_to_list(cl.effect) == list(expect_effect), (id_, seq_to_list(cl.effect))
        return fixture(
            id_, "alp.cover", source, desc,
            {"p0": list(p0), "p1": list(p1), "action": action, "time": time},
            {"mode": "exact", "classifier": cl_to_dict(cl)},
        )

    out.append(build("cover_1", "01001101", "00011111", 3, 100,
                     "test: tests/lcs/agents/acs2/test_alp.py::test_should_create_new_classifier_using_covering",
                     "covering classifier anticipates the observed change",
                     "#1#0##0#", "#0#1##1#"))
    out.append(build("cover_2", "0000", "0110", 2, 5,
                     "generated: deterministic",
                     "covering specializes only changed positions",
                     "#00#", "#11#"))
    return out


def build_updates() -> List[Dict]:
    out = []

    def quality(id_, op, q, beta, desc):
        cfg = make_cfg(beta=beta)
        cl = Classifier(quality=q, cfg=cfg)
        getattr(cl, op)()
        return fixture(
            id_, f"Classifier.{op}", "generated: deterministic", desc,
            {"q": q, "beta": beta},
            {"mode": "exact", "q": cl.q},
        )

    out.append(quality("increase_quality_1", "increase_quality", 0.5, 0.05, "q toward 1 by beta"))
    out.append(quality("increase_quality_2", "increase_quality", 0.4, 0.05, "q toward 1 by beta"))
    out.append(quality("decrease_quality_1", "decrease_quality", 0.5, 0.05, "q toward 0 by beta"))
    out.append(quality("decrease_quality_2", "decrease_quality", 0.4, 0.05, "q toward 0 by beta"))

    def tav(id_, exp, talp, tav_in, time, beta, desc):
        cfg = make_cfg(beta=beta)
        cl = Classifier(experience=exp, talp=talp, tav=tav_in, cfg=cfg)
        cl.update_application_average(time)
        return fixture(
            id_, "Classifier.update_application_average", "generated: deterministic", desc,
            {"exp": exp, "talp": talp, "tav": tav_in, "time": time, "beta": beta},
            {"mode": "exact", "tav": cl.tav, "talp": cl.talp},
        )

    out.append(tav("tav_running_mean_exp1", 1, 0, 0.0, 10, 0.05, "MAM running mean at exp=1"))
    out.append(tav("tav_running_mean_exp5", 5, 5, 2.0, 30, 0.05, "MAM running mean at exp=5"))
    out.append(tav("tav_running_mean_exp19", 19, 100, 10.0, 160, 0.05, "MAM running mean just below 1/beta boundary"))
    out.append(tav("tav_exponential_exp20", 20, 100, 10.0, 160, 0.05, "MAM exponential exactly at exp=1/beta"))
    out.append(tav("tav_exponential_exp50", 50, 873, 71.3967, 900, 0.05, "MAM exponential above boundary"))

    def rl_update(id_, r, ir, reward, max_fitness, beta, gamma, source, desc, r_lit=None, ir_lit=None):
        cfg = make_cfg(beta=beta, gamma=gamma)
        cl = Classifier(reward=r, immediate_reward=ir, cfg=cfg)
        rl.update_classifier(cl, reward, max_fitness, beta, gamma)
        if r_lit is not None:
            assert abs(cl.r - r_lit) < 0.001, (id_, cl.r)
        if ir_lit is not None:
            assert abs(cl.ir - ir_lit) < 0.001, (id_, cl.ir)
        return fixture(
            id_, "rl.update_classifier", source, desc,
            {"r": r, "ir": ir, "reward": reward, "max_fitness": max_fitness, "beta": beta, "gamma": gamma},
            {"mode": "exact", "r": cl.r, "ir": cl.ir},
        )

    out.append(rl_update("rl_goal_reward", 0.5, 0.0, 1000, 1000, 0.05, 0.95,
                         "test: tests/lcs/strategies/test_reinforcement_learning.py::test_should_update_classifier",
                         "reward and bootstrap propagate into r and ir", 97.975, 50.0))
    out.append(rl_update("rl_episode_end", 0.5, 0.0, 0, 0, 0.05, 0.95,
                         "generated: deterministic", "terminal step with zero reward and zero bootstrap"))
    out.append(rl_update("rl_intermediate_bootstrap", 10.0, 5.0, 0, 2.0, 0.05, 0.95,
                         "generated: deterministic", "zero step reward but nonzero bootstrap"))

    return out


def build_subsumption() -> List[Dict]:
    out = []

    def subsume(id_, c1, expect, source, desc):
        cfg = make_cfg()
        cl = Classifier(condition=c1["c"], action=c1["a"], effect=c1["e"],
                        quality=c1["q"], reward=c1["r"], experience=c1["exp"], cfg=cfg)
        other = Classifier(condition=c1["oc"], action=c1["oa"], effect=c1["oe"],
                           quality=c1["oq"], reward=c1["or"], experience=c1["oexp"], cfg=cfg)
        result = does_subsume(cl, other, cfg.theta_exp)
        assert result is expect, (id_, result)
        return fixture(
            id_, "subsumption.does_subsume", source, desc,
            {"cl": cl_to_dict(cl), "other": cl_to_dict(other),
             "theta_exp": cfg.theta_exp, "theta_r": cfg.theta_r, "theta_i": cfg.theta_i},
            {"mode": "exact", "result": result},
        )

    src = "test: tests/lcs/strategies/test_subsumption.py::test_should_subsume_another_classifier"
    out.append(subsume("does_subsume_true_1",
                       {"c": "###0####", "a": 3, "e": "##1#####", "q": 0.93, "r": 1.35, "exp": 23,
                        "oc": "1##0####", "oa": 3, "oe": "##1#####", "oq": 0.5, "or": 0.35, "oexp": 1},
                       True, src, "general reliable experienced classifier subsumes specific"))
    out.append(subsume("does_subsume_false_action",
                       {"c": "10##0#1#", "a": 6, "e": "01####0#", "q": 0.84, "r": 0.33, "exp": 3,
                        "oc": "10####2#", "oa": 3, "oe": "01####0#", "oq": 0.5, "or": 0.41, "oexp": 1},
                       False, src, "low experience and action mismatch prevents subsumption"))
    out.append(subsume("does_subsume_true_2",
                       {"c": "######0#", "a": 6, "e": "########", "q": 0.99, "r": 11.4, "exp": 32,
                        "oc": "###1##0#", "oa": 6, "oe": "########", "oq": 0.5, "or": 9.89, "oexp": 1},
                       True, src, "non-changing general classifier subsumes specific"))

    def subsumer(id_, exp, q, marked, expect, source, desc):
        cfg = make_cfg()
        cl = Classifier(experience=exp, quality=q, cfg=cfg)
        if marked:
            cl.mark[3].add("1")
        result = is_subsumer(cl, cfg.theta_exp)
        assert result is expect, (id_, result)
        return fixture(
            id_, "subsumption.is_subsumer", source, desc,
            {"cl": cl_to_dict(cl), "theta_exp": cfg.theta_exp, "theta_r": cfg.theta_r},
            {"mode": "exact", "result": result},
        )

    src2 = "test: tests/lcs/strategies/test_subsumption.py::test_should_distinguish_classifier_as_subsumer"
    out.append(subsumer("is_subsumer_too_young", 1, 0.5, False, False, src2, "young low-quality is not a subsumer"))
    out.append(subsumer("is_subsumer_ok", 30, 0.92, False, True, src2, "experienced reliable unmarked is a subsumer"))
    out.append(subsumer("is_subsumer_not_experienced", 15, 0.92, False, False, src2, "insufficient experience"))
    out.append(subsumer("is_subsumer_marked", 30, 0.92, True, False,
                        "test: tests/lcs/strategies/test_subsumption.py::test_should_not_distinguish_marked_classifier_as_subsumer",
                        "marked classifier is never a subsumer"))

    return out


def build_numerosity_collapse() -> List[Dict]:
    out = []
    cfg = make_cfg()

    def eq(id_, a, b, expect, desc):
        cla = Classifier(condition=a["c"], action=a["a"], effect=a["e"],
                         quality=a.get("q", 0.5), reward=a.get("r", 0.5),
                         numerosity=a.get("num", 1), cfg=cfg)
        clb = Classifier(condition=b["c"], action=b["a"], effect=b["e"],
                         quality=b.get("q", 0.5), reward=b.get("r", 0.5),
                         numerosity=b.get("num", 1), cfg=cfg)
        result = (cla == clb)
        assert result is expect, (id_, result)
        return fixture(
            id_, "Classifier.__eq__", "generated: deterministic", desc,
            {"cl": cl_to_dict(cla), "other": cl_to_dict(clb)},
            {"mode": "exact", "result": result},
        )

    out.append(eq("identity_same_cae_differ_qrn",
                  {"c": "1##0####", "a": 3, "e": "##1#####", "q": 0.9, "r": 5.0, "num": 4},
                  {"c": "1##0####", "a": 3, "e": "##1#####", "q": 0.2, "r": 0.1, "num": 1},
                  True, "identity ignores q, r, numerosity"))
    out.append(eq("identity_differ_condition",
                  {"c": "1##0####", "a": 3, "e": "##1#####"},
                  {"c": "1##1####", "a": 3, "e": "##1#####"}, False, "different condition breaks identity"))
    out.append(eq("identity_differ_action",
                  {"c": "1##0####", "a": 3, "e": "##1#####"},
                  {"c": "1##0####", "a": 4, "e": "##1#####"}, False, "different action breaks identity"))
    out.append(eq("identity_differ_effect",
                  {"c": "1##0####", "a": 3, "e": "##1#####"},
                  {"c": "1##0####", "a": 3, "e": "##0#####"}, False, "different effect breaks identity"))

    def add_dedup(id_, pop_specs, child_spec, source, desc):
        population = ClassifiersList(*[
            Classifier(condition=s["c"], action=s["a"], effect=s["e"],
                       quality=s["q"], reward=s.get("r", 0.5), experience=s.get("exp", 1), cfg=cfg)
            for s in pop_specs
        ])
        new_list = ClassifiersList()
        child = Classifier(condition=child_spec["c"], action=child_spec["a"], effect=child_spec["e"],
                           quality=child_spec["q"], reward=child_spec.get("r", 0.5),
                           experience=child_spec.get("exp", 1), cfg=cfg)
        input_population = [cl_to_dict(cl) for cl in population]
        input_child = cl_to_dict(child)
        add_classifier(child, population, new_list, cfg.theta_exp)
        merged = len(new_list) == 0
        return fixture(
            id_, "alp.add_classifier", source, desc,
            {
                "population": input_population,
                "new_list": [],
                "child": input_child,
                "theta_exp": cfg.theta_exp,
            },
            {
                "mode": "exact",
                "new_list_size": len(new_list),
                "population_size": len(population),
                "merged_existing": merged,
                "population": sort_population([cl_to_dict(cl) for cl in population]),
            },
        )

    out.append(add_dedup(
        "add_classifier_identical_merges",
        [{"c": "1##0####", "a": 3, "e": "##1#####", "q": 0.5, "exp": 1}],
        {"c": "1##0####", "a": 3, "e": "##1#####", "q": 0.5, "exp": 1},
        "generated: deterministic",
        "identical (C,A,E) child increases existing quality, not appended"))

    out.append(add_dedup(
        "add_classifier_distinct_appends",
        [{"c": "1##0####", "a": 3, "e": "##1#####", "q": 0.5, "exp": 1}],
        {"c": "0##0####", "a": 3, "e": "##1#####", "q": 0.5, "exp": 1},
        "generated: deterministic",
        "distinct child with no subsumer is appended to the new list"))

    return out


def build_learning_step() -> List[Dict]:
    out = []

    def transition(id_, cfg_over, pop_specs, p0, action, p1, time, reward, max_fitness, source, desc):
        cfg = make_cfg(length=len(p0), **cfg_over)

        def build_population():
            return ClassifiersList(*[
                Classifier(
                    condition=s["c"], action=s["a"], effect=s["e"],
                    quality=s.get("q", 0.5), reward=s.get("r", 0.5),
                    immediate_reward=s.get("ir", 0.0), experience=s.get("exp", 1),
                    talp=s.get("talp", 0), tga=s.get("tga", 0), tav=s.get("tav", 0.0), cfg=cfg)
                for s in pop_specs
            ])

        population = build_population()
        p0p, p1p = perception(p0), perception(p1)
        action_set = ClassifiersList(*[cl for cl in population
                                       if cl.action == action and cl.does_match(p0p)])
        match_set = ClassifiersList(*[cl for cl in population if cl.does_match(p1p)])

        ClassifiersList.apply_alp(population, match_set, action_set, p0p, action, p1p,
                                  time, cfg.theta_exp, cfg)
        ClassifiersList.apply_reinforcement_learning(action_set, reward, max_fitness,
                                                     cfg.beta, cfg.gamma)

        return fixture(
            id_, "ClassifiersList.apply_alp+apply_reinforcement_learning", source, desc,
            {
                "config": {"classifier_length": cfg.classifier_length, "beta": cfg.beta,
                           "gamma": cfg.gamma, "theta_exp": cfg.theta_exp,
                           "theta_r": cfg.theta_r, "theta_i": cfg.theta_i},
                "population": [cl_to_dict(cl) for cl in build_population()],
                "p0": list(p0), "action": action, "p1": list(p1), "time": time,
                "reward": reward, "max_fitness": max_fitness,
            },
            {"mode": "exact", "population_after": sort_population([cl_to_dict(cl) for cl in population])},
        )

    out.append(transition(
        "learning_step_covering", {}, [],
        "00000000", 0, "01000000", 10, 0, 0.0,
        "generated: deterministic (covering only)",
        "empty action set triggers deterministic covering"))

    out.append(transition(
        "learning_step_unexpected", {},
        [{"c": "########", "a": 0, "e": "########", "q": 0.5, "exp": 1, "r": 0.5}],
        "00000000", 0, "01000000", 10, 0, 0.0,
        "generated: deterministic (unexpected + covering dedup)",
        "incorrect anticipation specializes a child that covering then merges"))

    out.append(transition(
        "learning_step_expected_none", {},
        [{"c": "#0######", "a": 0, "e": "#1######", "q": 0.6, "exp": 5, "r": 0.5, "talp": 0, "tav": 0.0}],
        "00000000", 0, "01000000", 10, 1000, 12.5,
        "generated: deterministic (expected case, no child)",
        "correct anticipation increases quality and applies reinforcement"))

    return out


def main() -> None:
    random.seed(0)
    categories = {
        "condition_matching": build_condition_matching(),
        "effect_anticipation": build_effect_anticipation(),
        "mark_differences": build_mark_differences(),
        "alp_expected_case": build_expected_case(),
        "alp_unexpected_case": build_unexpected_case(),
        "alp_covering": build_covering(),
        "updates_qrir_tav": build_updates(),
        "subsumption": build_subsumption(),
        "numerosity_collapse": build_numerosity_collapse(),
        "learning_step": build_learning_step(),
    }

    total = 0
    for name, fixtures in categories.items():
        for fx in fixtures:
            fx["category"] = name
        path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"category": name, "fixtures": fixtures}, handle, indent=2)
            handle.write("\n")
        total += len(fixtures)
        print(f"{name}: {len(fixtures)} fixtures -> {os.path.basename(path)}")

    print(f"TOTAL: {total} fixtures")


if __name__ == "__main__":
    main()
