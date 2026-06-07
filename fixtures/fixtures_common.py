import random
from typing import Callable, Dict, List, Optional

from lcs import Perception
from lcs.agents.acs import Condition
from lcs.agents.acs2 import Classifier, Configuration, Effect
from lcs.agents.acs2.alp import expected_case

WILDCARD = "#"
PROPERTY_SAMPLES = 2000


def make_cfg(length: int = 8, actions: int = 8, **overrides) -> Configuration:
    return Configuration(
        classifier_length=length,
        number_of_possible_actions=actions,
        **overrides,
    )


def seq_to_list(seq) -> List[str]:
    return [str(a) for a in seq]


def mark_to_list(mark) -> List[List[str]]:
    return [sorted(str(v) for v in attribute) for attribute in mark]


def perception(values: List[str]) -> Perception:
    return Perception([str(v) for v in values])


def cl_to_dict(cl: Classifier) -> Dict:
    return {
        "condition": seq_to_list(cl.condition),
        "action": cl.action,
        "effect": seq_to_list(cl.effect),
        "mark": mark_to_list(cl.mark),
        "q": cl.q,
        "r": cl.r,
        "ir": cl.ir,
        "num": cl.num,
        "exp": cl.exp,
        "talp": cl.talp,
        "tga": cl.tga,
        "tav": cl.tav,
    }


def dict_to_cl(data: Dict, cfg: Configuration) -> Classifier:
    cl = Classifier(
        condition=list(data["condition"]),
        action=data["action"],
        effect=list(data["effect"]),
        quality=data["q"],
        reward=data["r"],
        immediate_reward=data["ir"],
        numerosity=data["num"],
        experience=data["exp"],
        talp=data["talp"],
        tga=data["tga"],
        tav=data["tav"],
        cfg=cfg,
    )
    for idx, values in enumerate(data["mark"]):
        for value in values:
            cl.mark[idx].add(str(value))
    return cl


def canonical_key(cl_dict: Dict):
    return (
        "".join(cl_dict["condition"]),
        cl_dict["action"],
        "".join(cl_dict["effect"]),
    )


def sort_population(pop_dicts: List[Dict]) -> List[Dict]:
    return sorted(pop_dicts, key=canonical_key)


def get_differences_candidates(mark_list: List[List[str]], p0: List[str]) -> List[int]:
    return [
        idx
        for idx, values in enumerate(mark_list)
        if len(values) > 0 and str(p0[idx]) not in set(values)
    ]


def sample_get_differences(
    mark_factory: Callable[[], "Perception"],
    p0_values: List[str],
    samples: int = PROPERTY_SAMPLES,
) -> Dict:
    p0 = perception(p0_values)
    chosen_indices = set()
    specificities = set()
    for seed in range(samples):
        random.seed(seed)
        mark = mark_factory()
        diff = mark.get_differences(p0)
        specificities.add(diff.specificity)
        for idx, attr in enumerate(diff):
            if attr != WILDCARD:
                chosen_indices.add(idx)
    return {
        "allowed_specialized_indices": sorted(chosen_indices),
        "allowed_specificity": sorted(specificities),
    }


def sample_expected_case(
    cl_factory: Callable[[], Classifier],
    p0_values: List[str],
    time: int,
    samples: int = PROPERTY_SAMPLES,
) -> Dict:
    p0 = perception(p0_values)
    specificities = set()
    effects = set()
    actions = set()
    qualities = set()
    marked_flags = set()
    for seed in range(samples):
        random.seed(seed)
        cl = cl_factory()
        child = expected_case(cl, p0, time)
        assert child is not None
        specificities.add(child.condition.specificity)
        effects.add("".join(seq_to_list(child.effect)))
        actions.add(child.action)
        qualities.add(round(child.q, 12))
        marked_flags.add(child.is_marked())
    return {
        "allowed_specificity": sorted(specificities),
        "effect": list(effects)[0] if len(effects) == 1 else sorted(effects),
        "effect_invariant": len(effects) == 1,
        "action": list(actions)[0] if len(actions) == 1 else sorted(actions),
        "q": list(qualities)[0] if len(qualities) == 1 else sorted(qualities),
        "is_marked": list(marked_flags)[0] if len(marked_flags) == 1 else sorted(marked_flags),
    }
