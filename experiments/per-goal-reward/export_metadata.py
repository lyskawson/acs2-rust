import json
from pathlib import Path

from protocol import MECHANISM_METRICS
from grid_protocol import atomic_json


def column_semantics():
    return dict(
        D_tie_warning='D_tie_* are inherited compatibility names, not evidence of action-score ties in E+1/S+1. A classifier with no goal attribute still uses r(goal), so its score can change with the goal and reverse action rankings without a tie. The exported profiles do not include losing-action scores and cannot establish the frequency of actual cross-action ties.',
        D_pair_population='Nonfallback pairs with the same start, different goals and disjoint optimal-action sets, weighted by the product of recorded pair weights.',
        D_tie_weight='Weight of pairs with different chosen actions and at least one winning classifier with zero specified goal attributes in EACH profile. Scores are not compared across goals or across all actions.',
        D_tie_implied='D_tie_weight divided by the weight of all eligible D pairs; this is a structural co-occurrence fraction, not a tie fraction in E+1/S+1.',
        D_tie_share='D_tie_weight divided by D_different_weight; this is a structural co-occurrence share among action changes, not a tie share in E+1/S+1.',
        D_different_weight='Weight of eligible D pairs with different chosen actions.',
        winner_goal_attributes='specified_goal_attributes describes the syntax of a classifier condition. A weighted winner-profile share credits each decision equally across its globally top-scoring winning macroclassifiers; fallback contributes zero. This is not the share of decisions depending on the goal, and it does not identify a hidden classifier selected by shuffling.')


def annotate_outputs(directory):
    directory = Path(directory)
    path = directory / 'mechanism-seed-metrics.json'
    payload = json.loads(path.read_text())
    payload['column_semantics'] = column_semantics()
    atomic_json(path, payload)
    extra = sorted(set(payload['rows'][0]) - set(MECHANISM_METRICS) - {'maze', 'arm', 'seed'})
    scope = dict(plan_sha256=payload['plan_sha256'],
        registered_outputs=['registered-comparisons.csv', 'registered-comparisons.json', 'registered-decisions.json'],
        registered_seed_metrics={'success-seed-metrics': ['primary', 'final'], 'mechanism-seed-metrics': list(MECHANISM_METRICS)},
        post_hoc_columns={'mechanism-seed-metrics': extra},
        post_hoc_outputs=['post-hoc-curves.csv', 'post-hoc-curves.json', 'post-hoc-curve-summary.csv', 'figures/*'],
        note='Ancillary columns retained by the shared diagnostic exporter are post hoc and have no registered contrasts or verdicts here. Read column_semantics before using the inherited D_tie_* names or winner goal-attribute profiles.',
        column_semantics={'mechanism-seed-metrics': column_semantics()})
    atomic_json(directory / 'output-scope.json', scope)
