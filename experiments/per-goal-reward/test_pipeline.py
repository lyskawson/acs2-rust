import copy
import math
import unittest
from unittest.mock import patch

from protocol import ARMS, NEW_ARMS, E, E1, S, S1, G, BLOCKS, configurations
from analyze import calculate


class PipelineTests(unittest.TestCase):
    def test_driver_separates_arms_seeds_mazes_checkpoints_and_unequal_blocks(self):
        configs = [dict(config, targets=[11, 37]) for config in configurations()]
        inputs = dict(new={}, success={}, pairs={})
        storage, truth, expected = {}, {}, {}
        for m, config in enumerate(configs):
            maze = config['task']
            truth[maze] = {}
            for b, n in enumerate([2, 3, 5]):
                for i in range(n):
                    truth[maze][(0, 0), (b, i)] = dict(cap=1, shortest_distance=1, value=1000.,
                        action_values=[1000., 0., 0., 0., 0., 0., 0., 0.], optimal_actions=[0])
            for a, arm in enumerate(ARMS):
                for seed in range(42, 62):
                    s = seed - 42
                    rows = []
                    for p, point in enumerate(config['targets']):
                        blocks, weighted = {}, []
                        slope = .5 + .13 * a + .003 * s + .017 * m + .07 * p
                        for b, (name, n) in enumerate(zip(BLOCKS, [2, 3, 5])):
                            records = []
                            for i in range(n):
                                action = (a + s + m + 2 * b + i + p) % 2
                                fallback = (2 * a + s + m + b + i + p) % 7 == 0
                                q = 1000. if action == 0 else 0.
                                estimate = slope * q + 19 + 7 * a + 2 * s + m
                                weighted.append((1 / (3 * n), action, fallback, q, estimate))
                                records.append(dict(start='(0, 0)', goal=f'Goal {{ symbols: [Token({b}), Token({i})] }}',
                                    weight=1 / n, first_action=action, shortest_distance=1,
                                    estimated_first_action_value=estimate, success=action == 0, discounted_return=q,
                                    first_decision=dict(random_fallback=fallback, winners=[] if fallback else [dict(specified_goal_attributes=a % 3)])))
                            blocks[name] = dict(success=.01 * (a + 1) ** 2 * (1 + .01 * s + .02 * m + .03 * b + .1 * p), starts=records)
                        rows.append(dict(agent=arm, seed=seed, task=maze, nominal_step=point, evaluation_blocks=blocks))
                        if p == 1:
                            eligible = [(w, action, estimate) for w, action, fallback, _, estimate in weighted if not fallback]
                            mass = math.fsum(w for w, _, _ in eligible)
                            correct = math.fsum(w for w, action, _ in eligible if action == 0)
                            mean_q = math.fsum(w * q for w, _, _, q, _ in weighted)
                            q_variance = math.fsum(w * (q - mean_q) ** 2 for w, _, _, q, _ in weighted)
                            expected[maze, arm, seed] = dict(F=correct / mass,
                                F_mod=float(correct >= mass - correct),
                                estimate_minus_q=math.fsum(w * (estimate - q) for w, _, _, q, estimate in weighted),
                                estimate_above_v=math.fsum(w for w, _, estimate in eligible if estimate > 1000.) / mass,
                                within_start_slope=slope, within_start_sd_estimate=slope * math.sqrt(q_variance),
                                primary=.01 * (a + 1) ** 2 * (1 + .01 * s + .02 * m + .06 + .05),
                                final=.01 * (a + 1) ** 2 * (1 + .01 * s + .02 * m + .06 + .1))
                    key = config['id'], arm, seed
                    pair_path, compact_path = ('pairs', *key), ('success', *key)
                    if arm in NEW_ARMS:
                        inputs['new'][key] = pair_path
                    else:
                        inputs['success'][key] = compact_path
                        inputs['pairs'][key] = pair_path
                    storage[pair_path] = rows
                    storage[compact_path] = copy.deepcopy(rows)
                    for row in storage[compact_path]:
                        for block in row['evaluation_blocks'].values():
                            block['starts'] = None
        with patch('analyze.read_rows', side_effect=lambda path: iter(storage[path])):
            result = calculate(configs, inputs, truth)
            self.assertEqual(len(result['registered']), 120)
            for row in result['registered']:
                seeds = range(42, 62) if row['group'] == 'success' else range(42, 47)
                differences = []
                for seed in seeds:
                    value = lambda arm: expected[row['maze'], arm, seed][row['metric']]
                    differences.append({'S+1 - S': lambda: value(S1) - value(S),
                        'E+1 - E': lambda: value(E1) - value(E), 'S+1 - G': lambda: value(S1) - value(G),
                        'interaction': lambda: (value(S1) - value(S)) - (value(E1) - value(E))}[row['comparison']]())
                for actual, wanted in zip(row['seed_differences'], differences):
                    self.assertAlmostEqual(actual, wanted, places=9)
            key = configs[0]['id'], E, 42
            saved = storage[inputs['success'][key]]
            storage[inputs['success'][key]] = saved[:-1]
            with self.assertRaises(ValueError): calculate(configs, inputs, truth)
            storage[inputs['success'][key]] = saved
            storage[inputs['success'][key]][0]['agent'] = S
            with self.assertRaises(ValueError): calculate(configs, inputs, truth)
