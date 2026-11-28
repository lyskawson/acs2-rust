import json
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import unittest

from pair_analysis import counter_tables, read_runs, summarize_run, verdicts
from pair_measures import ANALYSIS, DECISIONS, combine, distance_bin, distance_sums, judge, measures, paired, pooled
from pair_register import COUNTERS, predictions
from pair_sources import ARMS, E1, S1, TU_ARMS, pair_universe

T4 = 2.7764451051977987
GAMMA, REWARD = 0.95, 1000.0
GOAL = 'Goal { symbols: [Token(%d), Token(%d)] }'


def value(distance, cap):
    return REWARD * GAMMA ** (distance - 1) if 0 < distance <= cap else 0.0


def truth_pair(start, goal, distance, successor_distances, cap):
    values = [REWARD if d == 0 else GAMMA * value(d, cap - 1) for d in successor_distances]
    best = max(values)
    return dict(start=list(start), goal=list(goal), shortest_distance=distance, value=value(distance, cap),
                action_values=values, optimal_actions=[a for a, v in enumerate(values) if v == best],
                successors=[list(start)] * 8)


def model(cap=5):
    near = truth_pair((1, 1), (1, 3), 2, [2, 3, 3, 3, 1, 3, 3, 3], cap)
    far = truth_pair((1, 1), (5, 5), 7, [7, 8, 8, 8, 6, 8, 8, 8], cap)
    three = truth_pair((2, 2), (2, 5), 3, [3, 4, 4, 2, 4, 4, 4, 4], cap)
    return ANALYSIS.truth_index(dict(gamma=0.95, reward=1000, actions=8, cap=cap, pairs=[near, far, three]))


def record(start, goal, distance, action, estimate, fallback=False, success=False, weight=1.0):
    return dict(start=str(tuple(start)), goal=GOAL % tuple(goal), shortest_distance=distance, first_action=action,
                estimated_first_action_value=estimate, success=success, weight=weight, steps=distance,
                discounted_return=0.0, first_decision=dict(random_fallback=fallback, winners=[] if fallback else [
                    dict(specified_goal_attributes=0, specified_state_attributes=0, q=1.0, r=1.0, fitness=1.0,
                         experience=1, numerosity=1, selection_score=1.0)]))


def cases():
    return [record((1, 1), (1, 3), 2, 4, 960.0, success=True),
            record((1, 1), (1, 3), 2, 0, 900.0),
            record((1, 1), (1, 3), 2, 4, 10.0, fallback=True, success=True),
            record((1, 1), (5, 5), 7, 2, 5.0),
            record((2, 2), (2, 5), 3, 1, 700.0)]


class DecisionDefinitions(unittest.TestCase):
    def test_wall_move_and_fallback_follow_per_pair_diagnostics(self):
        attached = ANALYSIS.attach(cases(), model())
        wall = attached[1]
        self.assertEqual(wall['q_star'], GAMMA * GAMMA * REWARD)
        self.assertNotIn(0, wall['optimal'])
        sums = combine(distance_sums(attached).values())
        result = measures(sums)
        self.assertEqual((sums['eligible'], sums['optimal'], sums['fallbacks']), (4, 2, 1))
        self.assertEqual(result['F'], 0.5)
        weighted = [dict(item, weight=1 / len(attached)) for item in attached]
        reference = ANALYSIS.statistics_for(weighted)
        self.assertAlmostEqual(result['F'], reference['F'], places=15)
        self.assertAlmostEqual(result['fallback_share'], reference['random_fallback_share'], places=15)
        self.assertAlmostEqual(result['value_error'], reference['values']['all']['estimate_minus_q'], places=9)
        self.assertAlmostEqual(result['overestimation'], DECISIONS.decision_metrics(weighted)['estimate_above_v'], places=15)
        errors = [item['estimated_first_action_value'] - item['q_star'] for item in attached]
        self.assertAlmostEqual(result['absolute_value_error'], statistics.mean(map(abs, errors)), places=9)
        self.assertNotEqual(result['absolute_value_error'], abs(result['value_error']))
        self.assertEqual(result['success'], 0.4)

    def test_fallback_counts_neither_as_optimal_nor_as_non_optimal(self):
        only_fallback = ANALYSIS.attach([cases()[2]], model())
        self.assertIsNone(measures(combine(distance_sums(only_fallback).values()))['F'])
        with_fallback = ANALYSIS.attach(cases()[:3], model())
        without = ANALYSIS.attach(cases()[:2], model())
        self.assertEqual(measures(combine(distance_sums(with_fallback).values()))['F'],
                         measures(combine(distance_sums(without).values()))['F'])

    def test_pair_beyond_cap_has_zero_truth_and_every_action_optimal(self):
        far = ANALYSIS.attach([cases()[3]], model())[0]
        self.assertEqual((far['q_star'], far['v_star']), (0.0, 0.0))
        self.assertEqual(far['optimal'], set(range(8)))
        result = measures(combine(distance_sums([far]).values()))
        self.assertEqual((result['F'], result['overestimation'], result['value_error']), (1.0, 1.0, 5.0))
        self.assertEqual(distance_bin(far['shortest_distance']), '7+')

    def test_recorded_distance_must_match_the_truth(self):
        wrong = dict(cases()[0], shortest_distance=3)
        with self.assertRaisesRegex(ValueError, 'STOP'):
            ANALYSIS.attach([wrong], model())

    def test_distance_bins(self):
        self.assertEqual([distance_bin(d) for d in (1, 2, 3, 4, 5, 6, 7, 8, 30)],
                         ['1-2', '1-2', '3-4', '3-4', '5-6', '5-6', '7+', '7+', '7+'])
        for bad in (0, -1, 2.0, None):
            with self.assertRaises(ValueError):
                distance_bin(bad)


def sums(pairs, optimal, error=0.0):
    return dict(pairs=pairs, fallbacks=0, eligible=pairs, optimal=optimal, error=error, absolute_error=abs(error),
                overestimated=0, successes=optimal)


class Pooling(unittest.TestCase):
    def test_unequal_maze_sizes_weight_pairs_not_mazes(self):
        cell = {'small': {2: sums(1, 1)}, 'large': {2: sums(2, 0), 4: sums(1, 0)}}
        pooled_sums = pooled(cell, ['small', 'large'], [2, 4])
        self.assertEqual(measures(pooled_sums)['F'], 0.25)
        maze_means = statistics.mean([measures(pooled(cell, [task], [2, 4]))['F'] for task in cell])
        self.assertEqual(maze_means, 0.5)
        self.assertEqual(measures(pooled(cell, ['small', 'large'], [2]))['F'], 1 / 3)
        self.assertIsNone(pooled(cell, ['small'], [4]))

    def test_t4_interval_with_nonzero_variance(self):
        left = {42: 0.6, 43: 0.5, 44: 0.9, 45: 0.7, 46: 0.8}
        right = {seed: 0.4 for seed in left}
        result = paired(left, right)
        differences = [left[seed] - 0.4 for seed in sorted(left)]
        se = statistics.stdev(differences) / math.sqrt(5)
        self.assertGreater(se, 0)
        self.assertAlmostEqual(result['mean'], 0.3)
        self.assertAlmostEqual(result['ci_low'], 0.3 - T4 * se)
        self.assertAlmostEqual(result['ci_high'], 0.3 + T4 * se)
        self.assertEqual(judge(result, 1), 'supported')
        self.assertEqual(judge(result, -1), 'contradicted')
        wide = paired({42: 0.9, 43: -0.5, 44: 0.4, 45: -0.2, 46: 0.1}, {seed: 0.0 for seed in left})
        self.assertLess(wide['ci_low'], 0)
        self.assertGreater(wide['ci_high'], 0)
        self.assertEqual(judge(wide, 1), 'inconclusive')
        self.assertEqual(judge(wide, -1), 'inconclusive')

    def test_undefined_seed_makes_the_comparison_inconclusive(self):
        left = {42: None, 43: 0.5, 44: 0.9, 45: 0.7, 46: 0.8}
        result = paired(left, {seed: 0.1 for seed in left})
        self.assertIsNone(result['mean'])
        self.assertEqual(result['n'], 4)
        self.assertEqual(judge(result, 1), 'inconclusive')

    def test_comparison_refuses_mixed_seeds(self):
        with self.assertRaises(ValueError):
            paired({seed: 0.5 for seed in range(42, 47)}, {seed: 0.5 for seed in range(43, 48)})


def plan_cells():
    return [dict(group='no_aliasing', contrast='R 1 - S+1', measure='F', predicted_sign=-1, bin='1-2', checkpoint=1000)]


class Verdicts(unittest.TestCase):
    def test_registered_cells(self):
        counts = {item['id']: len(item['cells']) for item in predictions()}
        self.assertEqual(counts, dict(P1=8, P2=8, P3=6, P4=6, P5=4))
        p2 = next(item for item in predictions() if item['id'] == 'P2')
        self.assertEqual({cell['measure'] for cell in p2['cells']}, {'value_error'})
        self.assertEqual({cell['bin'] for cell in p2['cells']}, {'5-6', '7+'})

    def test_verdict_uses_the_named_cell(self):
        plan = dict(predictions=[dict(id='P3', cells=plan_cells())])
        row = dict(group='no_aliasing', contrast='R 1 - S+1', measure='F', bin='1-2', checkpoint=1000, n=5,
                   mean=-0.2, ci_low=-0.3, ci_high=-0.1)
        other = dict(row, checkpoint=3000, mean=0.2, ci_low=0.1, ci_high=0.3)
        cells, summary = verdicts(plan, [row, other])
        self.assertEqual(cells[0]['verdict'], 'supported')
        self.assertEqual(summary['P3'], dict(cells=1, supported=1, contradicted=0, inconclusive=0))


def row(step, starts, test_pairs=3):
    return dict(nominal_step=step, task='toy', agent=S1, seed=42, cap=5,
                evaluation_blocks=dict(test=dict(evaluation_pairs=test_pairs, starts=starts)),
                replay_diagnostics=dict(realized_relabeled_share=0.5, fallbacks=2))


def config():
    return dict(targets=[1000, 3000], cap=5, group='no_aliasing',
                registration=dict(blocks=dict(test=dict(evaluation_pairs=3, pair_weight='1/3'))))


class MissingRecords(unittest.TestCase):
    def run_file(self, rows):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            path.write_text(''.join(json.dumps(item) + '\n' for item in rows))
            truth = Path(folder) / 'truth.json'
            truth.write_text(json.dumps(dict(gamma=0.95, reward=1000, actions=8, cap=5, pairs=[
                truth_pair((1, 1), (1, 3), 2, [2, 3, 3, 3, 1, 3, 3, 3], 5),
                truth_pair((1, 1), (5, 5), 7, [7, 8, 8, 8, 6, 8, 8, 8], 5),
                truth_pair((2, 2), (2, 5), 3, [3, 4, 4, 2, 4, 4, 4, 4], 5)])))
            from pair_sources import sha256
            return summarize_run((str(path), sha256(path), 'toy', S1, 42, config(), str(truth), sha256(truth)))

    def test_complete_records_are_summarized_per_checkpoint(self):
        starts = [dict(item, weight=1 / 3) for item in (cases()[0], cases()[3], cases()[4])]
        _, _, _, pairs, counters = self.run_file([row(1000, starts), row(3000, starts)])
        self.assertEqual(set(pairs), {1000, 3000})
        self.assertEqual(set(pairs[1000]), {2, 3, 7})
        self.assertEqual(counters[1000]['replay_diagnostics.fallbacks'], 2)

    def test_missing_per_pair_record_stops(self):
        starts = [dict(item, weight=1 / 3) for item in (cases()[0], cases()[3], cases()[4])]
        for broken in ([row(1000, starts), row(3000, [])], [row(1000, starts), row(3000, starts[:2])],
                       [row(1000, starts), dict(row(3000, starts), evaluation_blocks=dict(test=dict(evaluation_pairs=3)))]):
            with self.assertRaisesRegex(ValueError, 'STOP'):
                self.run_file(broken)

    def test_unequal_recorded_weights_stop(self):
        starts = [dict(item, weight=w) for item, w in zip((cases()[0], cases()[3], cases()[4]), (0.5, 0.25, 0.25))]
        with self.assertRaisesRegex(ValueError, 'equal'):
            pair_universe(dict(evaluation_pairs=3, starts=starts), config())

    def test_missing_run_stops_the_analysis(self):
        plan = dict(inputs_fields=['task', 'arm', 'seed', 'archive', 'path', 'sha256', 'kind'], inputs=[],
                    truth_models={})
        with self.assertRaisesRegex(ValueError, 'STOP'):
            read_runs(plan, {'toy': config()}, Path('.'), 1)


class Counters(unittest.TestCase):
    def test_arm_without_a_counter_is_reported_not_derived(self):
        tu = {name: 1.0 for name in COUNTERS}
        reference = {name: (1.0 if name.startswith('replay_diagnostics') else None) for name in COUNTERS}
        counters = {('toy', arm, seed): {1000: dict(tu if arm in TU_ARMS else reference)}
                    for arm in ARMS for seed in range(42, 62)}
        availability, per_maze, _ = counter_tables(counters, {'toy': dict(config(), targets=[1000])})
        status = {(item['arm'], item['counter']): item['status'] for item in availability}
        self.assertEqual(status[E1, 'trajectory_utility.diagnostics.distinct_assigned_goals'], 'not recorded')
        self.assertEqual(status[S1, 'replay_diagnostics.fallbacks'], 'recorded')
        self.assertEqual(status[TU_ARMS[0], 'trajectory_utility.diagnostics.tu_fallbacks'], 'recorded')
        self.assertFalse(any(item['arm'] == E1 and item['counter'].startswith('trajectory_utility') for item in per_maze))


if __name__ == '__main__':
    unittest.main()
