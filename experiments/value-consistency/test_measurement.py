import copy
import json
from pathlib import Path
import struct
import sys
import unittest

from protocol import ROOT, configurations
from archive import require_coverage
from pilot_equivalence import compare, CONTROL_CHANGED_FIELDS

sys.path.append(str(ROOT / 'experiments/grid'))
from collect import validate_row
from vcp_schema import POSITIVE_UPPER_BOUNDS


def decode(value):
    if isinstance(value, str) and value.startswith('f64:'):
        return struct.unpack('>d', bytes.fromhex(value[4:]))[0]
    if isinstance(value, list):
        return list(map(decode, value))
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / 'fixtures/vcp_pair_regression.json').read_text())
        self.configs = {config['task']: dict(config, evaluation_blocks=['validation']) for config in configurations()}
        self.rows = [dict(decode(row), commit='pinned', source_state='clean', host='fixture-host',
                         cpu_model='AMD EPYC 9554 64-Core Processor')
                     for config in self.fixture['configurations'] for row in config['rows']]
        self.temperatures = [config['temperature'] for config in self.fixture['configurations'] for _ in config['rows']]

    def validate(self, row, temperature=1):
        config = dict(self.configs[row['task']], agent_settings={row['agent']: dict(temperature=temperature)})
        validate_row(row, config, row['agent'], row['seed'], 'pinned', row['nominal_step'])

    def test_actual_pilot_rows_pass_shared_collection_guards(self):
        self.assertGreaterEqual(len(self.rows), 8)
        for row, temperature in zip(self.rows, self.temperatures):
            self.validate(row, temperature)

    def test_pinned_scope_contains_every_first_temperature_control_and_both_heads_of_the_declared_second_temperature(self):
        selected = {(self.configs[row['task']]['id'], row['agent'], row['agent_parameters']['temperature']): None
                    for row in self.rows}
        second = require_coverage(selected, True)
        self.assertEqual(second, self.fixture['second_temperature_mazes'])
        self.assertEqual(len(selected), self.fixture['source_pilot_runs'])
        self.assertEqual(len(self.rows), 2 * self.fixture['source_pilot_runs'])

    def test_ensemble_cost_and_behavior_defects_are_refused(self):
        row = next(row for row in self.rows if row['agent'] == 'acs2vcp_k4')
        mutations = [
            (['population_classifiers'], row['population_classifiers'] * 2),
            (['replay_diagnostics', 'priority', 'value_queries'], row['replay_diagnostics']['priority']['value_queries'] + 1),
            (['replay_diagnostics', 'priority', 'accepted_draws'], row['replay_updates'] + 1),
            (['agent_parameters', 'priority_clock'], 'after_learning'),
            (['agent_parameters', 'sampler_stream'], 1),
            (['agent_parameters', 'temperature'], 2),
            (['agent_parameters', 'importance_weights'], 'inverse_probability'),
            (['match_formations_train'], row['match_formations_train'] - 1),
            (['replay_diagnostics', 'objective_evaluations_by_purpose', 'scoring', 'objective_evaluations'], 0),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(row)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_a_head_zero_only_metric_or_an_acceptance_coin_in_the_control_is_refused(self):
        row = copy.deepcopy(next(row for row in self.rows if any(
            abs(head['success'] - row['evaluation_blocks']['validation']['success']) > 1e-12
            for head in row['evaluation_blocks']['validation']['heads'][:1])))
        row['evaluation_blocks']['validation']['success'] = row['evaluation_blocks']['validation']['heads'][0]['success']
        with self.assertRaises(ValueError):
            self.validate(row)
        control = copy.deepcopy(next(row for row in self.rows if row['agent'].endswith('_uniform')))
        control['replay_diagnostics']['priority']['acceptance_coins'] = 1
        with self.assertRaises(ValueError):
            self.validate(control)

    def revised_row(self, pilot):
        row = copy.deepcopy(pilot)
        heads = row['agent_parameters']['heads']
        priority = row['replay_diagnostics']['priority']
        prioritized = row['agent_parameters']['temperature'] > 0
        priority['training_variance_measured'] = prioritized
        def logarithmic(count):
            return dict(count=count, exact_zero_count=0, exact_zero_share=0.0,
                        positive_bin_counts=[count] + [0] * (len(POSITIVE_UPPER_BOUNDS) - 1),
                        positive_bin_upper_bounds=POSITIVE_UPPER_BOUNDS,
                        bin_rule='exact_zero_separate_positive_lower_exclusive_upper_inclusive')
        if prioritized:
            for name in ['proposal_variance', 'accepted_variance']:
                priority[name]['logarithmic'] = logarithmic(priority[name]['count'])
        else:
            priority['avoided_value_queries'] = priority['value_queries']
            priority['avoided_classifier_perception_tests'] = 100
            row['match_formations_train'] -= priority['value_queries']
            row['classifier_perception_tests_train'] -= 100
            priority['value_queries'] = 0
            for name in ['proposal_variance', 'accepted_variance', 'candidate_acceptance_sums']:
                priority[name] = None
        row['ensemble']['measurement_diagnostics_inline_bytes'] = 432
        row['ensemble']['agent_and_measurement_known_bytes_lower_bound'] = row['ensemble']['agent_known_bytes_lower_bound'] + 432
        for block in row['evaluation_blocks'].values():
            block['head_disagreement'] = dict(count=1, sum=0.0, sum_squares=0.0, minimum=0.0,
                maximum=0.0, mean=0.0, histogram=[1] + [0] * 15,
                logarithmic=dict(logarithmic(1), exact_zero_count=1, exact_zero_share=1.0,
                                 positive_bin_counts=[0] * len(POSITIVE_UPPER_BOUNDS)),
                exact_zero_share=1.0, quantiles={name: 0.0 for name in ['p25', 'p50', 'p75', 'p90', 'p99']},
                value_queries=heads, variance_maximum=250000.0, quantile_rule='nearest_rank_ceil_n_times_percentile',
                reference_set='non_identity_knowledge_transitions_from_evaluated_block_start_goal_pairs_equal_transition_weights',
                cost_scope='evaluation_only_training_counters_paused')
        return row

    def test_revised_rows_require_query_free_controls_logarithmic_summaries_and_evaluation_disagreement(self):
        for agent in ['acs2vcp_k2', 'acs2vcp_k4_uniform']:
            pilot = next(row for row in self.rows if row['agent'] == agent)
            row = self.revised_row(pilot)
            self.validate(row)
            self.assertEqual(compare(pilot, row), CONTROL_CHANGED_FIELDS if agent.endswith('_uniform') else [])
            for path, value in [
                    (['evaluation_blocks', 'validation', 'head_disagreement', 'value_queries'], 0),
                    (['evaluation_blocks', 'validation', 'head_disagreement', 'logarithmic', 'exact_zero_count'], 0),
                    (['evaluation_blocks', 'validation', 'head_disagreement', 'cost_scope'], 'training'),
                    (['ensemble', 'agent_and_measurement_known_bytes_lower_bound'], 0),
                    (['replay_diagnostics', 'priority', 'value_queries'], 1)]:
                with self.subTest(agent=agent, path=path):
                    changed = copy.deepcopy(row)
                    target = changed
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    with self.assertRaises(ValueError):
                        self.validate(changed)

    def test_pilot_equivalence_allows_only_the_six_documented_control_field_changes(self):
        pilot = next(row for row in self.rows if row['agent'].endswith('_uniform'))
        row = self.revised_row(pilot)
        for field in ['knowledge', 'population_classifiers', 'replay_updates', 'classifier_perception_tests_train']:
            changed = copy.deepcopy(row)
            changed[field] += 0.125 if field == 'knowledge' else 1
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    compare(pilot, changed)
        config = dict(self.configs[pilot['task']], vcp_diagnostics='evaluation_disagreement')
        with self.assertRaises(ValueError):
            validate_row(pilot, config, pilot['agent'], 42, 'pinned', pilot['nominal_step'])

    def test_revised_controls_refuse_nonzero_value_queries_and_nonnull_training_variances(self):
        for agent in ['acs2vcp_k2_uniform', 'acs2vcp_k4_uniform']:
            pilot = next(row for row in self.rows if row['agent'] == agent)
            row = self.revised_row(pilot)
            self.validate(row)
            for queries in [1, row['agent_parameters']['heads'] * row['replay_diagnostics']['priority']['proposals']]:
                changed = copy.deepcopy(row)
                changed['replay_diagnostics']['priority']['value_queries'] = queries
                changed['match_formations_train'] += queries
                with self.subTest(agent=agent, queries=queries):
                    with self.assertRaisesRegex(ValueError, 'priority queries'):
                        self.validate(changed)
            for field in ['proposal_variance', 'accepted_variance']:
                changed = copy.deepcopy(row)
                changed['replay_diagnostics']['priority'][field] = copy.deepcopy(pilot['replay_diagnostics']['priority'][field])
                with self.subTest(agent=agent, variance=field):
                    with self.assertRaisesRegex(ValueError, 'unmeasured training variance'):
                        self.validate(changed)


if __name__ == '__main__':
    unittest.main()
