import copy
import json
from pathlib import Path
import struct
import unittest

from agent_names import ARCHIVED_RUN_NAMES, agent_settings, canonical_row, canonical_run_name
from collect import validate_row
from grid_identity import compare
from grid_protocol import learning_fields


class AgentNamesTests(unittest.TestCase):
    def test_archived_names_are_complete_and_one_to_one(self):
        self.assertEqual(ARCHIVED_RUN_NAMES, dict(
            acs2='acs2', acs2er='acs2er', acs2er_u1='acs2er_one_update',
            er_episodic='acs2er_episodic', her_full='acs2her',
            her_values='acs2her_values', her_values_shortened='acs2her_values_shortened'))
        self.assertEqual(len(set(ARCHIVED_RUN_NAMES.values())), len(ARCHIVED_RUN_NAMES))
        for old, new in ARCHIVED_RUN_NAMES.items():
            self.assertEqual(canonical_run_name(old), new)
            self.assertEqual(canonical_run_name(new), new)

    def test_normalization_preserves_the_closed_analysis_label_and_other_fields(self):
        row = dict(agent='her_values', arm='her_values_p2of3', knowledge=0.123)
        original = copy.deepcopy(row)
        mapped = canonical_row(row)
        self.assertEqual(mapped, dict(row, agent='acs2her_values'))
        self.assertEqual(row, original)

    def test_identity_compares_new_rows_with_their_own_archived_arm(self):
        for old, new in ARCHIVED_RUN_NAMES.items():
            row = dict(agent=old, seed=42, nominal_step=1000, knowledge=0.1,
                       evaluation_blocks=dict(test=dict(success=0.2)))
            current = dict(row, agent=new)
            self.assertEqual(compare([current], [row], [42], [1000], ['test'])['differences'], 0)
            self.assertEqual(learning_fields(current), learning_fields(row))
            wrong = dict(current, agent=next(name for name in ARCHIVED_RUN_NAMES.values() if name != new))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                compare([wrong], [row], [42], [1000], ['test'])

    def test_settings_are_read_under_both_names_without_changing_manifests(self):
        for old, new in ARCHIVED_RUN_NAMES.items():
            config = dict(agent_settings={old: dict(relabeled_proportion=1 / 3)})
            original = copy.deepcopy(config)
            self.assertEqual(agent_settings(config, new), dict(relabeled_proportion=1 / 3))
            self.assertEqual(agent_settings(config, old), dict(relabeled_proportion=1 / 3))
            self.assertEqual(config, original)

    def test_duplicate_settings_under_archived_and_current_names_are_refused(self):
        config = dict(agent_settings=dict(her_values={}, acs2her_values={}))
        with self.assertRaisesRegex(ValueError, 'duplicate settings'):
            agent_settings(config, 'acs2her_values')

    def test_collection_validates_old_and_new_fixture_names_without_rewriting_rows(self):
        def decode(value):
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            if isinstance(value, str) and value.startswith('f64:'):
                return struct.unpack('>d', bytes.fromhex(value[4:]))[0]
            return value
        fixture = Path(__file__).resolve().parents[2] / 'fixtures/her_pair_regression.json'
        oracle = json.loads(fixture.read_text())
        for configuration in oracle['configurations']:
            row = decode(configuration['rows'][0])
            row.update(commit='pinned', source_state='clean', host='unit-host', cpu_model='unit-cpu',
                       wall_seconds_train=0, wall_seconds_eval=0, wall_seconds_total=0)
            original = copy.deepcopy(row)
            config = dict(task=configuration['task'], cap=configuration['cap'],
                          row_schema=row['schema'], row_pool=row['goal_pool'], encoding=row['goal_encoding'],
                          protocol=row['goal_protocol']['id'], registration=row['goal_protocol']['configuration'],
                          knowledge_transition_goal_pairs=row['knowledge_transition_goal_pairs'],
                          agent_settings={row['agent']: dict(relabeled_proportion=
                              row['agent_parameters'].get('relabeled_proportion', 0))})
            for name in [row['agent'], canonical_run_name(row['agent'])]:
                current = dict(row, agent=name)
                for manifest_name in [row['agent'], canonical_run_name(row['agent'])]:
                    validate_row(current, config, manifest_name, 42, 'pinned', row['nominal_step'])
                self.assertEqual(row, original)
                wrong = dict(current, agent='acs2')
                with self.assertRaisesRegex(ValueError, 'agent: expected'):
                    validate_row(wrong, config, name, 42, 'pinned', row['nominal_step'])


if __name__ == '__main__':
    unittest.main()
