import copy
import json
import math
import struct
import unittest

from protocol import ROOT, BLOCKS, configurations, recorded
from grid_identity import compare_per_pair
from row_schema import validate_blocks
from pair_schema import validate_pairs


def decode(value):
    if isinstance(value, str) and value.startswith('f64:'):
        return struct.unpack('>d', bytes.fromhex(value[4:]))[0]
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def sample():
    fixture = json.loads((ROOT / 'fixtures/hindsight_admissibility_regression.json').read_text())
    row = decode(fixture['configurations'][0]['rows'][0])
    config = recorded(configurations()[0])
    for name, block in row['evaluation_blocks'].items():
        entries = config['registration']['blocks'][name]['goal_starts']
        count = sum(len(entry['starts']) for entry in entries)
        block['starts'] = [dict(start=str(tuple(start)), goal='Goal { symbols: [Token(%d), Token(%d)] }' % tuple(entry['goal']),
            weight=1.0 / count, first_action=0, estimated_first_action_value=0.5,
            discounted_return=0.0, success=False, steps=config['cap'], shortest_distance=1,
            first_decision=dict(random_fallback=True, winners=[])) for entry in entries for start in entry['starts']]
    return row, config


class SchemaIdentityTests(unittest.TestCase):
    def test_only_opted_in_five_arm_registered_rows_with_every_pair_are_accepted(self):
        row, config = sample()
        validate_blocks(row, config)
        with self.assertRaises(ValueError):
            validate_blocks(row, configurations()[0])
        for key, value in [('per_start_diagnostics', False), ('encoding', 'perception'), ('evaluation_blocks', ['test']),
                           ('protocol', None), ('agent_settings', {'acs2her': {'relabeled_proportion': .5}})]:
            changed = dict(config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_blocks(row, changed)
        row['agent'] = 'acs2her_values'
        with self.assertRaises(ValueError):
            validate_blocks(row, config)

    def test_missing_duplicate_foreign_and_malformed_records_are_refused(self):
        original, config = sample()
        for mutate in [lambda records: records.pop(), lambda records: records.__setitem__(0, records[1]),
                       lambda records: records[0].update(first_action=8),
                       lambda records: records[0].update(weight=.5),
                       lambda records: records[0]['first_decision'].update(random_fallback=False),
                       lambda records: records[0].update(extra=1)]:
            row = copy.deepcopy(original)
            mutate(row['evaluation_blocks']['test']['starts'])
            with self.assertRaises(ValueError):
                validate_blocks(row, config)

    def test_identity_strips_only_records_and_detects_every_mutated_numeric_field(self):
        row, _ = sample()
        reference = copy.deepcopy(row)
        for block in reference['evaluation_blocks'].values():
            block['starts'] = None
        self.assertEqual(compare_per_pair([row], [reference], [42], [1000], BLOCKS)['differences'], 0)
        paths = []
        def scan(value, prefix=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key != 'starts':
                        scan(child, prefix + (key,))
            elif type(value) in (int, float):
                paths.append(prefix)
        scan(reference)
        self.assertGreater(len(paths), 100)
        for path in paths:
            changed = copy.deepcopy(row)
            parent = changed
            for key in path[:-1]:
                parent = parent[key]
            old = parent[path[-1]]
            parent[path[-1]] = math.nextafter(old, math.inf) if type(old) is float else old + 1
            with self.subTest(path=path), self.assertRaises(ValueError):
                compare_per_pair([changed], [reference], [42], [1000], BLOCKS)
        changed = copy.deepcopy(row)
        changed['additional_numeric_field'] = 0.5
        with self.assertRaises(ValueError):
            compare_per_pair([changed], [reference], [42], [1000], BLOCKS)


    def test_different_winner_selection_scores_are_rejected_even_with_valid_individual_scores(self):
        row, config = sample()
        winner = dict(specified_state_attributes=1, specified_goal_attributes=1, q=.5, r=20.0,
                      fitness=10.0, experience=4, numerosity=2, selection_score=20.0)
        other = dict(winner, r=40.0, fitness=20.0, selection_score=40.0)
        record = row['evaluation_blocks']['test']['starts'][0]
        record['first_decision'] = dict(random_fallback=False, winners=[winner, dict(winner)])
        validate_blocks(row, config)
        record['first_decision']['winners'][1] = other
        with self.assertRaisesRegex(ValueError, 'non-tied'):
            validate_blocks(row, config)


if __name__ == '__main__':
    unittest.main()
