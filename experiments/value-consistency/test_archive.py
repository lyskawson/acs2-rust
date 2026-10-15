import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from archive import audit, require_agreement, require_coverage
from protocol import AGENTS, PRIORITIZED, SECOND_TEMPERATURE_ORDER, configurations


class ArchiveTests(unittest.TestCase):
    def test_archived_binary_substitution_is_refused_even_when_metadata_is_unchanged(self):
        identity = dict(binary_sha256=hashlib.sha256(b'frozen').hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pilot-registration.json').write_text(json.dumps(dict(measurement=identity, authorized_cpu_hours=5)))
            (root / 'acs2-measure-musl').write_bytes(b'substituted')
            with patch('archive.pilot_measurement_identity', return_value=identity):
                with self.assertRaisesRegex(ValueError, 'binary differs'):
                    audit(root)

    def test_recovery_cannot_drop_a_control_or_charge_only_one_head_count_at_the_second_temperature(self):
        configs = configurations()
        first = {(config['id'], agent, 0 if agent.endswith('_uniform') else 1): None
                 for config in configs for agent in AGENTS}
        second = next(config for config in configs if config['task'] == SECOND_TEMPERATURE_ORDER[0])
        selected = first | {(second['id'], agent, 24): None for agent in PRIORITIZED}
        self.assertEqual(require_coverage(selected, True), [SECOND_TEMPERATURE_ORDER[0]])
        for key in [(configs[0]['id'], AGENTS[-1], 0), (second['id'], PRIORITIZED[-1], 24)]:
            changed = dict(selected)
            changed.pop(key)
            with self.assertRaises(ValueError):
                require_coverage(changed, True)
        out_of_order = next(config for config in configs if config['task'] == SECOND_TEMPERATURE_ORDER[1])
        changed = first | {(out_of_order['id'], agent, 24): None for agent in PRIORITIZED}
        with self.assertRaises(ValueError):
            require_coverage(changed, True)

    def test_interrupted_prefixes_must_match_complete_reruns_bit_for_bit(self):
        prefix = dict(nominal_step=1000, knowledge=0.5, host='old', wall_seconds_total=10.0,
                      replay_diagnostics=dict(priority=dict(variance_sum=0.125)))
        complete = dict(copy.deepcopy(prefix), host='new', wall_seconds_total=20.0)
        observations = [dict(rows=[prefix]), dict(rows=[complete, dict(complete, nominal_step=3000)])]
        require_agreement(observations)
        complete['replay_diagnostics']['priority']['variance_sum'] += 1e-16
        with self.assertRaisesRegex(ValueError, 'disagree'):
            require_agreement(observations)
        require_agreement([dict(rows=[dict(nominal_step=1000, value=0.0)])])
        with self.assertRaises(ValueError):
            require_agreement([dict(rows=[dict(nominal_step=1000, value=0.0)]),
                               dict(rows=[dict(nominal_step=1000, value=-0.0)])])


if __name__ == '__main__':
    unittest.main()
