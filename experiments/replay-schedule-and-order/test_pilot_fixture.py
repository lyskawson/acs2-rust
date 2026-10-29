import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import ER, SCHEDULES, SHORTENED
from grid_identity import field_bits
from verify_pilot_fixture import verify


class PilotFixtureTests(unittest.TestCase):
    def test_binary_gate_checks_every_schedule_row_bit_and_refuses_a_changed_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / 'binary'
            binary.write_bytes(b'frozen binary')
            rows = [dict(nominal_step=point, value=0.5) for point in (1000, 3000)]
            fixture = root / 'fixture.json'
            configs = [dict(task='maze4', cap=5, agent=agent, mode=mode, proportion=0.5,
                           rows=field_bits(rows)) for agent in (SHORTENED, ER) for mode in SCHEDULES]
            fixture.write_text(json.dumps(dict(configurations=configs, excluded=[])))
            commands = []
            def measure(command, **_):
                commands.append(command)
                Path(command[command.index('--out') + 1]).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with patch('verify_pilot_fixture.subprocess.run', side_effect=measure):
                result = verify(binary, fixture, 'maze4', root / 'complete')
                self.assertEqual((result['settings'], result['rows'], result['verified']), (6, 12, True))
                self.assertEqual(sum('--relabel-proportion' in command for command in commands), 3)
                rows[0]['value'] = 0.5000000000000001
                with self.assertRaisesRegex(ValueError, 'differs from the original schedule fixture'):
                    verify(binary, fixture, 'maze4', root / 'changed')


if __name__ == '__main__':
    unittest.main()
