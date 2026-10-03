import copy
import json
import math
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

from analyze_grid import COST_FIELDS, analyze, analyze_values
from grid_protocol import PLAN_SHA256, accounting, allocation_rows, compare_pilot, load_plan, require_budget, seconds, summarize, verdict


class GridTests(unittest.TestCase):
    def test_paired_scores_use_equal_point_weights_and_seed_pairing(self):
        seeds = list(range(42, 62))
        config = {'targets': [1, 10, 10000]}
        series = {(agent, seed): [seed / 100, seed / 100 + 0.1, seed / 100 + 0.2 + (agent == 'acs2er') * 0.3]
                  for agent in ('acs2', 'acs2er') for seed in seeds}
        result = analyze_values(series, config, seeds, 0, 1)
        self.assertAlmostEqual(result['seed_scores'][0]['acs2'], 0.52)
        self.assertAlmostEqual(result['paired_score_difference']['mean'], 0.1)
        self.assertAlmostEqual(result['paired_score_difference']['se'], 0)
        self.assertEqual(len(result['points']), 6)
        del series['acs2', 42]
        with self.assertRaises(ValueError):
            analyze_values(series, config, seeds, 0, 1)

    def test_t_interval_uses_sample_variance_and_is_not_truncated(self):
        result = summarize(list(range(20)))
        self.assertEqual(result['mean'], 9.5)
        self.assertAlmostEqual(result['se'], math.sqrt(35 / 20))
        self.assertAlmostEqual(result['ci_high'] - 9.5, 2.093024054408263 * math.sqrt(35 / 20))
        self.assertLess(summarize([0] * 19 + [1])['ci_low'], 0)
        for values in ([1] * 19, [float('nan')] * 20):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_verdict_boundary_and_priority(self):
        self.assertEqual(verdict({'ci_low': 0.1}, {'ci_low': 0.1}, {'mean': 0.99}, 0.1, 1), 'floor_baselines_retain_for_successors')
        self.assertEqual(verdict({'ci_low': 0.1}, {'ci_low': 0.2}, {'mean': 0.9}, 0.1, 1), 'too_easy_no_speed_ranking')
        self.assertEqual(verdict({'ci_low': 0.1}, {'ci_low': 0.2}, {'mean': 0.89}, 0.1, 1), 'compare')

    def test_pilot_comparison_excludes_only_hardware_and_timing(self):
        pilot = [{'nominal_step': 1, 'success': 0.5, 'preset': {'epsilon': 0.8}, 'host': 'a', 'wall_seconds_train': 5}]
        grid = copy.deepcopy(pilot)
        grid[0].update(host='b', wall_seconds_train=7)
        self.assertEqual(compare_pilot(grid, pilot), 3)
        grid[0]['preset']['epsilon'] = 0.7
        with self.assertRaises(ValueError):
            compare_pilot(grid, pilot)
        with self.assertRaises(ValueError):
            compare_pilot([], pilot)

    def test_accounting_counts_allocations_once_and_reserves_pending_caps(self):
        raw = 'JobID|JobIDRaw|State|CPUTimeRAW|\n12_0|99|COMPLETED|60|\n12_0.batch|99.batch|COMPLETED|60|\n12_0.0|99.0|COMPLETED|58|\n12_1|100|RUNNING|20|\n12_[2-3%1]|12_[2-3%1]|PENDING|0|\n'
        records = allocation_rows(raw)
        self.assertEqual(len(records), 2)
        ledger = [{'array_id': 12, 'runs': [{'limit_seconds': 100}] * 4}]
        state = accounting(ledger, records)
        self.assertEqual(state, {'spent_seconds': 80, 'reserved_seconds': 280})
        self.assertEqual(require_budget(7905, state, 100), 8365)
        with self.assertRaises(ValueError):
            require_budget(7905, state, 200 * 3600)
        with self.assertRaises(ValueError):
            accounting([{'array_id': None}], [])

    def test_slurm_time_units_and_plan_checksum(self):
        self.assertEqual(seconds('01:02:03'), 3723)
        self.assertEqual(seconds('1-00:01:00'), 86460)
        self.assertEqual(seconds('02:03.5'), 123.5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'plan.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                load_plan(path)

    def test_full_analysis_rejects_tail_missing_runs_and_changed_preset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'grid'
            pilot = Path(temporary) / 'pilot'
            config = dict(id='fixture', task='maze4', cap=5, pool='full', row_pool='full', encoding='coordinates', targets=[1000])
            plan = dict(configurations=[config], seeds=list(range(42, 62)), agents=['acs2', 'acs2er'],
                        measurement_commit='a' * 40, binary_sha256='b' * 64, cpu_model='cpu',
                        partition='lem-cpu-normal', first_row_timeout_seconds=60)
            ledger = []
            for array_id, agent in enumerate(plan['agents'], 1):
                runs = []
                for index, seed in enumerate(plan['seeds']):
                    path = root / 'batches' / '01-fixture' / 'runs' / f'fixture_{agent}_s{seed}'
                    path.mkdir(parents=True)
                    row = dict.fromkeys(COST_FIELDS, 0)
                    row.update(schema=2, commit='a' * 40, source_state='clean', task='maze4', cap=5,
                               agent=agent, seed=seed, nominal_step=1000, actual_steps=1000, starts=None,
                               goal_pool='full', goal_encoding='coordinates', evaluated_policy='greedy_change_anticipating_population',
                               host='node', cpu_model='cpu', preset={'replay_updates_per_step': 3},
                               online_updates=1000 if agent == 'acs2' else 0,
                               replay_updates=2997 if agent == 'acs2er' else 0,
                               random_floor=0.1, reachable_within_cap=1, success=0.5)
                    (path / 'rows.jsonl').write_text(json.dumps(row) + '\n')
                    if seed == 42:
                        reference = pilot / 'runs' / path.name
                        reference.mkdir(parents=True)
                        (reference / 'rows.jsonl').write_text(json.dumps(row) + '\n')
                    manifest = dict(agent=agent, seed=seed, configuration=config, plan_sha256=PLAN_SHA256)
                    launch = dict(manifest=manifest, binary_sha256='b' * 64, partition='lem-cpu-normal',
                                  array_job_id=array_id, array_task_id=index)
                    completion = dict(exit_code=0, failure=None, incomplete_final_line=False, rows_seen=1,
                                      first_row_seconds=0.1, user_cpu_seconds=1, system_cpu_seconds=0.1, peak_rss_kib=100)
                    resource = dict(nominal_step=1000, train_cpu_seconds_estimate=0.8, eval_cpu_seconds_estimate=0.2)
                    (path / 'launch.json').write_text(json.dumps(launch))
                    (path / 'completion.json').write_text(json.dumps(completion))
                    (path / 'resources.jsonl').write_text(json.dumps(resource) + '\n')
                    runs.append(dict(agent=agent, seed=seed, host='node'))
                ledger.append(dict(array_id=array_id, configuration_id='fixture', runs=runs))
            (root / 'ledger.json').write_text(json.dumps(ledger))
            with patch('analyze_grid.load_plan', return_value=plan):
                result = analyze(root, pilot, 'fixture')
                self.assertEqual(result['runs'], 40)
                self.assertEqual(result['pilot_identity']['matched_runs'], 2)
                original_dir = root / 'batches/01-fixture/runs/fixture_acs2_s42'
                duplicate = root / 'attempts/duplicate/measurement/fixture_acs2_s42'
                shutil.copytree(original_dir, duplicate)
                duplicate_launch = json.loads((duplicate / 'launch.json').read_text())
                duplicate_launch.update(array_job_id=99, array_task_id=0, started_unix=100)
                (duplicate / 'launch.json').write_text(json.dumps(duplicate_launch))
                extra = dict(configuration_id='fixture', array_id=99, runs=[dict(agent='acs2', seed=42,
                             attempt_id='duplicate', relative_directory=str(duplicate.relative_to(root)))])
                ledger.append(extra)
                (root / 'ledger.json').write_text(json.dumps(ledger))
                duplicate_row = json.loads((duplicate / 'rows.jsonl').read_text())
                duplicate_row['host'] = 'other-approved-model-host'
                (duplicate / 'rows.jsonl').write_text(json.dumps(duplicate_row) + '\n')
                with_duplicates = analyze(root, pilot, 'fixture')
                self.assertEqual(with_duplicates['points'], result['points'])
                self.assertEqual(len(with_duplicates['attempts']), 41)
                self.assertEqual(with_duplicates['pilot_identity']['matched_runs'], 2)
                duplicate_row['cpu_model'] = 'different CPU'
                (duplicate / 'rows.jsonl').write_text(json.dumps(duplicate_row) + '\n')
                with self.assertRaisesRegex(ValueError, 'invalid runs'):
                    analyze(root, pilot, 'fixture')
                duplicate_row['cpu_model'] = 'cpu'
                duplicate_row['success'] = 0.6
                (duplicate / 'rows.jsonl').write_text(json.dumps(duplicate_row) + '\n')
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    analyze(root, pilot, 'fixture')
                ledger.pop()
                (root / 'ledger.json').write_text(json.dumps(ledger))
                with self.assertRaisesRegex(ValueError, 'unregistered'):
                    analyze(root, pilot, 'fixture')
                shutil.rmtree(root / 'attempts')
                path = root / 'batches/01-fixture/runs/fixture_acs2_s42/rows.jsonl'
                original = path.read_bytes()
                path.write_bytes(original + b'{"partial":')
                with self.assertRaises(ValueError):
                    analyze(root, pilot, 'fixture')
                self.assertEqual(path.read_bytes(), original + b'{"partial":')
                changed = json.loads(original)
                changed['preset']['epsilon'] = 0.7
                path.write_text(json.dumps(changed) + '\n')
                with self.assertRaises(ValueError):
                    analyze(root, pilot, 'fixture')
                path.write_bytes(original)
                hidden = path.parent.with_name('unexpected_run')
                path.parent.rename(hidden)
                with self.assertRaises(ValueError):
                    analyze(root, pilot, 'fixture')


if __name__ == '__main__':
    unittest.main()
