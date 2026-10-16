import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import math
import statistics
import unittest
from unittest.mock import patch

from analyze import analyze, analyze_maze, metric_summaries, pooled_bins
from validate_tables import validate


def evaluation(counts):
    return dict(evaluation_pairs=10, success=sum(counts)/len(counts)/10,
                heads=[dict(success=count/10) for count in counts])


def fixture():
    registered = json.loads(Path(__file__).with_name('plan-draft.json').read_text())
    config = dict(id='synthetic', targets=[100, 1000], cap=8,
                  registration=dict(blocks=dict(test=dict(evaluation_pairs=10))))
    plan = dict(agents=registered['agents'], seeds=list(range(42, 62)), analysis=dict(comparisons=registered['analysis']['comparisons'],
        thresholds=dict(synthetic=dict(acs2_final='1/2', acs2er_final='1/2')),
        baseline_threshold_crossing_checks=dict(synthetic=dict(acs2_final=dict(acs2=1000, acs2er=1000),
                                                               acs2er_final=dict(acs2=1000, acs2er=1000)))))
    patterns = dict(acs2vcp_k2=[[4, 6], [8, 10]], acs2vcp_k4=[[4, 6, 4, 6], [8, 10, 8, 10]],
                    acs2vcp_k2_uniform=[[2, 4], [6, 8]], acs2vcp_k4_uniform=[[2, 4, 2, 4], [6, 8, 6, 8]],
                    acs2her=[[4], [8]], acs2er_episodic=[[3], [7]], acs2=[[0], [10]], acs2er=[[0], [10]])
    datasets = {name: [dict(seed=seed, nominal_step=point, knowledge=(seed-42)/20,
                           evaluation_blocks=dict(test=evaluation(counts)),
                           replay_diagnostics=dict(priority=dict(proposal_variance=None)))
                       for seed in range(42, 62) for point, counts in zip(config['targets'], values)]
                for name, values in patterns.items()}
    for row in datasets['acs2']:
        row['replay_diagnostics'] = None
    return plan, config, datasets


class ArtifactAnalysisTests(unittest.TestCase):
    def test_exported_tables_retain_both_views_exact_scores_and_separate_run_costs(self):
        plan, config, datasets = fixture()
        plan['configurations'] = [config]
        plan['analysis']['thresholds']['synthetic']['acs2_final'] = '1'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = {}
            for name, rows in datasets.items():
                for seed in range(42, 62):
                    path = root / (name + str(seed))
                    path.mkdir()
                    selected = copy.deepcopy([row for row in rows if row['seed'] == seed])
                    for row in selected:
                        row.update(actual_steps=row['nominal_step'], wall_seconds_train=8., wall_seconds_eval=1., wall_seconds_total=9.)
                        if name in plan['agents']:
                            block = row['evaluation_blocks']['test']
                            for head in block['heads']:
                                head['success'] -= (seed % 2)/10
                            block['success'] = sum(head['success'] for head in block['heads']) / len(block['heads'])
                            row['replay_diagnostics']['priority']['value_queries'] = 0 if name.endswith('_uniform') else 10
                        row['evaluation_blocks'].update(training=copy.deepcopy(row['evaluation_blocks']['test']),
                                                         validation=copy.deepcopy(row['evaluation_blocks']['test']))
                    (path / 'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
                    (path / 'completion.json').write_text(json.dumps(dict(user_cpu_seconds=9., system_cpu_seconds=1.,
                                                                          elapsed_seconds=12., peak_rss_kib=3000)))
                    (path / 'resources.jsonl').write_text(''.join(json.dumps(dict(train_cpu_seconds_estimate=value))+'\n' for value in [2., 5.]))
                    entries[config['id'], name, seed] = dict(path=path, allocation_cpu_seconds=15)
            current = {key: value for key, value in entries.items() if key[1] in plan['agents']}
            references = {key: value for key, value in entries.items() if key not in current}
            verification = dict(pilot_identity_rows=180, runs=80, grid_allocated_cpu_seconds=1200,
                                pilot_allocated_cpu_seconds=16041, allocated_cpu_seconds=17241, batches={})
            with patch('analyze.verify_study', return_value=(current, verification)):
                with patch('analyze.reused', return_value=references), patch('analyze.approved', return_value=plan):
                    destination = root / 'analysis'
                    analyze(root, root, root, destination)
            def table(name):
                with (destination / (name+'.csv')).open() as stream:
                    return list(csv.DictReader(stream))
            self.assertEqual(len(table('primary')), 16)
            self.assertEqual(len(table('comparisons')), 18)
            self.assertEqual({row['view'] for row in table('comparisons')}, {'checkpoint_mean_within_seed', 'final_point'})
            self.assertEqual(len(table('seed-scores')), 320)
            self.assertEqual(len(table('checkpoint-scores')), 320)
            self.assertEqual(len(table('thresholds')), 16)
            self.assertEqual(len(table('learning-curves')), 48)
            self.assertEqual(len(table('run-costs')), 160)
            self.assertEqual(sum(row['allocation_scope'] == 'new_selected_run' for row in table('run-costs')), 80)
            self.assertEqual(sum(row['allocation_scope'] == 'reused_reference' for row in table('run-costs')), 80)
            self.assertEqual(float(table('run-costs')[0]['process_cpu_seconds']), 10.)
            self.assertEqual(float(table('run-costs')[0]['allocation_cpu_seconds']), 15.)
            self.assertEqual(float(table('run-costs')[0]['training_cpu_seconds_estimate']), 7.)
            self.assertTrue((destination / 'files.sha256.json').exists())
            checked = validate(destination, plan)
            self.assertEqual(checked['checkpoints'], 320)
            self.assertEqual(checked['comparisons'], 18)
            def mutate(name, change, message):
                path = destination / (name+'.csv')
                original = path.read_bytes()
                rows = table(name)
                change(rows)
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                manifest = destination / 'files.sha256.json'
                hashes = json.loads(manifest.read_text())
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(hashes))
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        validate(destination, plan)
                finally:
                    path.write_bytes(original)
                    hashes[path.name] = hashlib.sha256(original).hexdigest()
                    manifest.write_text(json.dumps(hashes))
            def unpaired(rows):
                row = rows[0]
                scores = table('seed-scores')
                values = [[float(item['score']) for item in scores if item['arm'] == row[side] and item['view'] == row['view']]
                          for side in ['left', 'right']]
                error = math.sqrt(sum(statistics.variance(group)/20 for group in values))
                row.update(se=str(error), ci_low=str(float(row['mean'])-2.093024054408263*error),
                           ci_high=str(float(row['mean'])+2.093024054408263*error))
            mutate('comparisons', unpaired, 'statistic differs: se')
            def strict_threshold(rows):
                row = next(item for item in rows if item['arm'] == 'acs2' and item['threshold'] == 'acs2_final')
                row['first_registered_nominal_step'] = ''
            mutate('thresholds', strict_threshold, 'threshold crossing differs')
            def charge_reference(rows):
                next(row for row in rows if row['arm'] == 'acs2her')['allocation_scope'] = 'new_selected_run'
            mutate('run-costs', charge_reference, 'reference cost was charged')

    def test_all_nine_comparisons_have_both_views_and_twenty_seed_units(self):
        plan, config, datasets = fixture()
        result, scores = analyze_maze(datasets, config, plan)
        self.assertEqual(len(result['comparisons']), 9)
        self.assertEqual(result['arms']['acs2vcp_k2']['thresholds']['acs2_final'], 100)
        self.assertAlmostEqual(float(scores['acs2vcp_k2'][42]['checkpoint_mean_within_seed']), .7)
        for comparison in result['comparisons']:
            self.assertEqual(set(comparison['views']), {'checkpoint_mean_within_seed', 'final_point'})
            for value in comparison['views'].values():
                self.assertEqual(value['n'], 20)
        for comparison in result['comparisons'][:2]:
            for value in comparison['views'].values():
                self.assertAlmostEqual(value['mean'], .2)
        plan['analysis']['baseline_threshold_crossing_checks']['synthetic']['acs2_final']['acs2'] = 100
        with self.assertRaisesRegex(ValueError, 'baseline threshold'):
            analyze_maze(datasets, config, plan)

    def test_diagnostics_keep_unmeasured_variances_absent_and_reject_missing_cells(self):
        _, config, datasets = fixture()
        rows = datasets['acs2vcp_k2_uniform']
        summaries = metric_summaries(rows, config['targets'])
        for values in summaries.values():
            self.assertNotIn('replay_diagnostics.priority.proposal_variance', values)
            self.assertAlmostEqual(values['knowledge']['mean'], .475)
            self.assertEqual(values['knowledge']['n'], 20)
        with self.assertRaisesRegex(ValueError, 'missing or extra'):
            metric_summaries(rows[:-1], config['targets'])
        changed = copy.deepcopy(rows)
        changed[0]['replay_diagnostics']['priority']['proposal_variance'] = dict(mean=0)
        with self.assertRaisesRegex(ValueError, 'diagnostic fields differ'):
            metric_summaries(changed, config['targets'])

    def test_distributions_pool_raw_counts_and_keep_exact_zero_separate(self):
        value = dict(count=4, histogram=[3] + [0]*14 + [1], logarithmic=dict(exact_zero_count=2,
                     positive_bin_upper_bounds=[1., 250000.], positive_bin_counts=[1, 1]))
        rows = [dict(nominal_step=100, replay_diagnostics=dict(priority=dict(proposal_variance=value,
                     accepted_variance=None)), evaluation_blocks=dict(test=dict(head_disagreement=value))) for _ in range(2)]
        bins = pooled_bins(rows)
        self.assertEqual({row['scope'] for row in bins}, {'training.proposal_variance', 'evaluation.test'})
        for scope in {row['scope'] for row in bins}:
            zeros = next(row for row in bins if row['scope'] == scope and row['bin'] == 'zero')
            self.assertEqual((zeros['count'], zeros['total'], zeros['share']), (4, 8, .5))
            for kind in ['logarithmic', 'equal_width']:
                self.assertEqual(sum(row['count'] for row in bins if row['scope'] == scope and row['bins'] == kind), 8)

    def test_analysis_refuses_before_reading_references_or_creating_output_when_identities_are_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'analysis'
            with patch('analyze.verify_study', return_value=({}, dict(pilot_identity_rows=179))):
                with patch('analyze.reused') as references:
                    with self.assertRaisesRegex(ValueError, '180 pilot'):
                        analyze(Path(temporary), 'her', 'baseline', destination)
                    references.assert_not_called()
                    self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
