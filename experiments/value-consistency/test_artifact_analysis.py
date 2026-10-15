import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze import analyze, analyze_maze, metric_summaries, pooled_bins


def evaluation(counts):
    return dict(evaluation_pairs=10, success=sum(counts)/len(counts)/10,
                heads=[dict(success=count/10) for count in counts])


def fixture():
    registered = json.loads(Path(__file__).with_name('plan-draft.json').read_text())
    config = dict(id='synthetic', targets=[100, 1000])
    plan = dict(agents=registered['agents'], analysis=dict(comparisons=registered['analysis']['comparisons'],
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
    return plan, config, datasets


class ArtifactAnalysisTests(unittest.TestCase):
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
