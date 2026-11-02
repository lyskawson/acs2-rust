import argparse
import json
from pathlib import Path
import subprocess
import sys

from protocol import ROOT


def check(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    analysis = ROOT / 'experiments/per-pair-diagnostics/analysis.py'
    comparisons = analysis.with_name('comparisons.py')
    schema = ROOT / 'experiments/grid/pair_schema.py'
    mutations = [
        ('A1', analysis, "q_star=truth['action_values'][record['first_action']]", "q_star=truth['value']",
         'test_analysis.AnalysisTests.test_attach_uses_suboptimal_first_action_value_instead_of_state_value'),
        ('A2', analysis, 'C=ceiling(records, actions)', 'C=ceiling(eligible, actions)',
         'test_analysis.AnalysisTests.test_full_set_ceiling_and_random_reference_keep_goal_selective_fallback'),
        ('A3', analysis, 'statistics.stdev(values)', 'statistics.pstdev(values)',
         'test_analysis.AnalysisTests.test_nonzero_spread_uses_sample_deviation_and_t4_interval'),
        ('A4', analysis, 'T4 * se', '1.959963984540054 * se',
         'test_analysis.AnalysisTests.test_nonzero_spread_uses_sample_deviation_and_t4_interval'),
        ('A5', analysis, 'random_reference=weighted_mean(records,', 'random_reference=weighted_mean(eligible,',
         'test_analysis.AnalysisTests.test_full_set_ceiling_and_random_reference_keep_goal_selective_fallback'),
        ('A6', comparisons, "metrics[maze, arm, seed, point, 'pooled']", "metrics[maze, arm, seed, 30000 if point == 100000 else point, 'pooled']",
         'test_comparisons.ComparisonTests.test_registered_views_use_exact_points_paired_differences_and_common_all_pairs'),
        ('A7', analysis, 'return [record for records in blocks.values() for record in records]', "return list(blocks['test'])",
         'test_analysis.AnalysisTests.test_primary_pool_concatenates_three_blocks_without_changing_record_weights'),
        ('A8', schema, "if len({winner['selection_score'] for winner in decision['winners']}) > 1:", 'if False:',
         'test_schema_identity.SchemaIdentityTests.test_different_winner_selection_scores_are_rejected_even_with_valid_individual_scores'),
    ]
    evidence = []
    for name, path, old, new, test in mutations:
        original = path.read_bytes()
        if old not in original.decode():
            raise ValueError('mutation target changed: ' + name)
        try:
            path.write_text(original.decode().replace(old, new))
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest', test], cwd=analysis.parent,
                                    capture_output=True, text=True, timeout=60)
            log = result.stdout + result.stderr
            (output / (name + '.log')).write_text(log)
            if result.returncode == 0 or 'FAIL:' not in log:
                raise ValueError('mutation survived or failed without an assertion: ' + name)
            evidence.append(dict(mutation=name, rejected=True, test=test, returncode=result.returncode))
            print(json.dumps(evidence[-1]), flush=True)
        finally:
            path.write_bytes(original)
    (output / 'results.json').write_text(json.dumps(dict(applied=8, rejected=8, mutations=evidence), indent=2) + '\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    check(parser.parse_args().out)
