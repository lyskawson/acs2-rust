import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from protocol import ROOT


def run(output, rust=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    directory = ROOT / 'experiments/per-pair-diagnostics'
    python = lambda name: [sys.executable, '-B', '-m', 'unittest', name]
    mutations = [
        ('ceiling-across-all-cells', ROOT / 'experiments/per-pair-diagnostics/analysis.py',
         "by_start[record['start']].append(record)", "by_start[0].append(record)",
         python('test_analysis.AnalysisTests.test_ceiling_is_per_start_and_keeps_tied_optima_and_weights'), directory),
        ('mode-from-optimal-action', ROOT / 'experiments/per-pair-diagnostics/analysis.py',
         "if record['first_action'] == action", "if action in record['optimal']",
         python('test_analysis.AnalysisTests.test_mode_is_the_agents_action_even_when_it_is_always_suboptimal'), directory),
        ('fallbacks-in-behavior', ROOT / 'experiments/per-pair-diagnostics/analysis.py',
         "eligible = [record for record in records if not record['random_fallback']]", "eligible = list(records)",
         python('test_analysis.AnalysisTests.test_random_fallbacks_are_excluded_from_all_three_behavior_denominators'), directory),
        ('identity-skips-knowledge', ROOT / 'experiments/grid/grid_identity.py',
         '    def compact(rows):\n', "    def compact(rows):\n        rows = [{key: value for key, value in row.items() if key != 'knowledge'} for row in rows]\n",
         python('test_schema_identity.SchemaIdentityTests.test_identity_strips_only_records_and_detects_every_mutated_numeric_field'), directory),
    ]
    if rust:
        test = lambda name: ['cargo', 'test', '--release', '-p', 'acs2-measure', '--test', 'per_pair_diagnostics', name]
        source = ROOT / 'acs2-measure/src/maze_truth.rs'
        mutations += [
            ('successor-without-discount', source,
             'GAMMA * discounted_value(topology.shortest_distance(next, goal), cap - 1)',
             'discounted_value(topology.shortest_distance(next, goal), cap - 1)',
             test('corridor_q_values_include_wall_bumps_successor_discount_and_remaining_cap'), ROOT),
            ('wall-bump-as-move', source, '.map(|action| topology.next_cell(start, action))',
             '.map(|action| { let next = topology.next_cell(start, action); if next == start { (start.0, start.1 + 1) } else { next } })',
             test('corridor_q_values_include_wall_bumps_successor_discount_and_remaining_cap'), ROOT),
            ('drop-tied-actions', source, '.filter_map(|(action, &value)| (value == best).then_some(action))',
             '.filter_map(|(action, &value)| (value == best).then_some(action)).take(1)',
             test('equally_short_routes_retain_every_optimal_action'), ROOT),
        ]
    evidence = []
    for name, path, old, new, command, cwd in mutations:
        original = path.read_bytes()
        if old not in original.decode():
            raise ValueError('mutation target changed: ' + name)
        try:
            path.write_text(original.decode().replace(old, new, 1))
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600)
            log = result.stdout + result.stderr
            (output / (name + '.log')).write_text(log)
            if result.returncode == 0 or ('FAIL:' not in log and 'test result: FAILED' not in log):
                raise ValueError('mutation did not cause an assertion failure: ' + name)
            evidence.append(dict(mutation=name, rejected=True, returncode=result.returncode,
                source=str(path.relative_to(ROOT)), restored_sha256=hashlib.sha256(original).hexdigest(),
                command=command, log=name + '.log'))
            print(json.dumps(evidence[-1]), flush=True)
        finally:
            path.write_bytes(original)
    (output / 'results.json').write_text(json.dumps(dict(applied=len(evidence), rejected=len(evidence), mutations=evidence), indent=2) + '\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--rust', action='store_true')
    args = parser.parse_args()
    run(args.out, args.rust)
