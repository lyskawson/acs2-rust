import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from archive import audit
from execution import measurement_identity, pilot_measurement_identity
from pilot_equivalence import binary64
from protocol import ROOT, configurations, draft

PILOT_COMMIT = '79648424076144c77a3a87acf87418bb61aa7c42'
REVIEW_COMMIT = '8ddacb8dd658a5cdfd0b72fc4799fc46b1773003'


def prepare(pilot_root, calibration_path, output):
    pilot_root, calibration_path, output = Path(pilot_root), Path(calibration_path), Path(output)
    summary, rows, _ = audit(pilot_root)
    archived_rows = [json.loads(line) for line in (pilot_root / 'pilot-rows.jsonl').read_text().splitlines()]
    if binary64(rows) != binary64(archived_rows):
        raise ValueError('pilot row summary differs from the audited raw trajectories')
    calibration = json.loads(calibration_path.read_text())
    pilot_digest = hashlib.sha256((pilot_root / 'pilot-rows.jsonl').read_bytes()).hexdigest()
    if calibration['pilot_rows_sha256'] != pilot_digest:
        raise ValueError('control calibration uses a different pilot archive')
    for name, digest in calibration['files'].items():
        path = calibration_path.parent / name
        if path.parent.resolve() != calibration_path.parent.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('control calibration rows changed')
    measured_rows = {}
    for row in rows:
        key = row['task'], row['agent'], row['agent_parameters']['temperature']
        measured_rows.setdefault(key, []).append(row)
    runs = []
    tasks = {config['id']: config['task'] for config in configurations()}
    for item in summary['selected_runs']:
        cost = dict(item['cost'])
        task_rows = measured_rows[tasks[item['configuration_id']], item['agent'], item['temperature']]
        final = max(task_rows, key=lambda row: row['nominal_step'])
        cost.update(training_wall_seconds=final['wall_seconds_train'],
                    classifier_perception_tests=final['classifier_perception_tests_train'],
                    pilot_population_sum_over_points=sum(row['population_classifiers'] for row in task_rows))
        runs.append(dict(item, cost=cost))
    measurement = measurement_identity()
    if calibration['measurement_commit'] != measurement['measurement_commit']:
        raise ValueError('control calibration and frozen measurement source differ')
    plan = draft(runs, rows, calibration, summary['allocation_cpu_seconds'], measurement)
    def changed(base):
        return subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'diff', '--name-status', base,
                                        measurement['measurement_commit'], '--', 'acs2-agents', 'acs2-measure', 'baseline', 'fixtures'], text=True).splitlines()
    plan['measurement_evidence'] = dict(pilot_measurement=pilot_measurement_identity(),
        source_diff_from_pilot=changed(PILOT_COMMIT), source_diff_from_review=changed(REVIEW_COMMIT),
        validity='The priority distribution, populations, update steps, acting/evaluation streams and all old learning/evaluation/knowledge fields remain pinned. T=0 skips only unused training value queries; the separate observer and immutable evaluation diagnostics consume no learner randomness. Header-preserving measurement sidecars account for their additional memory separately.',
        local_native_calibration_binary_sha256=calibration['binary_sha256'],
        calibration_source_commit=calibration['measurement_commit'])
    plan['cost_pilot'] = dict(runs=summary['runs'], rows=summary['rows'], allocations=summary['allocations'],
        allocation_cpu_seconds=summary['allocation_cpu_seconds'], second_temperature_mazes=summary['second_temperature_mazes'],
        summary_sha256=hashlib.sha256((pilot_root / 'pilot-summary.json').read_bytes()).hexdigest(), rows_sha256=pilot_digest,
        control_work_calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        generator_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['protocol.py', 'prepare_draft.py', 'pilot_equivalence.py', 'calibrate_controls.py']})
    content = (json.dumps(plan, indent=2) + '\n').encode()
    output.write_bytes(content)
    return dict(plan_sha256=hashlib.sha256(content).hexdigest(), forecast=plan['forecast'],
                requested_cpu_hour_limit=plan['requested_cpu_hour_limit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-root', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.pilot_root, args.calibration, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != 'forecast'}))
