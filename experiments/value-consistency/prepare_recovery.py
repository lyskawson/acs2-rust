import argparse
import hashlib
import json
from pathlib import Path

from execution import measurement_identity
from prepare_pilot import operations_snapshot
from protocol import RECOVERY_PARTS, SECOND_TEMPERATURE_ORDER, recovery
from grid_protocol import atomic_json


def prepare_recovery(output, destination, initial_registration, cancellation):
    output, destination = Path(output), Path(destination)
    initial = json.loads(Path(initial_registration).read_text())
    cancelled = json.loads(Path(cancellation).read_text())
    identity = measurement_identity()
    if initial['measurement'] != identity or initial['authorized_cpu_hours'] != 5:
        raise ValueError('initial pilot has a different identity or limit')
    if len(cancelled['verified_pending_jobs']) != 20 or len(cancelled['authorized_arrays']) != 5:
        raise ValueError('recovery needs the authorized twenty-task cancellation record')
    commit = operations_snapshot(output, 'recovery-operations')
    registration = dict(measurement=identity, authorized_cpu_hours=5,
        original_registration_sha256=hashlib.sha256(Path(initial_registration).read_bytes()).hexdigest(),
        cancellation_sha256=hashlib.sha256(Path(cancellation).read_bytes()).hexdigest(),
        pilot_parts=[recovery(part) for part in RECOVERY_PARTS],
        reuse='Keep the two completed Maze4 K=2 runs; retain and count every failed and cancelled initial allocation.',
        authorization='Explicit user approval to cancel twenty pending tasks and complete the cost pilot in smaller batches within five allocated CPU-hours.',
        second_temperature_order=SECOND_TEMPERATURE_ORDER,
        execution_order='Complete T=1 and both controls on all six mazes, then T=24 on both K values in the listed maze order. Admit each maze only if both hard limits fit. If the full second temperature cannot fit, stop at a nonempty prefix and report its exact maze set.',
        failure_rule='Any new node failure stops the recovery; preserve its STOP and account for it before any further recovery.',
        stage2_authorized=False)
    atomic_json(output / 'recovery-registration.json', registration)
    digest = hashlib.sha256((output / 'recovery-registration.json').read_bytes()).hexdigest()
    scope = dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-15',
                 batch_roots=[str(destination / part) for part in ['temperature1-and-controls', *RECOVERY_PARTS]])
    atomic_json(output / 'recovery-scope.json', scope)
    (output / 'recovery-manifests').mkdir()
    for part in RECOVERY_PARTS:
        manifest = recovery(part) | identity | dict(authorization_plan=str(destination / 'recovery-registration.json'),
            authorization_sha256=digest, budget_scope=str(destination / 'recovery-scope.json'),
            budget_scope_sha256=hashlib.sha256((output / 'recovery-scope.json').read_bytes()).hexdigest(),
            stop_path=str(destination / 'RECOVERY-STOP.json'), job_name='tu-vcp-pilot',
            operations_commit=commit, operations_manifest=str(destination / 'recovery-operations/manifest.json'),
            operations_manifest_sha256=hashlib.sha256((output / 'recovery-operations/manifest.json').read_bytes()).hexdigest())
        atomic_json(output / 'recovery-manifests' / (part + '.json'), manifest)
    return dict(output=str(output), operations_commit=commit, registration_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--initial-registration', required=True, type=Path)
    parser.add_argument('--cancellation', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_recovery(args.out, args.destination, args.initial_registration, args.cancellation)))
