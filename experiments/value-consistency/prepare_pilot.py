import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from execution import pilot_measurement_identity
from protocol import PILOT_PARTS, ROOT, pilot

sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import atomic_json


def operations_snapshot(output, name='operations'):
    output = Path(output)
    if any(parent.joinpath('.git').exists() for parent in [output.resolve(), *output.resolve().parents]):
        raise ValueError('pilot artifacts must be outside a checkout')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('operations source must be clean')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    operations = output / name
    files = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines()
    hashes = {}
    for name in files:
        path = operations / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        path.chmod((ROOT / name).stat().st_mode & 0o777)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(operations / 'manifest.json', dict(commit=commit, files=hashes))
    return commit


def prepare(output, destination):
    output, destination = Path(output), Path(destination)
    commit = operations_snapshot(output)
    operations = output / 'operations'
    identity = pilot_measurement_identity()
    registration = dict(measurement=identity, pilot_parts=[pilot(part) for part in PILOT_PARTS],
        authorized_cpu_hours=5, second_temperature='all six mazes if admission fits; otherwise the registered reduced set, never both second-temperature parts',
        execution_order='complete temperature1-and-controls, then admit exactly one temperature24 part',
        authorization='stage-1 user instruction; no tuning or final-grid authorization')
    atomic_json(output / 'pilot-registration.json', registration)
    digest = hashlib.sha256((output / 'pilot-registration.json').read_bytes()).hexdigest()
    scope = dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-15',
                 batch_roots=[str(destination / part) for part in PILOT_PARTS])
    atomic_json(output / 'scope.json', scope)
    (output / 'manifests').mkdir()
    for part in PILOT_PARTS:
        manifest = pilot(part) | identity | dict(authorization_plan=str(destination / 'pilot-registration.json'),
            authorization_sha256=digest, budget_scope=str(destination / 'scope.json'),
            budget_scope_sha256=hashlib.sha256((output / 'scope.json').read_bytes()).hexdigest(),
            stop_path=str(destination / 'STOP.json'), job_name='tu-vcp-pilot', operations_commit=commit,
            operations_manifest=str(destination / 'operations/manifest.json'),
            operations_manifest_sha256=hashlib.sha256((operations / 'manifest.json').read_bytes()).hexdigest())
        atomic_json(output / 'manifests' / (part + '.json'), manifest)
    print(json.dumps(dict(output=str(output), operations_commit=commit, registration_sha256=digest)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    prepare(args.out, args.destination)
