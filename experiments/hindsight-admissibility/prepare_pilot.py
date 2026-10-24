import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from execution import measurement_identity
from protocol import ROOT, pilot
from grid_protocol import atomic_json


def prepare(output, destination):
    output, destination = Path(output).resolve(), Path(destination)
    if any((parent / '.git').exists() for parent in (output, *output.parents)):
        raise ValueError('pilot artifacts must be outside a checkout')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('operations source must be clean')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    operations = output / 'operations'
    hashes = {}
    for name in subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines():
        source, target = ROOT / name, operations / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(source.stat().st_mode & 0o777)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(operations / 'manifest.json', dict(commit=commit, files=hashes))
    registration = dict(manifest=pilot() | measurement_identity(), authorization='Stage-1 user instruction, at most 2 allocated CPU-hours; no grid approval.')
    atomic_json(output / 'pilot-registration.json', registration)
    digest = hashlib.sha256((output / 'pilot-registration.json').read_bytes()).hexdigest()
    scope = dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0,
                 accounting_since='2026-10-24', batch_roots=[str(destination / 'batch')])
    atomic_json(output / 'scope.json', scope)
    manifest = registration['manifest'] | dict(authorization_plan=str(destination / 'pilot-registration.json'),
        authorization_sha256=digest, budget_scope=str(destination / 'scope.json'),
        budget_scope_sha256=hashlib.sha256((output / 'scope.json').read_bytes()).hexdigest(),
        stop_path=str(destination / 'STOP.json'), job_name='tu-her-admissibility-pilot', operations_commit=commit,
        operations_manifest=str(destination / 'operations/manifest.json'),
        operations_manifest_sha256=hashlib.sha256((operations / 'manifest.json').read_bytes()).hexdigest())
    atomic_json(output / 'pilot-manifest.json', manifest)
    return dict(operations_commit=commit, registration_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination)))
