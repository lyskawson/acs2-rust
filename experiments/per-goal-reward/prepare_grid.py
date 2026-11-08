import argparse
import json
from pathlib import Path
import shutil
import subprocess

from protocol import ROOT
from execution_grid import approved, scope, validate_inputs, checksum, APPROVED_SHA256
from collect import outside_repository
from grid_protocol import atomic_json
from verify_copies import inventory


def prepare(root, destination):
    root, destination = outside_repository(root), Path(destination)
    plan, approval = approved()
    if not destination.is_absolute() or (root / 'continuation').exists() or (root / 'execution-operations').exists():
        raise ValueError('stage2 is already prepared or its destination is not absolute')
    git = ['/usr/bin/git', '-C', str(ROOT)]
    if subprocess.check_output(git + ['status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean stage2 operations before preparation')
    validate_inputs(root, plan)
    if (root / 'registered-plan.json').read_bytes() != Path(__file__).with_name('plan.json').read_bytes():
        raise ValueError('registered plan bytes changed')
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    preserved = inventory(root)
    atomic_json(root / 'pilot-inventory.json', preserved)
    for source, target in [('plan.json', 'approved-plan.json'), ('approval.json', 'approval.json')]:
        shutil.copyfile(Path(__file__).with_name(source), root / target)
    (root / 'continuation').mkdir()
    atomic_json(root / 'continuation/scope.json', scope(destination, plan))
    files = {}
    for name in subprocess.check_output(git + ['ls-files', 'experiments', 'fixtures'], text=True).splitlines():
        target = root / 'execution-operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        files[name] = checksum(target)
    atomic_json(root / 'execution-operations/manifest.json', dict(commit=commit, files=files))
    result = dict(plan_sha256=APPROVED_SHA256, phase_cpu_hour_limit=29, pilot_cpu_seconds=1648,
        operations_commit=commit, pilot_inventory_sha256=checksum(root / 'pilot-inventory.json'),
        measurement_commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'], measurement_rebuilt=False)
    atomic_json(root / 'execution-preparation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.destination)))
