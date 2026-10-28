import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from approved_execution import APPROVED_SHA256, approved
from execution import scope
from protocol import ROOT
from collect import outside_repository
from grid_protocol import atomic_json


def prepare(root, destination):
    root, destination = outside_repository(root), Path(destination)
    if not root.exists() or not destination.is_absolute():
        raise ValueError('use the verified pilot study and its absolute cluster root')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before preparing approved execution')
    plan = approved(Path(__file__).with_name('plan.json'))
    if hashlib.sha256((root / 'pilot-rows.jsonl').read_bytes()).hexdigest() != plan['cost_pilot']['rows_sha256']:
        raise ValueError('pilot identity rows differ from the approved archive')
    inventory = json.loads((root / 'checksums.json').read_text())
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in inventory['files'].items()):
        raise ValueError('a frozen pilot artifact changed')
    operations = root / 'execution-operations'
    operations.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__).with_name('plan.json'), root / 'approved-plan.json')
    shutil.copyfile(Path(__file__).with_name('approval.json'), root / 'approval.json')
    atomic_json(root / 'scope.json', scope(APPROVED_SHA256, destination))
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    hashes = {}
    for name in subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines():
        source, target = ROOT / name, operations / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(operations / 'manifest.json', dict(commit=commit, files=hashes))
    return dict(operations_commit=commit, plan_sha256=APPROVED_SHA256, pilot_cpu_seconds=4784,
                phase_cpu_hour_limit=45, scope_roots=13, pilot_artifacts_verified=len(inventory['files']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.destination)))
