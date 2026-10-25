import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from execution import APPROVED_SHA256, approved, batches, pilot_subset, study_scope
from protocol import ROOT
from collect import outside_repository
from grid_protocol import atomic_json


def prepare(output, destination, pilot_rows):
    output, destination, pilot_rows = outside_repository(output), Path(destination), Path(pilot_rows)
    if output.exists() or not destination.is_absolute():
        raise ValueError('use a new study directory and an absolute cluster destination')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before preparing the study')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    registration = Path(__file__).with_name('plan.json')
    plan = approved(registration)
    if hashlib.sha256(pilot_rows.read_bytes()).hexdigest() != plan['cost_pilot']['rows_sha256']:
        raise ValueError('pilot rows differ from the approved registration')
    output.mkdir(parents=True)
    shutil.copyfile(registration, output / 'approved-plan.json')
    shutil.copyfile(registration.with_name('approval.json'), output / 'approval.json')
    shutil.copyfile(pilot_rows, output / 'pilot-rows.jsonl')
    atomic_json(output / 'scope.json', study_scope(plan, destination))
    (output / 'pilot').mkdir()
    for config, agent in batches(plan):
        (output / 'pilot' / (config['id'] + '_' + agent + '.jsonl')).write_bytes(pilot_subset(plan, config, agent, output))
    hashes = {}
    for name in subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines():
        target = output / 'operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(output / 'operations/manifest.json', dict(commit=commit, files=hashes))
    return dict(operations_commit=commit, plan_sha256=APPROVED_SHA256, batches=12, runs=240)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--pilot-rows', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.pilot_rows)))
