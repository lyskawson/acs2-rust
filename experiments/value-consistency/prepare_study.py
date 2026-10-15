import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from execution import APPROVED_SHA256, approved, study_scope
from protocol import ROOT


def prepare(output, destination, pilot_rows):
    output, destination, pilot_rows = Path(output).resolve(), Path(destination), Path(pilot_rows)
    if output.is_relative_to(ROOT) or output.exists() or not destination.is_absolute():
        raise ValueError('use a new study directory outside the checkout and an absolute cluster destination')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before preparing the study')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    registration = Path(__file__).with_name('plan-draft.json')
    plan = approved(registration)
    if hashlib.sha256(pilot_rows.read_bytes()).hexdigest() != plan['cost_pilot']['rows_sha256']:
        raise ValueError('pilot identity archive differs from the approved plan')
    output.mkdir(parents=True)
    shutil.copyfile(registration, output / 'approved-plan.json')
    shutil.copyfile(registration.with_name('approval.json'), output / 'approval.json')
    shutil.copyfile(pilot_rows, output / 'pilot-rows.jsonl')
    scope = study_scope(plan, destination)
    (output / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    files = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines()
    hashes = {}
    for name in files:
        target = output / 'operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    operations = dict(commit=commit, files=hashes)
    (output / 'operations/manifest.json').write_text(json.dumps(operations, indent=2) + '\n')
    print(json.dumps(dict(operations_commit=commit, plan_sha256=APPROVED_SHA256, batches=24, runs=480)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--pilot-rows', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.out, args.destination, args.pilot_rows)
