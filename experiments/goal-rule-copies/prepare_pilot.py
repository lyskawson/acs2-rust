import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from protocol import ARM, ROOT, configurations, pilot
from execution import identity, scope
from collect import outside_repository
from grid_protocol import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, destination, references):
    output, destination, references = outside_repository(output), Path(destination), Path(references)
    if output.exists() or not destination.is_absolute():
        raise ValueError('use a new external directory and absolute cluster root')
    git = ['/usr/bin/git', '-C', str(ROOT)]
    if subprocess.check_output(git + ['status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before preparation')
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True)
    files = {}
    for name in subprocess.check_output(git + ['ls-files', 'experiments'], text=True).splitlines():
        target = output / 'operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        files[name] = digest(target)
    atomic_json(output / 'operations/manifest.json', dict(commit=commit, files=files))
    shutil.copytree(references, output / 'references')
    index = json.loads((references / 'reference-index.json').read_text())
    for entry in index:
        if digest(output / 'references' / entry['file']) != entry['sha256']:
            raise ValueError('frozen reference checksum differs')
    atomic_json(output / 'reference-index.json', index)
    models = []
    for config in configurations():
        content = subprocess.check_output([str(ROOT / 'target/release/acs2-maze-truth'), config['task'], str(config['cap'])])
        path = output / 'models' / (config['task'] + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        models.append(dict(task=config['task'], file=str(path.relative_to(output)), sha256=digest(path)))
    atomic_json(output / 'model-index.json', models)
    registration = dict(manifest=pilot(identity()), destination=str(destination),
        **{name + '_sha256': digest(output / (name + '.json')) for name in ('reference-index', 'model-index')},
        authorization='Only six seed42 runs, new goal-rule arm, all checkpoints and three blocks, per-pair records; two CPU-hours including all allocations and reservations. Stop after pilot and registered plan forecast.')
    atomic_json(output / 'pilot-registration.json', registration)
    sha = digest(output / 'pilot-registration.json')
    atomic_json(output / 'pilot-scope.json', scope(sha, destination))
    operations = destination / 'operations/manifest.json'
    manifest = dict(pilot(identity()), authorization_plan=str(destination / 'pilot-registration.json'), authorization_sha256=sha,
        budget_scope=str(destination / 'pilot-scope.json'), budget_scope_sha256=digest(output / 'pilot-scope.json'),
        stop_path=str(destination / 'STOP.json'), operations_commit=commit, operations_manifest=str(operations),
        operations_manifest_sha256=digest(output / 'operations/manifest.json'), job_name='tu-goal-rule-copies')
    atomic_json(output / 'pilot-manifest.json', manifest)
    return dict(pilot_registration_sha256=sha, runs=6, hard_reservations_cpu_hours=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--references', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.references)))
