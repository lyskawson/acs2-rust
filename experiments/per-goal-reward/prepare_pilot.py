import argparse
import json
from pathlib import Path
import shutil
import subprocess

from protocol import ROOT, configurations, pilot
from execution import identity, scope
from freeze_references import checksum
from collect import outside_repository
from grid_protocol import atomic_json
from analysis import truth_index


def prepare(output, destination, references, models):
    output, destination, references, models = outside_repository(output), Path(destination), Path(references), Path(models)
    if output.exists() or not destination.is_absolute():
        raise ValueError('use a new external pilot directory and an absolute cluster destination')
    git = ['/usr/bin/git', '-C', str(ROOT)]
    if subprocess.check_output(git + ['status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before pilot preparation')
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True)
    files = {}
    for name in subprocess.check_output(git + ['ls-files', 'experiments', 'fixtures'], text=True).splitlines():
        target = output / 'operations' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        files[name] = checksum(target)
    atomic_json(output / 'operations/manifest.json', dict(commit=commit, files=files))
    shutil.copytree(references, output / 'references')
    index = json.loads((references / 'reference-index.json').read_text())
    for entry in index:
        if checksum(output / 'references' / entry['file']) != entry['sha256']:
            raise ValueError('pilot frozen reference checksum differs')
    atomic_json(output / 'reference-index.json', index)
    model_index = []
    for entry in json.loads((models / 'model-index.json').read_text()):
        source = models / entry['file']
        if checksum(source) != entry['sha256']:
            raise ValueError('archived environment model changed')
        value = json.loads(source.read_text())
        truth_index(value)
        config = next(config for config in configurations() if config['task'] == entry['task'])
        if value['cap'] != config['cap'] or value['task'] != config['task']:
            raise ValueError('environment model does not use the registered maze and cap')
        target = output / 'models' / (entry['task'] + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        model_index.append(dict(task=entry['task'], file=str(target.relative_to(output)), sha256=checksum(target)))
    if {entry['task'] for entry in model_index} != {config['task'] for config in configurations()}:
        raise ValueError('pilot requires all six environment models')
    atomic_json(output / 'model-index.json', model_index)
    registration = dict(manifest=pilot(identity()), destination=str(destination),
        reference_index_sha256=checksum(output / 'reference-index.json'), model_index_sha256=checksum(output / 'model-index.json'),
        authorization='Only twelve seed42 runs: two per-goal reward arms, six mazes, every checkpoint, three blocks and per-pair records. Four CPU-hours including all reservations. Stage1 ends with the plan forecast and report.')
    atomic_json(output / 'pilot-registration.json', registration)
    digest = checksum(output / 'pilot-registration.json')
    atomic_json(output / 'pilot-scope.json', scope(digest, destination))
    manifest = dict(pilot(identity()), authorization_plan=str(destination / 'pilot-registration.json'),
        authorization_sha256=digest, budget_scope=str(destination / 'pilot-scope.json'),
        budget_scope_sha256=checksum(output / 'pilot-scope.json'), stop_path=str(destination / 'STOP.json'),
        operations_commit=commit, operations_manifest=str(destination / 'operations/manifest.json'),
        operations_manifest_sha256=checksum(output / 'operations/manifest.json'), job_name='tu-per-goal-reward')
    atomic_json(output / 'pilot-manifest.json', manifest)
    return dict(pilot_registration_sha256=digest, runs=12, hard_reservations_cpu_hours=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('out', 'destination', 'references', 'models'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.references, args.models)))
