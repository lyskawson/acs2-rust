import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from protocol import ROOT, configurations
from references import catalog_sources, freeze
from execution import pilot_manifest, scope
from collect import outside_repository
from grid_protocol import atomic_json


def prepare(output, destination, baseline, her, admissibility):
    output, destination = outside_repository(output), Path(destination)
    if output.exists() or not destination.is_absolute():
        raise ValueError('use a new external directory and absolute cluster root')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before pilot preparation')
    sources = catalog_sources(baseline, her, admissibility)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True)
    freeze(output, sources)
    operations = output / 'operations'
    hashes = {}
    for name in subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines():
        target = operations / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(operations / 'manifest.json', dict(commit=commit, files=hashes))
    model_index = []
    for config in configurations():
        content = subprocess.check_output([str(ROOT / 'target/release/acs2-maze-truth'), config['task'], str(config['cap'])])
        path = output / 'models' / (config['task'] + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        model_index.append(dict(task=config['task'], file=str(path.relative_to(output)), sha256=hashlib.sha256(content).hexdigest()))
    atomic_json(output / 'model-index.json', model_index)
    manifest = pilot_manifest(output, destination)
    atomic_json(output / 'pilot-registration.json', dict(manifest=manifest, destination=str(destination),
        reference_index_sha256=hashlib.sha256((output / 'reference-index.json').read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256((output / 'model-index.json').read_bytes()).hexdigest(),
        authorization='Stage1 only: thirty seed42 runs, five unchanged arms, six mazes, all registered checkpoints and blocks, per-pair records. Four allocated CPU-hours including reservations. Stop before seeds43--46.'))
    digest = hashlib.sha256((output / 'pilot-registration.json').read_bytes()).hexdigest()
    atomic_json(output / 'pilot-scope.json', scope(digest, destination))
    manifest.update(authorization_plan=str(destination / 'pilot-registration.json'), authorization_sha256=digest,
        budget_scope=str(destination / 'pilot-scope.json'), budget_scope_sha256=hashlib.sha256((output / 'pilot-scope.json').read_bytes()).hexdigest(),
        stop_path=str(destination / 'STOP.json'), job_name='tu-pair-diagnostics-pilot', operations_commit=commit,
        operations_manifest=str(destination / 'operations/manifest.json'),
        operations_manifest_sha256=hashlib.sha256((operations / 'manifest.json').read_bytes()).hexdigest())
    atomic_json(output / 'pilot-manifest.json', manifest)
    return dict(operations_commit=commit, registration_sha256=digest, reference_runs=150, pilot_runs=30,
                hard_reservation_cpu_hours=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'destination', 'baseline', 'her', 'admissibility'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.baseline, args.her, args.admissibility)))
