import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from protocol import ROOT, PILOT_SETTINGS
from execution import pilot_manifest, scope
from references import shortened_rows
from grid_protocol import atomic_json
from collect import outside_repository


def prepare(output, destination, admissibility):
    output, destination = outside_repository(output), Path(destination)
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('operations source must be clean')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    references = shortened_rows(admissibility)
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for (config_id, seed), reference in references.items():
        name = f'references/{config_id}_s{seed}.jsonl'
        path = output / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(reference['data'])
        index.append(dict(configuration_id=config_id, seed=seed, file=name, sha256=reference['sha256']))
    atomic_json(output / 'reference-index.json', index)
    operations = output / 'operations'
    hashes = {}
    for name in subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'ls-files', 'experiments'], text=True).splitlines():
        source, target = ROOT / name, operations / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(source.stat().st_mode & 0o777)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(operations / 'manifest.json', dict(commit=commit, files=hashes))
    manifests = [pilot_manifest(index, output, destination) for index in range(len(PILOT_SETTINGS))]
    atomic_json(output / 'pilot-registration.json', dict(manifests=manifests,
        reference_index_sha256=hashlib.sha256((output / 'reference-index.json').read_bytes()).hexdigest(),
        authorization='Stage 1 user instruction; 30 seed42 allocations, at most 5 CPU-hours including reservations. No tuning, control grid or test grid approval.'))
    digest = hashlib.sha256((output / 'pilot-registration.json').read_bytes()).hexdigest()
    atomic_json(output / 'pilot-scope.json', scope(digest, destination))
    for index, manifest in enumerate(manifests):
        manifest.update(authorization_plan=str(destination / 'pilot-registration.json'), authorization_sha256=digest,
            budget_scope=str(destination / 'pilot-scope.json'), budget_scope_sha256=hashlib.sha256((output / 'pilot-scope.json').read_bytes()).hexdigest(),
            stop_path=str(destination / 'STOP.json'), job_name='tu-her-proportion-pilot', operations_commit=commit,
            operations_manifest=str(destination / 'operations/manifest.json'),
            operations_manifest_sha256=hashlib.sha256((operations / 'manifest.json').read_bytes()).hexdigest())
        atomic_json(output / f'pilot-manifest-{index}.json', manifest)
    return dict(operations_commit=commit, registration_sha256=digest, reference_cells=len(references))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--admissibility', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.admissibility)))
