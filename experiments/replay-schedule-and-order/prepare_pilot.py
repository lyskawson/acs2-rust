import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from protocol import ROOT, PROPORTION_SHA256, SHORTENED, BASE_PROPORTIONS, PILOT_SETTINGS, configurations
from execution import pilot_manifest, scope
from grid_attempts import select_attempts
from grid_protocol import atomic_json, load_plan
from collect import outside_repository


def reference_rows(archive):
    archive = Path(archive)
    subprocess.run([sys.executable, '-B', str(ROOT / 'experiments/hindsight-proportion/verify.py'),
                    '--root', str(archive)], check=True, capture_output=True, text=True)
    load_plan(archive / 'approved-plan.json', PROPORTION_SHA256)
    selections = json.loads((archive / 'selections.json').read_text())['selections']
    result = []
    for base in configurations():
        p = BASE_PROPORTIONS[base['task']]
        if selections[base['id']]['proportion'] != p:
            raise ValueError('base proportion differs from the approved saved selection')
        for purpose, batch_name, seeds in [('test', 'test-' + base['id'], [42]),
                                           ('screening-base', 'tuning', range(42, 47))]:
            batch = archive / batch_name
            manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
            configs = [config for config in manifest['configurations'] if config['registered_id'] == base['id']
                       and config['agent_settings'][SHORTENED]['relabeled_proportion'] == p]
            if len(configs) != 1:
                raise ValueError('selected-proportion reference is ambiguous')
            config = configs[0]
            selected, _ = select_attempts(batch, json.loads((batch / 'ledger.json').read_text()), config)
            for seed in seeds:
                path = selected[SHORTENED, seed]['path'] / 'rows.jsonl'
                data = path.read_bytes()
                rows = [json.loads(line) for line in data.splitlines()]
                expected_blocks = {'validation'} if purpose == 'screening-base' else {'training', 'validation', 'test'}
                if len(rows) != len(base['targets']) or any(set(row['evaluation_blocks']) != expected_blocks for row in rows):
                    raise ValueError('reference rows have an unregistered block or point grid')
                result.append(dict(purpose=purpose, configuration_id=base['id'], seed=seed,
                    data=data, source=str(path), sha256=hashlib.sha256(data).hexdigest()))
    return result


def prepare(output, destination, proportion):
    output, destination = outside_repository(output), Path(destination)
    if not destination.is_absolute():
        raise ValueError('destination must be the absolute cluster study root')
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('freeze clean operations before the pilot')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    sources = reference_rows(proportion)
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for source in sources:
        file = 'references/' + source['purpose'] + '_' + source['configuration_id'] + '_s' + str(source['seed']) + '.jsonl'
        path = output / file
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(source['data'])
        index.append(dict({key: value for key, value in source.items() if key != 'data'}, file=file))
    atomic_json(output / 'reference-index.json', index)
    selection_bytes = (Path(proportion) / 'selections.json').read_bytes()
    (output / 'base-selections.json').write_bytes(selection_bytes)
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
        base_selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        authorization='Stage1 user instruction: sixty seed42 allocations, six mazes, ten settings, all registered blocks and points; twelve CPU-hours including all reservations. No screening or test approval.'))
    digest = hashlib.sha256((output / 'pilot-registration.json').read_bytes()).hexdigest()
    atomic_json(output / 'pilot-scope.json', scope(digest, destination))
    for index, manifest in enumerate(manifests):
        manifest.update(authorization_plan=str(destination / 'pilot-registration.json'), authorization_sha256=digest,
            budget_scope=str(destination / 'pilot-scope.json'), budget_scope_sha256=hashlib.sha256((output / 'pilot-scope.json').read_bytes()).hexdigest(),
            stop_path=str(destination / 'STOP.json'), job_name='tu-replay-pilot', operations_commit=commit,
            operations_manifest=str(destination / 'operations/manifest.json'),
            operations_manifest_sha256=hashlib.sha256((operations / 'manifest.json').read_bytes()).hexdigest())
        atomic_json(output / f'pilot-manifest-{index}.json', manifest)
    return dict(operations_commit=commit, registration_sha256=digest, reference_cells=len(sources))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--proportion', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.destination, args.proportion)))
