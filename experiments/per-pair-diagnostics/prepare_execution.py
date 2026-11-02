import argparse
import hashlib
import json
from pathlib import Path
import shutil

from protocol import ROOT
from execution import scope, measurement_identity
from grid_protocol import atomic_json, load_plan
from collect import outside_repository

APPROVED_SHA256 = 'ec83bffbba7ca0d14315138abf03d6b485900d6f3c63dcfa0172ebe74a53d8d5'
OPERATIONS_COMMIT = '4c38c2817af882ce60738963c9aa30a1cbd1d61d'


def approved():
    path = Path(__file__).with_name('plan.json')
    plan = load_plan(path, APPROVED_SHA256)
    approval = json.loads(path.with_name('approval.json').read_text())
    if (approval.get('approved') is not True or approval.get('plan_sha256') != APPROVED_SHA256
            or approval.get('phase_cpu_hour_limit') != 15 or approval.get('includes_pilot_cpu_seconds') != 3672
            or approval.get('operations_commit') != OPERATIONS_COMMIT):
        raise ValueError('exact user approval differs')
    if any(plan[key] != value for key, value in measurement_identity().items()):
        raise ValueError('measurement identity changed')
    return plan, approval


def freeze_operations(base, target, plan_path, approval_path):
    base, target = Path(base), Path(target)
    manifest = json.loads((base / 'manifest.json').read_text())
    if manifest['commit'] != OPERATIONS_COMMIT:
        raise ValueError('operations differ from the approved pinned source')
    for name, digest in manifest['files'].items():
        path = base / name
        if not path.resolve().is_relative_to(base.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('pinned operations changed')
    shutil.copytree(base, target)
    additions = {}
    for source in (Path(plan_path), Path(approval_path)):
        relative = Path('experiments/per-pair-diagnostics') / source.name
        destination = target / relative
        if destination.exists():
            raise ValueError('registration would replace pinned operations')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        additions[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_json(target / 'manifest.json', dict(commit=OPERATIONS_COMMIT,
        base_manifest_sha256=hashlib.sha256((base / 'manifest.json').read_bytes()).hexdigest(),
        files=dict(manifest['files'], **additions), registration_additions=sorted(additions)))
    return len(manifest['files']), additions


def prepare(root, destination):
    root, destination = outside_repository(root), Path(destination)
    plan, approval = approved()
    if not destination.is_absolute() or (root / 'execution-operations').exists() or (root / 'replication').exists():
        raise ValueError('execution already prepared or destination is not absolute')
    inventory = json.loads((root / 'raw-checksums.json').read_text())
    for name, digest in inventory['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('verified pilot artifact changed: ' + name)
    for name, key in [('pilot-summary.json', 'summary_sha256'), ('reference-index.json', 'references_sha256'),
                      ('model-index.json', 'models_sha256'), ('pilot-registration.json', 'pilot_registration_sha256')]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != plan['cost_pilot'][key]:
            raise ValueError('approved pilot input differs')
    frozen = root.parent / 'pilot-frozen'
    shutil.copytree(root, frozen)
    path = Path(__file__).with_name('plan.json')
    shutil.copyfile(path, root / 'approved-plan.json')
    shutil.copyfile(path.with_name('approval.json'), root / 'approval.json')
    atomic_json(root / 'scope.json', scope(APPROVED_SHA256, destination))
    count, additions = freeze_operations(root / 'operations', root / 'execution-operations', path, path.with_name('approval.json'))
    result = dict(plan_sha256=APPROVED_SHA256, phase_cpu_hour_limit=15, pilot_cpu_seconds=3672,
        operations_commit=OPERATIONS_COMMIT, preserved_operations_files=count, registration_additions=additions,
        pilot_snapshot=str(frozen), measurement_rebuilt=False)
    atomic_json(root / 'execution-preparation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.destination)))
