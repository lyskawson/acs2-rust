import argparse
import hashlib
import json
from pathlib import Path
import sys

from execution import authorization_manifest as registered_manifest, measurement_identity
from grid_protocol import atomic_json, load_plan
from grid_dispatch import dispatch

APPROVED_SHA256 = '894596d2f792a2327ff5fd061c4344a17bcb96b01b9a96dd4f58a61a1e3a2888'
APPROVED_SECONDS = 45 * 3600


def approved(path):
    path = Path(path)
    plan = load_plan(path, APPROVED_SHA256)
    expected = json.loads(Path(__file__).with_name('approval.json').read_text())
    actual = json.loads(path.with_name('approval.json').read_text())
    if actual != expected or expected['plan_sha256'] != APPROVED_SHA256 or expected['phase_cpu_hour_limit'] * 3600 != APPROVED_SECONDS:
        raise ValueError('this exact registration and forty-five cumulative CPU-hours require the saved user approval')
    if expected['pilot_allocation_cpu_seconds'] != 4784 or any(plan.get(key) != value for key, value in measurement_identity().items()):
        raise ValueError('approved pilot or frozen measurement differs')
    return plan


def authorization_manifest(manifest, registration_path=None):
    if registration_path is None:
        raise ValueError('the approved registration is required')
    approved(registration_path)
    return registered_manifest(manifest, registration_path)


def manifest(root, purpose, configuration_id=None):
    root = Path(root)
    plan = approved(root / 'approved-plan.json')
    operations = root / 'execution-operations/manifest.json'
    proposed = dict(stage='tuning_pending_approval' if purpose == 'tuning' else 'approved_grid', purpose=purpose,
                    authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256)
    if configuration_id is not None:
        proposed['configuration_id'] = configuration_id
    result = authorization_manifest(proposed, root / 'approved-plan.json')
    return dict(result, authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        budget_scope=str(root / 'scope.json'), budget_scope_sha256=hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest(),
        stop_path=str(root / 'STOP.json'), job_name='tu-hindsight-proportion-' + purpose,
        operations_commit=json.loads(operations.read_text())['commit'], operations_manifest=str(operations),
        operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())


def execute(root, repository, purpose, configuration_id=None):
    root = Path(root)
    resolved = manifest(root, purpose, configuration_id)
    name = 'test-' + configuration_id if purpose == 'test' else purpose
    path = root / 'manifests' / (name + '.json')
    path.parent.mkdir(exist_ok=True)
    atomic_json(path, resolved)
    return dispatch(root / name, path, repository, root / 'approval.json')


if __name__ == '__main__':
    if sys.argv[1:] == ['--derive-manifest']:
        print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', required=True, type=Path)
        parser.add_argument('--repo', required=True, type=Path)
        parser.add_argument('--purpose', required=True, choices=['tuning', 'control', 'test'])
        parser.add_argument('--configuration-id')
        args = parser.parse_args()
        print(json.dumps(execute(args.root, args.repo, args.purpose, args.configuration_id)))
