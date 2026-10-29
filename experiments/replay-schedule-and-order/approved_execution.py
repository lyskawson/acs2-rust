import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import archive
from execution import authorization_manifest as registered_manifest, measurement_identity
from grid_protocol import atomic_json, load_plan
from grid_dispatch import dispatch

APPROVED_SHA256 = '01d93796e56efeea2dc2b85e0264bc98dfaec4a16fe7cb08052fd0515f7b1f71'
APPROVED_SECONDS = 60 * 3600


def approved(path):
    path = Path(path)
    plan = load_plan(path, APPROVED_SHA256)
    expected = json.loads(Path(__file__).with_name('approval.json').read_text())
    if json.loads(path.with_name('approval.json').read_text()) != expected:
        raise ValueError('the exact saved user approval is required')
    if (expected['plan_sha256'] != APPROVED_SHA256 or expected['approved'] is not True
            or expected['phase_cpu_hour_limit'] * 3600 != APPROVED_SECONDS
            or expected['pilot_allocation_cpu_seconds'] != 8147
            or plan['cost_pilot']['allocation_cpu_seconds'] != 8147):
        raise ValueError('the approved registration includes sixty hours and both pilot rounds')
    if any(plan.get(key) != value for key, value in measurement_identity().items()):
        raise ValueError('the approved frozen measurement identity changed')
    return plan


@contextmanager
def catalog_cache():
    original, entries = archive.catalog, {}
    def cached(path):
        key = Path(path).resolve()
        if key not in entries:
            entries[key] = original(path)
        return entries[key]
    archive.catalog = cached
    try:
        yield
    finally:
        archive.catalog = original


def authorization_manifest(manifest, registration_path=None):
    if registration_path is None:
        raise ValueError('the approved registration is required')
    approved(registration_path)
    with catalog_cache():
        return registered_manifest(manifest, registration_path)


def manifest(root, purpose, configuration_id=None, axis=None, slot=None):
    root = Path(root)
    approved(root / 'approved-plan.json')
    operations = root / 'execution-operations/manifest.json'
    proposed = dict(stage='tuning_pending_approval' if purpose == 'screening' else 'approved_grid', purpose=purpose,
        authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256)
    if purpose == 'test':
        proposed.update(configuration_id=configuration_id, axis=axis, slot=slot)
    result = authorization_manifest(proposed, root / 'approved-plan.json')
    return dict(result, authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        stop_path=str(root / 'STOP.json'), job_name='tu-replay-' + purpose,
        operations_commit=json.loads(operations.read_text())['commit'], operations_manifest=str(operations),
        operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())


def execute(root, repository, purpose, configuration_id=None, axis=None, slot=None, recovery=None):
    root = Path(root)
    resolved = manifest(root, purpose, configuration_id, axis, slot)
    name = 'screening' if purpose == 'screening' else f'test-{axis}-{configuration_id}-{slot}'
    path = root / 'manifests' / (name + '.json')
    path.parent.mkdir(exist_ok=True)
    atomic_json(path, resolved)
    return dispatch(root / name, path, repository, root / 'approval.json', recovery=recovery)


if __name__ == '__main__':
    if sys.argv[1:] == ['--derive-manifest']:
        print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', required=True, type=Path)
        parser.add_argument('--repo', required=True, type=Path)
        parser.add_argument('--purpose', required=True, choices=['screening', 'test'])
        parser.add_argument('--configuration-id')
        parser.add_argument('--axis', choices=['B', 'C'])
        parser.add_argument('--slot', choices=['shortened', 'er', 'order-control'])
        parser.add_argument('--recovery', type=Path)
        args = parser.parse_args()
        print(json.dumps(execute(args.root, args.repo, args.purpose, args.configuration_id, args.axis, args.slot, args.recovery)))
