import hashlib
import json
from pathlib import Path
import sys

from protocol import ARM, EXPERIMENT, pilot, recorded
from grid_protocol import load_plan


def identity():
    return json.loads(Path(__file__).with_name('measurement.json').read_text())


def scope(digest, destination):
    return dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-11-05',
                batch_roots=[str(Path(destination) / 'pilot')])


def authorization_manifest(manifest, registration_path=None):
    if manifest.get('stage') != 'authorized_cost_pilot' or registration_path is not None:
        raise ValueError('STOP: only the six seed42 pilot runs are authorized; stage2 requires approval and a CPU-hour limit')
    root = Path(manifest['authorization_plan']).parent
    digest = manifest['authorization_sha256']
    registration = load_plan(root / 'pilot-registration.json', digest)
    expected = pilot(identity())
    if registration['manifest'] != expected or registration['destination'] != str(root):
        raise ValueError('pilot differs from the six registered seed42 runs')
    for name in ('reference-index', 'model-index'):
        if hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest() != registration[name + '_sha256']:
            raise ValueError('frozen input index changed')
    scope_path = root / 'pilot-scope.json'
    if json.loads(scope_path.read_text()) != scope(digest, root):
        raise ValueError('pilot accounting scope changed')
    return dict(expected, budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(), stop_path=str(root / 'STOP.json'))


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
