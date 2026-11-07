import hashlib
import json
from pathlib import Path

from protocol import pilot, NEW_ARMS
from grid_authorization import require_derivation
from grid_protocol import load_plan


def identity():
    return json.loads(Path(__file__).with_name('measurement.json').read_text())


def scope(digest, destination):
    return dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0,
                accounting_since='2026-11-07', batch_roots=[str(Path(destination) / 'pilot')])


def registered_runs(plan):
    runs = []
    for config in plan['configurations']:
        for arm in NEW_ARMS:
            identifier = f"{config['id']}_{arm}_s42_a01"
            runs.append(dict(configuration_id=config['id'], agent=arm, seed=42, attempt_id=identifier,
                cause='registered first attempt', relative_directory=f'attempts/{identifier}/measurement/{config["id"]}_{arm}_s42',
                limit_seconds=config['limits_minutes'][arm] * 60))
    return runs


def authorization_manifest(manifest):
    if manifest.get('stage') != 'authorized_cost_pilot':
        raise ValueError('stage 2 requires a separately approved registered plan and CPU-hour limit')
    root = Path(manifest['authorization_plan']).parent
    digest = manifest['authorization_sha256']
    registration = load_plan(root / 'pilot-registration.json', digest)
    expected = pilot(identity())
    if registration['manifest'] != expected or registration['destination'] != str(root):
        raise ValueError('pilot differs from the twelve registered seed42 runs')
    for name in ('reference-index', 'model-index'):
        if hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest() != registration[name.replace('-', '_') + '_sha256']:
            raise ValueError('pilot frozen input index changed')
    scope_path = root / 'pilot-scope.json'
    if json.loads(scope_path.read_text()) != scope(digest, root):
        raise ValueError('pilot accounting scope changed')
    return dict(expected, budget_scope=str(scope_path),
        budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(), stop_path=str(root / 'STOP.json'))


def require_authorization(root, manifest):
    expected = authorization_manifest(manifest)
    if Path(root).resolve() != (Path(manifest['authorization_plan']).parent / 'pilot').resolve():
        raise ValueError('pilot result directory differs from its registration')
    require_derivation(manifest, expected)
