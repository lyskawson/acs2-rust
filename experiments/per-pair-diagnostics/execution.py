import hashlib
import json
import math
from pathlib import Path
import sys

from protocol import ARMS, BLOCKS, EXPERIMENT, pilot, configurations, recorded
from references import read
from grid_protocol import load_plan


def measurement_identity():
    return json.loads(Path(__file__).with_name('measurement.json').read_text())


def scope(digest, destination):
    return dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-11-02',
                batch_roots=[str(Path(destination) / name) for name in ('pilot', 'replication')])


def identity_references(manifest, root, declared):
    references = read(root)
    return {'|'.join([config['id'], arm, str(seed)]): [dict(path=str(Path(declared) / references[config['id'], arm, seed]['file']),
            sha256=references[config['id'], arm, seed]['sha256'], targets=config['targets'], blocks=BLOCKS,
            comparison='per_pair_records')]
        for config in manifest['configurations'] for arm in manifest['agents'] for seed in manifest['seeds']}


def pilot_manifest(root, declared):
    manifest = pilot(measurement_identity())
    return dict(manifest, identity_references=identity_references(manifest, root, declared))


def replication(plan, limit):
    configs = []
    for config in configurations():
        maximum = max(cell['limit_minutes'] for cell in plan['forecast']['cells'] if cell['configuration_id'] == config['id'])
        configs.append(dict(recorded(config, maximum), batch_group=config['id']))
    return dict(stage='approved_grid', experiment=EXPERIMENT, purpose='replication', **measurement_identity(),
        agents=list(ARMS), seeds=list(range(43, 47)), configurations=configs, batch_by='group',
        partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor',
        memory_mib=plan['forecast']['memory_mib'], concurrency=20, phase_cpu_hour_limit=limit,
        pilot_allocation_cpu_seconds=0, accounting_since='2026-11-02')


def authorization_manifest(manifest, registration_path=None):
    root = Path(manifest['authorization_plan']).parent
    if manifest['stage'] == 'authorized_cost_pilot':
        registration = load_plan(root / 'pilot-registration.json', manifest['authorization_sha256'])
        expected = pilot_manifest(root, root)
        if registration['manifest'] != expected or registration['destination'] != str(root):
            raise ValueError('pilot differs from the exact thirty-run registration')
        if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != registration['reference_index_sha256']:
            raise ValueError('pilot reference index changed')
        scope_path = root / 'pilot-scope.json'
    else:
        if registration_path is None:
            raise ValueError('stage2 requires approval of the registered plan and cumulative CPU-hour limit')
        digest = hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest()
        plan = load_plan(registration_path, digest)
        approval = json.loads(Path(registration_path).with_name('approval.json').read_text())
        limit = approval.get('phase_cpu_hour_limit')
        if (approval.get('approved') is not True or approval.get('plan_sha256') != digest
                or type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError('stage2 needs explicit plan and CPU-limit approval')
        if any(plan.get(key) != value for key, value in measurement_identity().items()):
            raise ValueError('measurement identity changed after pilot')
        if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != plan['cost_pilot']['references_sha256']:
            raise ValueError('registered reference index changed')
        expected = replication(plan, limit)
        expected['identity_references'] = identity_references(expected, root, root)
        scope_path = root / 'scope.json'
    if json.loads(scope_path.read_text()) != scope(manifest['authorization_sha256'], root):
        raise ValueError('every pilot and replication attempt must share one accounting scope')
    return dict(expected, budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                stop_path=str(root / 'STOP.json'))


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
