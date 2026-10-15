import copy
import hashlib
import json
from pathlib import Path
import sys

from protocol import AGENTS, PILOT_PARTS, RECOVERY_PARTS, ROOT, pilot, recovery

sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import load_plan

APPROVED_SHA256 = '9428773fa7861f6102fdccb165ae711b597d1fd61a9e644f8becb17399ebf4eb'
MEASUREMENT_COMMIT = '82c08b5a89a63878fddfb76be8d5e25d0ae17a41'
BINARY_SHA256 = '0c8dcb9c539462f9b84800519237a24e34c49e4d81b70a79a17cae0d7a98816d'


def approved(path):
    path = Path(path)
    plan = load_plan(path, APPROVED_SHA256)
    confirmation = json.loads(path.with_name('approval.json').read_text())
    expected = json.loads(Path(__file__).with_name('approval.json').read_text())
    if confirmation != expected or confirmation['approved'] is not True or confirmation['plan_sha256'] != APPROVED_SHA256:
        raise ValueError('VCP requires the registered user approval including hindsight and 110 CPU-hours')
    if plan['measurement_commit'] != MEASUREMENT_COMMIT or plan['binary_sha256'] != BINARY_SHA256:
        raise ValueError('VCP measurement differs from the approved source and binary')
    return plan


def batches(plan):
    return [(config, agent) for config in plan['configurations'] for agent in plan['agents']]


def study_scope(plan, root):
    return dict(authorization_sha256=APPROVED_SHA256, pilot_allocation_cpu_seconds=16041,
                accounting_since='2026-10-15', batch_roots=[str(Path(root) / (config['id'] + '_' + agent))
                                                         for config, agent in batches(plan)])


def pilot_references(plan, config, agent, local_root, declared_root):
    path = Path(local_root) / 'pilot-rows.jsonl'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['cost_pilot']['rows_sha256']:
        raise ValueError('VCP pilot identity rows changed')
    temperature = 0 if agent.endswith('_uniform') else 24
    rows = [json.loads(line) for line in data.splitlines()]
    selected = [row for row in rows if (row['task'], row['agent'], row['agent_parameters']['temperature']) ==
                (config['task'], agent, temperature)]
    if not selected:
        if config['task'] != 'maze6' or temperature != 24:
            raise ValueError('missing required VCP pilot identity oracle')
        return {}
    if [row['nominal_step'] for row in selected] != config['targets']:
        raise ValueError('VCP pilot identity checkpoints differ')
    key = '|'.join([config['id'], agent, '42'])
    return {key: [dict(path=str(Path(declared_root) / path.name), sha256=plan['cost_pilot']['rows_sha256'],
                       targets=config['targets'], blocks=['validation'], comparison='vcp_pilot')]}


def grid_manifest(plan, index, local_root, declared_root):
    if type(index) is not int or not 0 <= index < 24:
        raise ValueError('unregistered VCP grid batch')
    config, agent = batches(plan)[index]
    config = copy.deepcopy(config)
    return dict(stage='approved_grid', experiment='value-consistency', grid_batch=index,
                configurations=[config], agents=[agent], seeds=list(plan['seeds']), batch_size=20,
                partition=plan['execution']['partition'], cpu_model=plan['execution']['cpu_model'],
                concurrency=20, memory_mib=plan['forecast']['memory_mib'], phase_cpu_hour_limit=110,
                pilot_allocation_cpu_seconds=16041, accounting_since='2026-10-15',
                measurement_commit=MEASUREMENT_COMMIT, binary_sha256=BINARY_SHA256, preset=plan['preset'],
                identity_references=pilot_references(plan, config, agent, local_root, declared_root))


def measurement_identity(name='measurement.json'):
    path = Path(__file__).with_name(name)
    identity = json.loads(path.read_text())
    if len(identity.get('measurement_commit', '')) != 40 or len(identity.get('binary_sha256', '')) != 64:
        raise ValueError('VCP measurement source and binary have not been frozen')
    return identity


def pilot_measurement_identity():
    return measurement_identity('pilot-measurement.json')


def authorization_manifest(manifest, registration_path=None):
    if manifest['stage'] != 'authorized_cost_pilot':
        if manifest['stage'] != 'approved_grid' or registration_path is None:
            raise ValueError('VCP stage 2 requires the approved fixed-grid registration')
        plan = approved(registration_path)
        declared_root = Path(manifest['authorization_plan']).parent
        local_root = Path(registration_path).parent
        scope_path = local_root / 'scope.json'
        if json.loads(scope_path.read_text()) != study_scope(plan, declared_root):
            raise ValueError('VCP accounting must include all twenty-four registered roots and every pilot attempt')
        return dict(grid_manifest(plan, manifest.get('grid_batch'), local_root, declared_root),
                    authorization_plan=str(declared_root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
                    budget_scope=str(declared_root / 'scope.json'), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                    stop_path=str(declared_root / 'STOP.json'), job_name='tu-value-consistency')
    part = manifest.get('pilot_part')
    if part in RECOVERY_PARTS:
        return recovery(part) | pilot_measurement_identity()
    if part not in PILOT_PARTS:
        raise ValueError('unregistered VCP pilot part')
    return pilot(part) | pilot_measurement_identity()


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
