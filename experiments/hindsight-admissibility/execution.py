import copy
import hashlib
import json
from pathlib import Path
import sys

from protocol import pilot
from grid_protocol import load_plan

APPROVED_SHA256 = 'f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da'
MEASUREMENT_COMMIT = '3828e430b13138b7458d65a840c1c833382f6c4a'
BINARY_SHA256 = 'a7422da3904e9b0d1a5df2152c3508bda66f27f86651ee3af929e230e9391893'
PILOT_SECONDS = 1723


def approved(path):
    path = Path(path)
    try:
        plan = load_plan(path, APPROVED_SHA256)
        confirmation = json.loads(path.with_name('approval.json').read_text())
    except OSError as error:
        raise ValueError('the registered plan and its explicit approval are required') from error
    expected = json.loads(Path(__file__).with_name('approval.json').read_text())
    if confirmation != expected or confirmation.get('approved') is not True or confirmation.get('plan_sha256') != APPROVED_SHA256:
        raise ValueError('hindsight admissibility requires this approved registration and 30 CPU-hours')
    if (confirmation['phase_cpu_hour_limit'] != 30 or confirmation['pilot_allocation_cpu_seconds'] != PILOT_SECONDS
            or plan['measurement_commit'] != MEASUREMENT_COMMIT or plan['binary_sha256'] != BINARY_SHA256):
        raise ValueError('approved measurement or cumulative allocation limit differs')
    return plan


def batches(plan):
    return [(config, agent) for config in plan['configurations'] for agent in plan['agents']]


def study_scope(plan, root):
    return dict(authorization_sha256=APPROVED_SHA256, pilot_allocation_cpu_seconds=PILOT_SECONDS,
                accounting_since='2026-10-24', batch_roots=[str(Path(root) / (config['id'] + '_' + agent))
                                                         for config, agent in batches(plan)])


def pilot_subset(plan, config, agent, root):
    data = (Path(root) / 'pilot-rows.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['cost_pilot']['rows_sha256']:
        raise ValueError('pilot identity archive differs from the approved bytes')
    lines = [line for line in data.splitlines() if
             (json.loads(line)['task'], json.loads(line)['agent'], json.loads(line)['seed']) == (config['task'], agent, 42)]
    if [json.loads(line)['nominal_step'] for line in lines] != config['targets']:
        raise ValueError('pilot identity archive is missing a registered point')
    return b'\n'.join(lines) + b'\n'


def pilot_references(plan, config, agent, local_root, declared_root):
    name = config['id'] + '_' + agent + '.jsonl'
    expected = pilot_subset(plan, config, agent, local_root)
    path = Path(local_root) / 'pilot' / name
    if path.read_bytes() != expected:
        raise ValueError('pilot cell was not derived from the approved complete archive')
    return {'|'.join([config['id'], agent, '42']): [dict(path=str(Path(declared_root) / 'pilot' / name),
        sha256=hashlib.sha256(expected).hexdigest(), targets=config['targets'], blocks=['training', 'validation', 'test'])]}


def grid_manifest(plan, index, local_root, declared_root):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('unregistered admissibility grid batch')
    config, agent = batches(plan)[index]
    return dict(stage='approved_grid', experiment='hindsight-admissibility', grid_batch=index,
        configurations=[copy.deepcopy(config)], agents=[agent], seeds=list(plan['seeds']), batch_size=20,
        partition=plan['execution']['partition'], cpu_model=plan['execution']['cpu_model'], concurrency=20,
        memory_mib=plan['forecast']['memory_mib'], phase_cpu_hour_limit=30,
        pilot_allocation_cpu_seconds=PILOT_SECONDS, accounting_since='2026-10-24',
        measurement_commit=MEASUREMENT_COMMIT, binary_sha256=BINARY_SHA256, preset=plan['preset'],
        identity_references=pilot_references(plan, config, agent, local_root, declared_root))


def measurement_identity():
    identity = json.loads(Path(__file__).with_name('measurement.json').read_text())
    if len(identity.get('measurement_commit', '')) != 40 or len(identity.get('binary_sha256', '')) != 64:
        raise ValueError('measurement identity is not frozen')
    return identity


def authorization_manifest(manifest, registration_path=None):
    if manifest['stage'] != 'authorized_cost_pilot':
        if manifest['stage'] != 'approved_grid' or registration_path is None:
            raise ValueError('the approved fixed-grid registration is required')
        plan = approved(registration_path)
        local_root = Path(registration_path).parent
        declared_root = Path(manifest['authorization_plan']).parent
        scope_path = local_root / 'scope.json'
        if json.loads(scope_path.read_text()) != study_scope(plan, declared_root):
            raise ValueError('accounting must cover all twelve roots and every pilot allocation')
        return dict(grid_manifest(plan, manifest.get('grid_batch'), local_root, declared_root),
            authorization_plan=str(declared_root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
            budget_scope=str(declared_root / 'scope.json'), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
            stop_path=str(declared_root / 'STOP.json'), job_name='tu-hindsight-admissibility')
    return pilot() | measurement_identity()


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
