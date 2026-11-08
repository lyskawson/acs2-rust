import hashlib
import json
from pathlib import Path

from protocol import EXPERIMENT, E1, recorded
from execution import identity
from grid_authorization import require_derivation
from grid_protocol import load_plan

APPROVED_SHA256 = '185e080e5df99cb933a65b76147463a19ea4ef6aa05fe646eb0e0b9be79f46a3'


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved(registration_path=None):
    source = Path(__file__).with_name('plan.json')
    path = Path(registration_path) if registration_path else source
    plan = load_plan(path, APPROVED_SHA256)
    approval = json.loads(path.with_name('approval.json').read_text())
    expected = json.loads(source.with_name('approval.json').read_text())
    if (path.read_bytes() != source.read_bytes() or approval != expected
            or approval.get('approved') is not True or approval.get('plan_sha256') != APPROVED_SHA256
            or type(approval.get('phase_cpu_hour_limit')) is not int or approval['phase_cpu_hour_limit'] != 29
            or approval.get('includes_pilot_cpu_seconds') != 1648
            or any(plan.get(key) != value for key, value in identity().items())
            or any(approval.get(key) != plan[key] for key in ('measurement_commit', 'binary_sha256'))):
        raise ValueError('stage2 requires the exact approved plan, binary and cumulative 29 CPU-hour limit')
    return plan, approval


def batch_name(array):
    prefix = 'episodic' if array['agent'] == E1 else 'shortened'
    suffix = 'pairs' if array['kind'] == 'per_pair' else 'compact-' + '-'.join(
        identifier.split('_c')[0] for identifier in array['configurations'])
    return prefix + '-' + suffix


def scope(destination, plan):
    root = Path(destination)
    return dict(authorization_sha256=APPROVED_SHA256, pilot_allocation_cpu_seconds=0,
        accounting_since='2026-11-07', accounting_root=str(root),
        batch_roots=[str(root / 'pilot')] + [str(root / batch_name(array)) for array in plan['forecast']['arrays']],
        immutable_batch_roots=[str(root / 'pilot')])


def validate_inputs(root, plan):
    root = Path(root)
    for name in ('pilot-summary', 'pilot-registration', 'reference-index', 'model-index'):
        if checksum(root / (name + '.json')) != plan['cost_pilot'][name + '_sha256']:
            raise ValueError('registered pilot input changed: ' + name)


def manifest_for(plan, index):
    if type(index) is not int or not 0 <= index < len(plan['forecast']['arrays']):
        raise ValueError('unknown registered forecast array')
    array = plan['forecast']['arrays'][index]
    configs = [recorded(next(config for config in plan['configurations'] if config['id'] == identifier),
                        array['limit_minutes'], array['kind'] == 'per_pair') for identifier in array['configurations']]
    return dict(stage='approved_grid', experiment=EXPERIMENT, forecast_index=index,
        **identity(), agents=[array['agent']], seeds=array['seeds'], configurations=configs,
        partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor',
        memory_mib=plan['forecast']['memory_mib'], concurrency=20, phase_cpu_hour_limit=29,
        pilot_allocation_cpu_seconds=0, accounting_since='2026-11-07')


def registered_runs(manifest):
    runs = []
    for config in manifest['configurations']:
        for agent in manifest['agents']:
            for seed in manifest['seeds']:
                identifier = f"{config['id']}_{agent}_s{seed}_a01"
                runs.append(dict(configuration_id=config['id'], agent=agent, seed=seed, attempt_id=identifier,
                    cause='registered first attempt', relative_directory=f'attempts/{identifier}/measurement/{config["id"]}_{agent}_s{seed}',
                    limit_seconds=config['limits_minutes'][agent] * 60))
    return runs


def require_authorization(root, manifest):
    if manifest.get('stage') != 'approved_grid' or manifest.get('authorization_sha256') != APPROVED_SHA256:
        raise ValueError('only approved stage2 replications are authorized')
    source = Path(manifest['authorization_plan'])
    plan, _ = approved(source)
    common = source.parent
    validate_inputs(common, plan)
    index = manifest.get('forecast_index')
    expected = manifest_for(plan, index)
    if Path(root).resolve() != (common / batch_name(plan['forecast']['arrays'][index])).resolve():
        raise ValueError('batch root differs from the registered forecast array')
    scope_path = common / 'continuation/scope.json'
    if json.loads(scope_path.read_text()) != scope(common, plan):
        raise ValueError('stage2 accounting scope changed')
    require_derivation(manifest, dict(expected, budget_scope=str(scope_path),
        budget_scope_sha256=checksum(scope_path), stop_path=str(common / 'continuation/STOP.json')))
