import hashlib
import json
from pathlib import Path
import sys

from protocol import ARM, EXPERIMENT, pilot, recorded
from grid_protocol import load_plan

APPROVED_SHA256 = 'cb38ee22abc0a73c2411adc0a2e3ba921c36b45290353359bbe9ea95d6e57449'


def identity():
    return json.loads(Path(__file__).with_name('measurement.json').read_text())


def scope(digest, destination):
    return dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-11-04',
                batch_roots=[str(Path(destination) / 'pilot')])


def approved(registration_path=None):
    source = Path(__file__).with_name('plan.json')
    path = Path(registration_path) if registration_path else source
    if not path.exists():
        raise ValueError('stage2 approval requires the registered plan')
    plan = load_plan(path, APPROVED_SHA256)
    if path.read_bytes() != source.read_bytes():
        raise ValueError('approved registration differs from the source plan')
    approval = json.loads(path.with_name('approval.json').read_text())
    if (approval != json.loads(source.with_name('approval.json').read_text())
            or approval.get('approved') is not True or approval.get('plan_sha256') != APPROVED_SHA256
            or type(approval.get('phase_cpu_hour_limit')) is not int or approval['phase_cpu_hour_limit'] != 15
            or approval.get('includes_pilot_cpu_seconds') != 865
            or any(plan.get(key) != value for key, value in identity().items())):
        raise ValueError('stage2 approval or measurement identity differs')
    return plan, approval


def continuation_scope(destination):
    root = Path(destination)
    return dict(authorization_sha256=APPROVED_SHA256, pilot_allocation_cpu_seconds=0,
        accounting_since='2026-11-04', accounting_root=str(root),
        batch_roots=[str(root/name) for name in ('pilot','diagnostics','compact')],
        immutable_batch_roots=[str(root/'pilot')])


def validate_inputs(root, plan):
    for name in ('pilot-summary','pilot-registration','reference-index','model-index'):
        if hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest() != plan['cost_pilot'][name+'_sha256']:
            raise ValueError('registered pilot input changed: '+name)


def continuation(plan, purpose):
    if purpose not in ('diagnostics','compact'):
        raise ValueError('only registered diagnostics or compact replications are authorized')
    arrays = plan['forecast']['arrays'][:1] if purpose == 'diagnostics' else plan['forecast']['arrays'][1:]
    configs = []
    for number, array in enumerate(arrays):
        for identifier in array['configurations']:
            config = next(config for config in plan['configurations'] if config['id'] == identifier)
            configs.append(dict(recorded(config, array['limit_minutes'], purpose=='diagnostics'),batch_group=str(number)))
    seeds = arrays[0]['seeds']
    if any(array['seeds'] != seeds for array in arrays):
        raise ValueError('registered array seed sets differ within a stream')
    grouping = dict(batch_size=24) if purpose=='diagnostics' else dict(batch_by='group')
    return dict(stage='approved_grid',experiment=EXPERIMENT,purpose=purpose,**identity(),agents=[ARM],
        seeds=seeds,configurations=configs,**grouping,partition='lem-cpu-normal',
        cpu_model='AMD EPYC 9554 64-Core Processor',memory_mib=plan['forecast']['memory_mib'],
        concurrency=20,phase_cpu_hour_limit=15,pilot_allocation_cpu_seconds=0,accounting_since='2026-11-04')


def authorization_manifest(manifest, registration_path=None):
    if manifest.get('stage') != 'authorized_cost_pilot':
        if manifest.get('stage') != 'approved_grid' or registration_path is None:
            raise ValueError('stage2 approval requires the exact plan and CPU-hour limit')
        plan, _ = approved(registration_path)
        root = Path(registration_path).parent
        if manifest.get('authorization_sha256') != APPROVED_SHA256:
            raise ValueError('execution belongs to a different registration')
        validate_inputs(root,plan)
        scope_path = root/'continuation/scope.json'
        if json.loads(scope_path.read_text()) != continuation_scope(root):
            raise ValueError('all attempts must belong to the cumulative accounting scope')
        return dict(continuation(plan,manifest.get('purpose')),budget_scope=str(scope_path),
            budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
            stop_path=str(root/'continuation/STOP.json'))
    if registration_path is not None:
        raise ValueError('pilot authorization does not use a stage2 approval')
    root = Path(manifest['authorization_plan']).parent
    digest = manifest['authorization_sha256']
    registration = load_plan(root / 'pilot-registration.json', digest)
    expected = pilot(identity())
    if registration['manifest'] != expected or registration['destination'] != str(root):
        raise ValueError('pilot differs from six registered seed42 runs')
    for name in ('reference-index', 'model-index'):
        if hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest() != registration[name + '_sha256']:
            raise ValueError('pilot frozen input index changed')
    scope_path = root / 'pilot-scope.json'
    if json.loads(scope_path.read_text()) != scope(digest, root):
        raise ValueError('pilot accounting scope changed')
    return dict(expected, budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                stop_path=str(root / 'STOP.json'))


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
