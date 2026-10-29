import hashlib
import json
import math
import os
from pathlib import Path
import sys

from protocol import (BLOCKS, CORRECTED_PILOT_INDICES, ER, EXPERIMENT, PILOT_SETTINGS, SCHEDULES, SCREEN_SETTINGS, SHORTENED,
                      configurations, pilot, pilot_roots, scope_roots, setting)
from previous_pilot import pilot_batch, reference_path
from tuning import select
from grid_protocol import load_plan
from grid_identity import compare, field_bits


def measurement_identity():
    value = json.loads(Path(__file__).with_name('measurement.json').read_text())
    if len(value.get('measurement_commit', '')) != 40 or len(value.get('binary_sha256', '')) != 64:
        raise ValueError('measurement identity must be frozen')
    return value


def read_index(root):
    root = Path(root)
    result = {}
    for item in json.loads((root / 'reference-index.json').read_text()):
        path = root / item['file']
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('frozen reference artifact changed')
        key = item['purpose'], item['configuration_id'], item['seed']
        if key in result:
            raise ValueError('duplicate frozen reference cell')
        result[key] = dict(item, rows=[json.loads(line) for line in path.read_bytes().splitlines()])
    expected = {('test', config['id'], 42) for config in configurations()} | {
        ('screening-base', config['id'], seed) for config in configurations() for seed in range(42, 47)}
    if set(result) != expected:
        raise ValueError('frozen base reference grid is incomplete')
    return result


def scope(digest, root, previous=None):
    registration = Path(root) / 'pilot-registration.json'
    if previous is None and registration.is_file():
        previous = json.loads(registration.read_text()).get('previous_pilot', {}).get('root')
    result = dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28',
                  batch_roots=[str(Path(root) / name) for name in scope_roots(previous)])
    if previous is not None:
        result.update(accounting_root=os.path.commonpath([str(root), str(previous)]),
                      immutable_batch_roots=[str(Path(previous) / name) for name in pilot_roots()])
    return result


def pilot_manifest(index, root, declared):
    result = dict(pilot(index), **measurement_identity(), identity_references={})
    refs = read_index(root)
    if PILOT_SETTINGS[index][1] == 'base':
        for config in result['configurations']:
            source = refs['test', config['registered_id'], 42]
            result['identity_references']['|'.join([config['id'], SHORTENED, '42'])] = [dict(
                path=str(Path(declared) / source['file']), sha256=source['sha256'], targets=config['targets'], blocks=BLOCKS)]
    return result


def derive(plan, purpose, selection=None, configuration_id=None, axis=None, slot=None):
    if purpose == 'screening':
        configs = []
        for base in plan['configurations']:
            for mode in SCREEN_SETTINGS:
                cell = next(cell for cell in plan['forecast']['cells'] if (cell['purpose'], cell['configuration_id'], cell['mode']) == ('screening', base['id'], mode))
                configs.append(dict(setting(base, SHORTENED, mode, ['validation'], cell['limit_minutes']), batch_group=cell['batch_group']))
        return dict(stage='tuning_pending_approval', purpose=purpose, configurations=configs,
                    agents=[SHORTENED], seeds=list(range(42, 47)), batch_by='group')
    if purpose != 'test' or selection is None or axis not in ('B', 'C'):
        raise ValueError('conditional test needs the saved screening choice and axis')
    base = next((config for config in plan['configurations'] if config['id'] == configuration_id), None)
    if base is None or slot not in (['shortened', 'er'] if axis == 'B' else ['shortened', 'er', 'order-control']):
        raise ValueError('unregistered test maze or arm')
    mode = selection['selections'][base['id']][axis]['setting']
    if mode not in (SCHEDULES if axis == 'B' else ['backward']):
        raise ValueError('base wins or test setting is unregistered: no run')
    agent = ER if slot == 'er' else SHORTENED
    if slot == 'order-control':
        mode = 'random-pass'
    cell = next(cell for cell in plan['forecast']['cells'] if
        (cell['purpose'], cell['configuration_id'], cell['agent'], cell['mode']) == ('test', base['id'], agent, mode))
    return dict(stage='approved_grid', purpose=purpose, configuration_id=configuration_id, axis=axis, slot=slot,
        configurations=[setting(base, agent, mode, BLOCKS, cell['limit_minutes'])],
        agents=[agent], seeds=list(range(42, 62)), batch_size=20)


def choose(plan, entries, references, default_rows):
    result = {}
    expected_entries = {(setting(config, SHORTENED, mode, ['validation'])['id'], SHORTENED, seed)
        for config in plan['configurations'] for mode in SCREEN_SETTINGS for seed in range(42, 47)}
    if set(entries) != expected_entries:
        raise ValueError('screening catalog differs from the exact registered grid')
    expected_keys = {(config['id'], 42) for config in plan['configurations']}
    if set(default_rows) != expected_keys:
        raise ValueError('default pilot identity is incomplete')
    for config in plan['configurations']:
        compare(default_rows[config['id'], 42], references['test', config['id'], 42]['rows'], [42], config['targets'], BLOCKS)
        base = [row for seed in range(42, 47) for row in references['screening-base', config['id'], seed]['rows']]
        if any(set(row['evaluation_blocks']) != {'validation'} for row in base):
            raise ValueError('reused screening base must contain validation only')
        by_mode = {mode: [row for seed in range(42, 47) for row in entries[
            setting(config, SHORTENED, mode, ['validation'])['id'], SHORTENED, seed]['rows']] for mode in SCREEN_SETTINGS}
        result[config['id']] = {axis: select(dict(base=base, **{mode: by_mode[mode] for mode in modes}), config['targets'], modes)
                               for axis, modes in [('B', SCHEDULES), ('C', ['backward'])]}
    return dict(selections=result, seeds=list(range(42, 47)), evaluation_blocks=['validation'],
                rule=plan['screening']['selection'], test_access='none')


def verified_screening(root):
    from grid_authorization import require_derivation
    root = Path(root)
    manifest = load_plan(root / 'screening/plan.json', (root / 'screening/plan.sha256').read_text().strip())
    require_derivation(manifest, authorization_manifest(manifest, root / 'approved-plan.json'))
    return manifest


def saved_selection(root, plan):
    from archive import catalog
    root = Path(root)
    verified_screening(root)
    references = read_index(root)
    if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != plan['cost_pilot']['references_sha256']:
        raise ValueError('reference index differs from registered pilot')
    pilot_entries = catalog(root / pilot_roots()[-1])[0]
    defaults = {(config['registered_id'], 42): pilot_entries[config['id'], SHORTENED, 42]['rows']
                for config in json.loads((root / pilot_roots()[-1] / 'plan.json').read_text())['configurations']}
    selected = choose(plan, catalog(root / 'screening')[0], references, defaults)
    if field_bits(json.loads((root / 'selections.json').read_text())) != field_bits(selected):
        raise ValueError('saved selection differs from complete validation-only screening')
    return selected


def study_references(manifest, root, declared):
    from archive import catalog
    result = {}
    for config in manifest['configurations']:
        agent = manifest['agents'][0]
        settings = config['agent_settings'][agent]
        mode = settings['replay_schedule'] if settings['replay_schedule'] != 'constant' else settings['replay_order']
        pilot_index = PILOT_SETTINGS.index((agent, mode))
        pilot_entries = catalog(pilot_batch(root, pilot_index))[0]
        for seed in manifest['seeds']:
            refs = []
            if seed == 42:
                source = pilot_entries[config['id'], agent, 42]
                refs.append(dict(path=str(reference_path(root, declared, source['file'])), sha256=source['sha256'],
                    targets=config['targets'], blocks=config['evaluation_blocks']))
            if manifest['purpose'] == 'test' and manifest['slot'] == 'shortened' and seed in range(42, 47):
                base = next(base for base in configurations() if base['id'] == config['registered_id'])
                source_id = setting(base, SHORTENED, mode, ['validation'])['id']
                source = catalog(Path(root) / 'screening')[0][source_id, SHORTENED, seed]
                refs.append(dict(path=str(Path(declared) / Path(source['file']).relative_to(root)), sha256=source['sha256'],
                    targets=config['targets'], blocks=['validation']))
            if refs:
                result['|'.join([config['id'], agent, str(seed)])] = refs
    return result


def authorization_manifest(manifest, registration_path=None):
    root = Path(manifest['authorization_plan']).parent
    if manifest['stage'] == 'authorized_cost_pilot':
        registration = load_plan(root / 'pilot-registration.json', manifest['authorization_sha256'])
        if manifest['pilot_index'] not in CORRECTED_PILOT_INDICES or registration['pilot_indices'] != CORRECTED_PILOT_INDICES:
            raise ValueError('the corrected pilot permits only C and default identity allocations')
        if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != registration['reference_index_sha256']:
            raise ValueError('pilot reference index changed')
        result = pilot_manifest(manifest['pilot_index'], root, root)
        if registration['manifests'][manifest['pilot_index']] != result:
            raise ValueError('pilot differs from its registered exact setting')
        scope_path = root / 'pilot-scope.json'
    else:
        if registration_path is None:
            raise ValueError('stage2 requires explicit plan and CPU-limit approval')
        registered_bytes = Path(__file__).with_name('plan.json').read_bytes()
        digest = hashlib.sha256(registered_bytes).hexdigest()
        plan = load_plan(registration_path, digest)
        if any(plan.get(key) != value for key, value in measurement_identity().items()):
            raise ValueError('registered measurement identity differs from the frozen pilot')
        approval = json.loads(Path(registration_path).with_name('approval.json').read_text())
        limit = approval.get('phase_cpu_hour_limit')
        if (approval.get('approved') is not True or approval.get('plan_sha256') != digest
                or type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError('stage2 requires explicit plan and CPU-limit approval')
        selection = saved_selection(root, plan) if manifest['purpose'] == 'test' else None
        result = dict(derive(plan, manifest['purpose'], selection, manifest.get('configuration_id'), manifest.get('axis'), manifest.get('slot')),
            experiment=EXPERIMENT, partition=plan['execution']['partition'], cpu_model=plan['execution']['cpu_model'],
            memory_mib=plan['forecast']['memory_mib'], concurrency=20, phase_cpu_hour_limit=limit,
            pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28', **measurement_identity())
        result['identity_references'] = study_references(result, root, root)
        scope_path = root / 'scope.json'
    if json.loads(scope_path.read_text()) != scope(manifest['authorization_sha256'], root):
        raise ValueError('all pilot, screening and conditional test roots must share one accounting scope')
    return dict(result, budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                stop_path=str(root / 'STOP.json'))


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
