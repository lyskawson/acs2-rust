import hashlib
import json
import math
from pathlib import Path
import sys

from protocol import BLOCKS, HER, SHORTENED, pilot, pilot_roots, scope_roots, setting
from grid_protocol import load_plan
from grid_identity import compare, field_bits
from tuning import select
from references import read_index


def measurement_identity():
    identity = json.loads(Path(__file__).with_name('measurement.json').read_text())
    if len(identity.get('measurement_commit', '')) != 40 or len(identity.get('binary_sha256', '')) != 64:
        raise ValueError('measurement identity is not frozen')
    return identity


def identity_references(root, declared, config, agent, seed, blocks, source):
    return dict(path=str(Path(declared) / source['file']), sha256=source['sha256'], targets=config['targets'], blocks=blocks)


def pilot_manifest(index, root, declared):
    result = pilot(index) | measurement_identity()
    refs = read_index(root)
    result['identity_references'] = {}
    if index == 4:
        for config in result['configurations']:
            result['identity_references']['|'.join([config['id'], SHORTENED, '42'])] = [
                identity_references(root, declared, config, SHORTENED, 42, ['validation'], refs[config['registered_id'], 42])]
    return result


def scope(digest, root):
    return dict(authorization_sha256=digest, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28',
                batch_roots=[str(Path(root) / name) for name in scope_roots()])


def derive(plan, purpose, selection=None, configuration_id=None):
    forecast = plan['forecasts']['with_four_fifths']
    if purpose == 'tuning':
        configs = []
        for config in plan['configurations']:
            for p in plan['candidates']:
                cell = next(cell for cell in forecast['cells'] if (cell['purpose'], cell['configuration_id'], cell['proportion']) == ('tuning', config['id'], p))
                item = setting(config, SHORTENED, p, ['validation'], cell['limit_minutes'])
                configs.append(dict(item, batch_group=cell['batch_group']))
        return dict(stage='tuning_pending_approval', purpose=purpose, configurations=configs,
                    agents=[SHORTENED], seeds=list(plan['tuning']['seeds']), batch_by='group')
    if purpose == 'control':
        configs = []
        for config in plan['configurations']:
            cell = next(cell for cell in forecast['cells'] if (cell['purpose'], cell['configuration_id']) == ('control', config['id']))
            configs.append(setting(config, HER, 0.5, BLOCKS, cell['limit_minutes']))
        return dict(stage='approved_grid', purpose=purpose, configurations=configs,
                    agents=[HER], seeds=list(plan['control']['seeds']), batch_size=20)
    if purpose != 'test' or selection is None:
        raise ValueError('test manifests require the saved validation selection')
    config = next((config for config in plan['configurations'] if config['id'] == configuration_id), None)
    if config is None:
        raise ValueError('unregistered test maze')
    p = selection['selections'][config['id']]['proportion']
    if p not in plan['candidates'] or p == 2 / 3:
        raise ValueError('unregistered p or coincident arm: reuse reference instead')
    cell = next(cell for cell in forecast['cells'] if cell['configuration_id'] == config['id'] and cell['purpose'] == 'test')
    return dict(stage='approved_grid', purpose=purpose, configuration_id=config['id'],
                configurations=[setting(config, SHORTENED, p, BLOCKS, cell['reservation_cpu_seconds'] // (20 * 60))],
                agents=[SHORTENED], seeds=list(plan['test']['seeds']), batch_size=20)


def choose(plan, entries, references):
    result = {}
    for config in plan['configurations']:
        for seed in plan['tuning']['seeds']:
            current = entries[setting(config, SHORTENED, 2 / 3, ['validation'])['id'], SHORTENED, seed]['rows']
            compare(current, references[config['id'], seed]['rows'], [seed], config['targets'], ['validation'])
        rows = [row for p in plan['candidates'] for seed in plan['tuning']['seeds']
                for row in entries[setting(config, SHORTENED, p, ['validation'])['id'], SHORTENED, seed]['rows']]
        result[config['id']] = select(rows, plan['candidates'], config['targets'])
    return dict(selections=result, candidates=list(plan['candidates']), seeds=list(plan['tuning']['seeds']),
                rule=plan['tuning']['selection'], evaluation_blocks=['validation'], test_access='none')


def registered_references(plan, root):
    data = (Path(root) / 'reference-index.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['cost_pilot']['references_sha256']:
        raise ValueError('references differ from the registered pilot archive')
    return read_index(root)


def study_references(plan, manifest, root, declared):
    from archive import catalog
    references = registered_references(plan, root)
    pilot_entries = {}
    for name in pilot_roots():
        entries, _ = catalog(Path(root) / name)
        for (config_id, agent, seed), entry in entries.items():
            row = entry['rows'][0]
            key = row['task'], agent, row['agent_parameters']['relabeled_proportion'], seed
            pilot_entries[key] = entry
    tuning_entries = catalog(Path(root) / 'tuning')[0] if manifest['purpose'] == 'test' else {}
    result = {}
    for config in manifest['configurations']:
        agent = manifest['agents'][0]
        p = config['agent_settings'][agent]['relabeled_proportion']
        for seed in manifest['seeds']:
            refs = []
            if manifest['purpose'] == 'tuning' and p == 2 / 3:
                refs.append(identity_references(root, declared, config, agent, seed, ['validation'], references[config['registered_id'], seed]))
            if seed == 42:
                entry = pilot_entries.get((config['task'], agent, p, seed))
                if entry:
                    path = Path(entry['file']).relative_to(root)
                    blocks = ['validation'] if manifest['purpose'] == 'tuning' else BLOCKS
                    refs.append(dict(path=str(Path(declared) / path), sha256=entry['sha256'], targets=config['targets'], blocks=blocks))
            if manifest['purpose'] == 'test' and seed in range(42, 47):
                source_id = setting(configuration_by_id(plan, config['registered_id']), SHORTENED, p, ['validation'])['id']
                entry = tuning_entries[source_id, SHORTENED, seed]
                refs.append(dict(path=str(Path(declared) / Path(entry['file']).relative_to(root)),
                    sha256=entry['sha256'], targets=config['targets'], blocks=['validation']))
            if refs:
                result['|'.join([config['id'], agent, str(seed)])] = refs
    return result


def configuration_by_id(plan, identifier):
    return next(config for config in plan['configurations'] if config['id'] == identifier)


def approved(path):
    path = Path(path)
    registered = Path(__file__).with_name('plan.json').read_bytes()
    plan = load_plan(path, hashlib.sha256(registered).hexdigest())
    approval = json.loads(path.with_name('approval.json').read_text())
    if (approval.get('approved') is not True or approval.get('plan_sha256') != hashlib.sha256(registered).hexdigest()
            or type(approval.get('phase_cpu_hour_limit')) not in (int, float)
            or not math.isfinite(approval['phase_cpu_hour_limit']) or approval['phase_cpu_hour_limit'] <= 0):
        raise ValueError('explicit approval of registered plan bytes and cumulative CPU limit is required')
    if any(plan.get(key) != value for key, value in measurement_identity().items()):
        raise ValueError('registered measurement identity differs')
    return plan, approval


def authorization_manifest(manifest, registration_path=None):
    declared = Path(manifest['authorization_plan']).parent
    if manifest['stage'] == 'authorized_cost_pilot':
        root = declared
        registration = load_plan(root / 'pilot-registration.json', manifest['authorization_sha256'])
        if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != registration['reference_index_sha256']:
            raise ValueError('pilot references changed')
        result = pilot_manifest(manifest['pilot_index'], root, declared)
        if registration['manifests'][manifest['pilot_index']] != result:
            raise ValueError('pilot registration changed')
        scope_path = root / 'pilot-scope.json'
        if json.loads(scope_path.read_text()) != scope(manifest['authorization_sha256'], declared):
            raise ValueError('pilot must account for all study roots')
        return dict(result, budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
                    stop_path=str(declared / 'STOP.json'))
    if registration_path is None:
        raise ValueError('approved study registration is required')
    plan, approval = approved(registration_path)
    root = Path(registration_path).parent
    selection = None
    if manifest.get('purpose') == 'test':
        from archive import catalog
        from grid_authorization import require_derivation
        tuning_manifest = load_plan(root / 'tuning/plan.json', (root / 'tuning/plan.sha256').read_text().strip())
        require_derivation(tuning_manifest, authorization_manifest(tuning_manifest, registration_path))
        entries = catalog(root / 'tuning')[0]
        selection = json.loads((root / 'selections.json').read_text())
        if field_bits(selection) != field_bits(choose(plan, entries, registered_references(plan, root))):
            raise ValueError('saved selection differs from complete validation tuning and identity')
    result = derive(plan, manifest.get('purpose'), selection, manifest.get('configuration_id'))
    scope_path = root / 'scope.json'
    if json.loads(scope_path.read_text()) != scope(manifest['authorization_sha256'], declared):
        raise ValueError('study scope must include every pilot, tuning, test and control root')
    return dict(result, experiment='hindsight-proportion', partition=plan['execution']['partition'],
        cpu_model=plan['execution']['cpu_model'], memory_mib=plan['forecasts']['with_four_fifths']['memory_mib'],
        concurrency=20, phase_cpu_hour_limit=approval['phase_cpu_hour_limit'], pilot_allocation_cpu_seconds=0,
        accounting_since='2026-10-28', identity_references=study_references(plan, result, root, declared), **measurement_identity())


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
