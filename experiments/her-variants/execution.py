import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))
from grid_protocol import atomic_json, load_plan
from grid_attempts import select_attempts
from grid_identity import compare
from matched_control import configurations as controls
from protocol import AGENTS, VALUES
from tuning import select
from verify import verify

APPROVED_SHA256 = 'e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444'
MEASUREMENT_COMMIT = 'a592a5bad8bbc13d62553cc09a72748182809879'
BINARY_SHA256 = '83ebf9a69d3d42cd8f275bab9f77f306fd25e5bcb7a665cf4771d169ab2ac3f2'


def approved(path):
    plan = load_plan(path, APPROVED_SHA256)
    if plan['measurement_commit'] != MEASUREMENT_COMMIT or plan['binary_sha256'] != BINARY_SHA256:
        raise ValueError('measurement differs from the approved frozen source and binary')
    return plan


def label(proportion):
    if proportion == 1 / 3:
        return 'p1of3'
    if proportion == 2 / 3:
        return 'p2of3'
    raise ValueError('unregistered tuning proportion')


def tuning_manifest(plan):
    configs = []
    for config in plan['configurations']:
        for proportion in plan['tuning']['proportions']:
            item = copy.deepcopy(config)
            item.update(id=config['id'] + '_' + label(proportion), registered_id=config['id'],
                        batch_group=config['id'], evaluation_blocks=['validation'],
                        agent_settings={agent: {'relabeled_proportion': proportion} for agent in VALUES})
            configs.append(item)
    return dict(configurations=configs, agents=list(VALUES), seeds=list(plan['tuning']['seeds']),
                stage='tuning_pending_approval', batch_by='group')


def selected_proportions(selection):
    return {key: {agent: result['proportion'] for agent, result in values.items()}
            for key, values in selection['selections'].items()}


def grid_manifest(plan, selection):
    choices = selected_proportions(selection)
    controls(plan, choices)
    configs = copy.deepcopy(plan['configurations'])
    for config in configs:
        config['agent_settings'] = {agent: {'relabeled_proportion': choices[config['id']][agent]}
                                    for agent in VALUES}
    return dict(configurations=configs, agents=list(AGENTS), seeds=list(plan['seeds']),
                stage='approved_grid', batch_size=20)


def control_manifest(plan, selection, configuration_id):
    required = controls(plan, selected_proportions(selection))
    config = next((item for item in required if item['id'] == configuration_id), None)
    if config is None:
        raise ValueError('this maze reuses its final-grid values arm; no extra control is allowed')
    return dict(configurations=[config], agents=['her_values'], seeds=list(plan['seeds']),
                stage='approved_grid', batch_size=20, arm='her_values_p2of3')


def catalog(root, validate=True):
    root = Path(root)
    result = verify(root) if validate else None
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    ledger = json.loads((root / 'ledger.json').read_text())
    entries = {}
    for config in plan['configurations']:
        selected, _ = select_attempts(root, ledger, config)
        for (agent, seed), attempt in selected.items():
            path = attempt['path'] / 'rows.jsonl'
            entries[config['id'], agent, seed] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), targets=config['targets'],
                rows=[json.loads(line) for line in path.read_text().splitlines()])
    return entries, result


def pilot_catalog(base):
    base = Path(base)
    result, costs = {}, []
    for name in ['cost-pilot', 'baseline-regression', 'u1-observer-pilot']:
        entries, checked = catalog(base / name)
        costs.append(checked['allocation_cpu_seconds'])
        if name == 'baseline-regression':
            continue
        for key, entry in entries.items():
            if (key[1] == 'acs2er_u1') == (name == 'u1-observer-pilot'):
                result[key] = entry
    if sum(costs) != 5445:
        raise ValueError('completed pilots differ from the approved accounting')
    return result


def reference(entry, blocks):
    return {key: entry[key] for key in ['path', 'sha256', 'targets']} | {'blocks': blocks}


def references(manifest, pilots, tuning=None):
    result = {}
    for config in manifest['configurations']:
        base_id = config.get('registered_id', config['id'])
        for agent in manifest['agents']:
            for seed in manifest['seeds']:
                refs = []
                proportion = config.get('agent_settings', {}).get(agent, {}).get('relabeled_proportion')
                validation_only = config.get('evaluation_blocks') == ['validation']
                if seed == 42 and (agent not in VALUES or proportion == 2 / 3):
                    refs.append(reference(pilots[base_id, agent, seed], ['validation'] if validation_only else ['training', 'validation', 'test']))
                if tuning is not None and agent in VALUES and seed in range(42, 47):
                    refs.append(reference(tuning[base_id + '_' + label(proportion), agent, seed], ['validation']))
                if refs:
                    result['|'.join(map(str, [config['id'], agent, seed]))] = refs
    return result


def choose(plan, tuning_entries):
    results = {}
    for config in plan['configurations']:
        results[config['id']] = {}
        for agent in VALUES:
            rows = [row for p in plan['tuning']['proportions'] for seed in plan['tuning']['seeds']
                    for row in tuning_entries[config['id'] + '_' + label(p), agent, seed]['rows']]
            results[config['id']][agent] = select(rows, plan['tuning']['proportions'], config['targets'])
    return dict(approved_plan_sha256=APPROVED_SHA256, selections=results,
                rule=plan['tuning']['selection'], seeds=plan['tuning']['seeds'],
                evaluation_blocks=['validation'], test_access='none')


def resolve(manifest, plan, study_root, operations_manifest, operations_commit):
    root = Path(study_root)
    scope = root / 'scope.json'
    result = dict(manifest, experiment='her-variants', partition='lem-cpu-normal',
        cpu_model=plan['execution']['cpu_model'], memory_mib=plan['forecast']['memory_mib'],
        concurrency=20, phase_cpu_hour_limit=60, pilot_allocation_cpu_seconds=5445,
        job_name='tu-her-variants',
        accounting_since='2026-10-12', measurement_commit=MEASUREMENT_COMMIT,
        binary_sha256=BINARY_SHA256, preset=plan['preset'],
        authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        budget_scope=str(scope), budget_scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest(),
        stop_path=str(root / 'STOP.json'), operations_commit=operations_commit,
        operations_manifest=str(operations_manifest),
        operations_manifest_sha256=hashlib.sha256(Path(operations_manifest).read_bytes()).hexdigest())
    return result


def create_scope(plan, root):
    root = Path(root)
    roots = [root / 'tuning', root / 'main-grid'] + [root / ('control-' + config['id']) for config in plan['configurations']]
    scope = dict(authorization_sha256=APPROVED_SHA256, pilot_allocation_cpu_seconds=5445,
                 accounting_since='2026-10-12', batch_roots=list(map(str, roots)))
    atomic_json(root / 'scope.json', scope)
    return scope
