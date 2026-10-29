import argparse
import hashlib
import json
from pathlib import Path
import shutil

from approved_execution import approved
from protocol import BASE_PROPORTIONS, ER, PROPORTION_SHA256, SHORTENED, label
from agent_names import canonical_run_name
from collect import outside_repository, validate_row
from grid_attempts import attempts, inspect_attempt, select_attempts
from grid_identity import compare
from grid_protocol import allocation_rows, atomic_json, load_plan


def selected(batch, registration, digest, config, agent, proportion=None):
    batch = Path(batch)
    source = load_plan(registration, digest)
    manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    if any(manifest[key] != source[key] for key in ('measurement_commit', 'binary_sha256', 'preset')):
        raise ValueError('reused source, binary or preset differs from its registration')
    configs = [value for value in manifest['configurations'] if value.get('registered_id', value['id']) == config['id']]
    if len(configs) != 1:
        raise ValueError('reused pair configuration is ambiguous')
    archived = configs[0]
    for key in ('task', 'cap', 'targets', 'registration', 'budget_steps', 'encoding', 'row_pool', 'protocol'):
        if archived[key] != config[key]:
            raise ValueError('reused pair configuration differs: ' + key)
    names = [name for name in manifest['agents'] if canonical_run_name(name) == agent]
    if len(names) != 1:
        raise ValueError('reused thesis agent name is ambiguous')
    name = names[0]
    ledger = json.loads((batch / 'ledger.json').read_text())
    choices, _ = select_attempts(batch, ledger, archived)
    actual = {seed: entry for (kind, seed), entry in choices.items() if kind == name}
    if set(actual) != set(range(42, 62)):
        raise ValueError('reference requires exactly twenty complete seeds')
    accounting = {row['JobID']: row for row in allocation_rows((batch / 'accounting.psv').read_text())}
    result = {}
    for seed, entry in actual.items():
        path = entry['path']
        rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
        if len(rows) != len(config['targets']):
            raise ValueError('reference has missing registered points')
        for row, point in zip(rows, config['targets']):
            validate_row(row, archived, name, seed, source['measurement_commit'], point)
            if row['preset'] != source['preset'] or set(row['evaluation_blocks']) != {'training', 'validation', 'test'}:
                raise ValueError('reference preset or evaluation blocks differ')
            if proportion is not None and row['agent_parameters']['relabeled_proportion'] != proportion:
                raise ValueError('reference is not the saved per-maze copy proportion')
        record = accounting[f"{entry['array_id']}_{entry['task_id']}"]
        if record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0' or record['AllocCPUS'] != '1':
            raise ValueError('reference allocation did not complete on one CPU')
        result[seed] = dict(path=path, rows=rows, record=record,
            source_registration_sha256=digest, source_manifest_sha256=(batch / 'plan.sha256').read_text().strip())
    for entry in attempts(batch, ledger):
        if entry['configuration_id'] != archived['id'] or entry['agent'] != name:
            continue
        _, partial = inspect_attempt(entry, archived)
        if partial:
            compare(result[entry['seed']]['rows'], partial, [entry['seed']],
                    [row['nominal_step'] for row in partial], ['training', 'validation', 'test'])
    return result


def prepare(root, proportion, her, baseline):
    root = outside_repository(root)
    plan = approved(root / 'approved-plan.json')
    proportion, her, baseline = Path(proportion), Path(her), Path(baseline)
    selections = json.loads((proportion / 'selections.json').read_text())['selections']
    output = root / 'analysis-references'
    output.mkdir(exist_ok=False)
    index = []
    for config in plan['configurations']:
        p = BASE_PROPORTIONS[config['task']]
        if selections[config['id']]['proportion'] != p:
            raise ValueError('reference proportion differs from the registered saved selection')
        sources = [
            (label(config, SHORTENED, 'base'), SHORTENED, selected(proportion / ('test-' + config['id']),
                proportion / 'approved-plan.json', PROPORTION_SHA256, config, SHORTENED, p)),
            (ER, ER, selected(her / 'main-grid', her / 'approved-plan.json', plan['references']['her_plan_sha256'], config, ER)),
            *[(name, name, selected(baseline, baseline / 'plan.json', plan['references']['baseline_plan_sha256'], config, name))
              for name in ('acs2', 'acs2er')],
        ]
        for arm, agent, values in sources:
            for seed, entry in values.items():
                relative = Path('cells') / config['id'] / arm / f's{seed}'
                directory = output / relative
                directory.mkdir(parents=True)
                checksums = {}
                for name in ('rows.jsonl', 'completion.json'):
                    shutil.copyfile(entry['path'] / name, directory / name)
                    checksums[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                index.append(dict(configuration_id=config['id'], arm=arm, agent=agent, seed=seed,
                    directory=str(relative), checksums=checksums, source=str(entry['path']),
                    source_registration_sha256=entry['source_registration_sha256'],
                    source_manifest_sha256=entry['source_manifest_sha256'],
                    allocation_cpu_seconds=int(entry['record']['CPUTimeRAW']), job_id_raw=entry['record']['JobIDRaw'],
                    reused=True, charged_to_this_study=False))
    atomic_json(output / 'index.json', index)
    return dict(runs=len(index), index_sha256=hashlib.sha256((output / 'index.json').read_bytes()).hexdigest(),
                new_allocations=0)


def read(root):
    plan = approved(Path(root) / 'approved-plan.json')
    root = Path(root) / 'analysis-references'
    result = {}
    for entry in json.loads((root / 'index.json').read_text()):
        directory = root / entry['directory']
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError('reference directory is outside its frozen collection')
        for name, digest in entry['checksums'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError('frozen reused reference changed')
        key = entry['configuration_id'], entry['arm'], entry['seed']
        if key in result:
            raise ValueError('duplicate reused reference')
        result[key] = dict(entry, path=directory)
    expected = {(config['id'], arm, seed) for config in plan['configurations']
                for arm in (label(config, SHORTENED, 'base'), ER, 'acs2', 'acs2er') for seed in range(42, 62)}
    if set(result) != expected:
        raise ValueError('reused references differ from six mazes, four registered arms and twenty seeds')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--proportion', required=True)
    parser.add_argument('--her', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.proportion, args.her, args.baseline)))
