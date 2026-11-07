import argparse
import hashlib
import json
from pathlib import Path
import shutil

from protocol import ROOT, REFERENCES, E, S, G, R, configurations, recorded
from agent_names import canonical_run_name
from collect import outside_repository, validate_row
from grid_attempts import attempts
from grid_protocol import allocation_rows, atomic_json


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_cost(source):
    source = Path(source)
    completion = json.loads(source.with_name('completion.json').read_text())
    if completion['exit_code'] != 0 or completion['failure']:
        raise ValueError('reference cost needs a complete original attempt')
    root = next((parent for parent in source.parents if (parent / 'ledger.json').exists()), None)
    if root is None:
        raise ValueError('original reference allocation ledger is missing')
    matched = [entry for entry in attempts(root, json.loads((root / 'ledger.json').read_text()))
               if entry['path'].resolve() == source.parent.resolve()]
    if len(matched) != 1:
        raise ValueError('reference cost needs one exact original attempt')
    job = f"{matched[0]['array_id']}_{matched[0]['task_id']}"
    accounts = {row['JobIDRaw']: row for row in allocation_rows((root / 'accounting.psv').read_text()) if row['JobID'] == job}
    if len(accounts) != 1:
        raise ValueError('reference cost needs one deduplicated allocation')
    account = next(iter(accounts.values()))
    if account['State'] != 'COMPLETED' or account['AllocCPUS'] != '1':
        raise ValueError('reference allocation did not complete on one CPU')
    return dict(allocation_cpu_seconds=int(account['CPUTimeRAW']), body_wall_seconds=completion['elapsed_seconds'],
        process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
        peak_rss_kib=completion['peak_rss_kib'], job_id_raw=account['JobIDRaw'])


def freeze(output, goal_rules, random_rules):
    output, goal_rules, random_rules = outside_repository(output), Path(goal_rules), Path(random_rules)
    if (random_rules / 'study').is_dir():
        random_rules = random_rules / 'study'
    sources = []
    for root, study, arm in [(goal_rules, 'goal-rule-copies', G), (random_rules, 'reward-update-dose-control', R)]:
        registered = ROOT / 'experiments' / study / 'plan.json'
        verified = json.loads((root / 'verification.json').read_text())
        if ((root / 'approved-plan.json').read_bytes() != registered.read_bytes() or verified.get('verified') is not True
                or verified['plan_sha256'] != checksum(registered)):
            raise ValueError('reference study is not the original verified approved archive')
        if arm == R:
            sources.extend(json.loads((root / 'reference-index.json').read_text()))
        for entry in verified['runs_index']:
            if entry['agent'] != arm:
                raise ValueError('reference verification has a swapped arm')
            source = root / entry['file']
            for kind in ['success'] + (['pairs'] if entry['seed'] < 47 else []):
                sources.append(dict(kind=kind, configuration_id=entry['configuration_id'], agent=arm,
                    seed=entry['seed'], source=str(source), sha256=entry['sha256'], charged_to_this_study=False))
    unique = {}
    for entry in sources:
        key = entry['kind'], entry['configuration_id'], entry['agent'], entry['seed']
        if key in unique and (unique[key]['source'] != entry['source'] or unique[key]['sha256'] != entry['sha256']):
            raise ValueError('conflicting original reference provenance')
        unique[key] = entry
    expected = {(kind, config['id'], arm, seed) for config in configurations() for arm in REFERENCES
                for kind, seeds in [('success', range(42, 62)), ('pairs', range(42, 47))] for seed in seeds}
    if set(unique) != expected:
        raise ValueError('references require four archived arms, six mazes, twenty success and five pair seeds')
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for (kind, identifier, arm, seed), entry in sorted(unique.items()):
        source = Path(entry['source'])
        if checksum(source) != entry['sha256']:
            raise ValueError('original reference rows changed')
        config = recorded(next(config for config in configurations() if config['id'] == identifier))
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if [row['nominal_step'] for row in rows] != config['targets']:
            raise ValueError('reference has different checkpoints')
        if rows[0]['evaluation_blocks']['training']['starts'] is None:
            del config['per_start_diagnostics']
            if kind == 'pairs':
                raise ValueError('reference pair records are missing')
        for row, point in zip(rows, config['targets']):
            validate_row(row, config, arm, seed, row['commit'], point)
            if canonical_run_name(row['agent']) != arm:
                raise ValueError('reference arm differs')
        relative = Path('data') / (entry['sha256'] + '.jsonl')
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if checksum(target) != entry['sha256']:
            raise ValueError('frozen reference bytes differ from the original')
        record = dict(kind=kind, configuration_id=identifier, agent=arm, seed=seed,
            file=str(relative), source=str(source), sha256=entry['sha256'], charged_to_this_study=False)
        if kind == 'pairs' and seed == 42 and arm in (E, S):
            record['cost'] = source_cost(source)
        index.append(record)
    atomic_json(output / 'reference-index.json', index)
    return dict(files=len({entry['file'] for entry in index}), entries=len(index), reused_success_runs=480,
                reused_pair_runs=120, source_bytes_preserved=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--goal-rules', required=True)
    parser.add_argument('--random-rules', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out, args.goal_rules, args.random_rules)))
