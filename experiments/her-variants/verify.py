import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid'))
from collect import validate_row
from grid_attempts import select_attempts
from grid_protocol import allocation_rows, load_plan, TERMINAL


def verify(root):
    root = Path(root)
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    ledger = json.loads((root / 'ledger.json').read_text())
    records = allocation_rows((root / 'accounting.psv').read_text())
    known = {row['JobID']: row for row in records}
    allocation_seconds = rows_count = 0
    cells, hashes = {}, {}
    for config in plan['configurations']:
        selected, inventory = select_attempts(root, ledger, config)
        if set(selected) != {(agent, seed) for agent in plan['agents'] for seed in plan['seeds']}:
            raise ValueError('incomplete pilot or grid')
        for attempt in inventory:
            job = f"{attempt['array_id']}_{attempt['task_id']}"
            record = known.get(job)
            if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
                raise ValueError('nonterminal or missing accounting')
            allocation_seconds += int(record['CPUTimeRAW'])
            if record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
                raise ValueError('wrong allocation partition or CPU count')
            if attempt['selected'] and (record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0'):
                raise ValueError('selected attempt did not complete successfully')
        for (agent, seed), attempt in selected.items():
            path = attempt['path']
            rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, agent, seed, plan['measurement_commit'], target)
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                    raise ValueError('row preset or hardware differs')
            rows_count += len(rows)
            completion = json.loads((path / 'completion.json').read_text())
            record = known[f"{attempt['array_id']}_{attempt['task_id']}"]
            cells[config['id'], agent, seed] = dict(body_wall_seconds=rows[-1]['wall_seconds_total'],
                allocation_cpu_seconds=int(record['CPUTimeRAW']), process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                peak_rss_kib=completion['peak_rss_kib'], final_population=rows[-1]['population_classifiers'],
                maximum_population=max(row['population_classifiers'] for row in rows))
            for name in ['rows.jsonl','resources.jsonl','cpu_samples.csv','launch.json','completion.json']:
                file = path / name
                hashes[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return dict(runs=len(cells), rows=rows_count, allocation_cpu_seconds=allocation_seconds,
                cells=[dict(configuration_id=key[0], agent=key[1], seed=key[2], **value) for key, value in cells.items()], hashes=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    (args.root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['runs','rows','allocation_cpu_seconds']}))
