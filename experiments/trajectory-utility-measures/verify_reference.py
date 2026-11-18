import argparse
import json
from pathlib import Path
import subprocess

from registration import ablation, checksum, OLD_TASKS, read, S1
from grid_identity import compare


def verify(binary, index, output):
    _, inherited = ablation()
    refs = {(entry['task'], entry['seed']): entry for entry in read(index) if entry['agent'] == S1}
    tasks = (*OLD_TASKS, 'littman57_alcs')
    target = 1000
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for config in inherited['configurations']:
        task = config['task']
        if task not in tasks:
            continue
        entry = refs[task, 42]
        original = Path(entry['source'])
        if checksum(original) != entry['sha256']:
            raise ValueError(f'original S+1 checksum differs for {task}')
        path = output / f'{task}.jsonl'
        subprocess.run([
            str(binary.resolve()), '--task', task, '--cap', str(config['cap']), '--pool', config['pool'],
            '--encoding', config['encoding'], '--protocol', config['protocol'], '--agents', S1,
            '--seeds', '42', '--targets', str(target), '--per-start', '--out', str(path),
        ], check=True)
        with original.open() as stream:
            reference = json.loads(next(stream))
        current = json.loads(path.read_text())
        if any(not block['starts'] for block in current['evaluation_blocks'].values()):
            raise ValueError('required original per-pair records are missing')
        result = compare([current], [reference], [42], [target], config['evaluation_blocks'])
        results.append(dict(task=task, reference=str(original), reference_sha256=entry['sha256'],
                            current=str(path), current_sha256=checksum(path), **result))
    if {row['task'] for row in results} != set(tasks):
        raise ValueError('the six original mazes and one ALCS maze must all be checked')
    (output / 'identity.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'{len(results)} archived S+1 checkpoint and per-pair identities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    verify(args.binary, args.index, args.out)
