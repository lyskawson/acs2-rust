import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from pilot_equivalence import compare
from protocol import AGENTS, ROOT, configurations


def calibrate(binary, pilot_root, output):
    binary, pilot_root, output = Path(binary).resolve(), Path(pilot_root), Path(output).resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('use a new calibration directory outside the repository')
    pilot_rows = [json.loads(line) for line in (pilot_root / 'pilot-rows.jsonl').read_text().splitlines()]
    if subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('calibrate the completed clean measurement source')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    old = {(row['task'], row['agent'], row['nominal_step']): row for row in pilot_rows if row['agent'].endswith('_uniform')}
    output.mkdir(parents=True)
    cells, files = [], {}
    started = time.monotonic()
    for config in configurations():
        for agent in AGENTS[2:]:
            path = output / f"{config['id']}_{agent}.jsonl"
            command = [str(binary), '--task', config['task'], '--cap', str(config['cap']),
                       '--protocol', 'maze-pairs-v1', '--agents', agent, '--seeds', '42',
                       '--targets', '1000,3000,30000', '--out', str(path)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=90)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if [row['nominal_step'] for row in rows] != [1000, 3000, 30000]:
                raise ValueError('calibration has missing registered checkpoints')
            for row in rows:
                if row['commit'] != commit or row['source_state'] != 'clean':
                    raise ValueError('native calibration binary does not identify the clean source commit')
                compare(old[config['task'], agent, row['nominal_step']], row)
            row = rows[-1]
            previous = old[config['task'], agent, row['nominal_step']]
            blocks = row['evaluation_blocks']
            cell = dict(configuration_id=config['id'], task=config['task'], agent=agent, seed=42,
                        nominal_step=row['nominal_step'], actual_steps=row['actual_steps'],
                        pilot_classifier_tests=previous['classifier_perception_tests_train'],
                        revised_classifier_tests=row['classifier_perception_tests_train'],
                        avoided_classifier_tests=row['replay_diagnostics']['priority']['avoided_classifier_perception_tests'],
                        ratio=row['classifier_perception_tests_train'] / previous['classifier_perception_tests_train'],
                        all_block_disagreement_transitions=sum(block['head_disagreement']['count'] for block in blocks.values()),
                        additional_policy_pairs=sum(block['evaluation_pairs'] for name, block in blocks.items() if name != 'validation'))
            cells.append(cell)
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(json.dumps(cell), flush=True)
    summary = dict(purpose='Sequential short deterministic work calibration; no policy ranking or temperature selection',
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   measurement_commit=commit,
                   pilot_rows_sha256=hashlib.sha256((pilot_root / 'pilot-rows.jsonl').read_bytes()).hexdigest(),
                   elapsed_wall_seconds=time.monotonic() - started, cells=cells, files=files)
    (output / 'calibration.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--pilot-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    calibrate(args.binary, args.pilot_root, args.out)
