import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = dict(maze4='0.6245', maze5='0.6321', maze6='0.6425', maze7='0.6194', mazema='0.7887', woods14='0.7964')
G = 'acs2her_shortened_goal_rules'


def derive(root):
    root = Path(root)
    verification = json.loads((root / 'verification.json').read_text())
    if verification.get('verified') is not True or verification.get('failed_attempts') != 0:
        raise ValueError('require the verified original G archive')
    sums = {maze: dict(with_goal=0, skipped_without_goal=0) for maze in EXPECTED}
    seen = set()
    for entry in verification['runs_index']:
        path = root / entry['file']
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('G source checksum or path differs')
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        row = rows[-1]
        key = row['task'], row['seed']
        if key in seen or row['agent'] != G or entry['agent'] != G or entry['seed'] != row['seed']:
            raise ValueError('duplicated or inconsistent G source')
        seen.add(key)
        if row['nominal_step'] != (4_000_000 if row['task'] == 'woods14' else 2_000_000):
            raise ValueError('G dose requires the final registered point')
        counts = row['replay_diagnostics']['reward_updates']['relabeled']
        for field in sums[row['task']]:
            if type(counts[field]) is not int or counts[field] < 0:
                raise ValueError('G dose requires nonnegative integer counters')
            sums[row['task']][field] += counts[field]
    if seen != {(maze, seed) for maze in EXPECTED for seed in range(42, 62)}:
        raise ValueError('G dose requires six mazes and all twenty seeds')
    for maze, counts in sums.items():
        probability = counts['skipped_without_goal'] / sum(counts.values())
        if format(probability, '.4f') != EXPECTED[maze]:
            raise ValueError('STOP: archived dose disagrees with the card: ' + maze)
        counts.update(skip_probability=probability, binary64_hex=probability.hex())
    return dict(source_study='goal-rule-copies', source_plan_sha256=verification['plan_sha256'],
                seeds=list(range(42, 62)), origin='Relabeled', point='final', mazes=sums)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(derive(args.archive), indent=2, sort_keys=True) + '\n')
