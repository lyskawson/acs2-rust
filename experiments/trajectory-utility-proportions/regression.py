import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

from proportions import BLOCKS, MEAN, RANDOM, S1, checksum, read, registered
from grid_identity import compare

EXISTING_ARMS = (
    'acs2', 'acs2_per_goal_reward', 'acs2er', 'acs2er_one_update', 'acs2er_episodic',
    'acs2er_episodic_per_goal_reward', 'acs2her', 'acs2her_nontrivial',
    'acs2her_shortened_goal_rules', 'acs2her_shortened_random_rules', S1, 'acs2her_shortened',
    'acs2her_values', 'acs2her_values_shortened', 'acs2vcp_k2', 'acs2vcp_k4',
    'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform', MEAN, RANDOM, 'acs2tuher_trial_update',
)


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def deterministic_registered(original):
    row = copy.deepcopy(original)
    row.pop('test_caps', None)
    if 'trajectory_utility' in row:
        cost = row['trajectory_utility']['cost']
        for field in tuple(cost):
            if field.endswith('cpu_seconds'):
                del cost[field]
    return row


def run(binary, path, cfg, arms, targets, flags=()):
    command = [str(binary), '--task', cfg['task'], '--cap', str(cfg['cap']),
               '--pool', cfg['pool'], '--encoding', cfg['encoding'], '--protocol', cfg['protocol'],
               '--agents', ','.join(arms), '--seeds', '42', '--targets', ','.join(map(str, targets)),
               '--per-start', '--out', str(path), *flags]
    subprocess.run(command, check=True, capture_output=True)
    return rows(path)


def compare_rows(actual, expected, targets):
    return compare(list(map(deterministic_registered, actual)), list(map(deterministic_registered, expected)),
                   [42], targets, BLOCKS)


def file_index(path, agents):
    result = {}
    for entry in read(path):
        if entry['agent'] not in agents or entry['seed'] != 42:
            continue
        key = entry['task'], entry['agent']
        if key in result:
            raise ValueError('duplicate archive source')
        result[key] = entry
    return result


def verify(binary, ablation_index, measures_index):
    _, _, configs, _ = registered()
    identity = []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for archive, arms in ((ablation_index, (S1,)), (measures_index, (MEAN, RANDOM))):
            index = file_index(archive, arms)
            for (task, agent), entry in index.items():
                if task not in configs:
                    raise ValueError('unregistered archive maze')
                path = Path(entry['source'])
                if checksum(path) != entry['sha256']:
                    raise ValueError(f'archive row checksum differs: {path}')
                original = rows(path)
                targets = [configs[task]['targets'][0]]
                first = original[0]
                if first['nominal_step'] != targets[0] or not all(first['evaluation_blocks'][name]['starts'] for name in BLOCKS):
                    raise ValueError('the first archive checkpoint or its pair records are missing')
                measured_agent, share = (MEAN, '0') if agent == S1 else (agent, '0.5')
                flags = ('--tu-share', share, '--tu-candidates', '8', '--tu-refresh', '2000')
                actual = run(binary, folder / 'archive-check.jsonl', configs[task], [measured_agent], targets, flags)
                comparison = compare_rows(actual, original, targets)
                identity.append(dict(task=task, agent=agent, path=str(path), sha256=entry['sha256'], **comparison))
            if set(index) != {(task, arm) for task in configs for arm in arms}:
                raise ValueError('archive coverage differs from the registration')
        cfg = configs['maze4']
        targets = [100, 300, 1000]
        off = run(binary, folder / 'existing-short.jsonl', cfg, EXISTING_ARMS, targets)
        on = run(binary, folder / 'existing-extended.jsonl', cfg, EXISTING_ARMS, targets, ('--second-test-cap', '100'))
        for agent in EXISTING_ARMS:
            actual = [row for row in on if row['agent'] == agent]
            expected = [row for row in off if row['agent'] == agent]
            if not all(row['test_caps']['extended']['cap'] == 100 for row in actual):
                raise ValueError('extended test results missing')
            compare_rows(actual, expected, targets)
    return dict(archive_rows=len(identity), identity=identity, existing_presets=list(EXISTING_ARMS),
                registered_rows_with_and_without_cap=len(EXISTING_ARMS) * len(targets),
                excluded_timing_fields='wall time and utility or extended evaluation CPU; build and host metadata')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--ablation-index', required=True, type=Path)
    parser.add_argument('--measures-index', required=True, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.binary.resolve(), args.ablation_index, args.measures_index)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('archive rows:', result['archive_rows'])
    print('registered rows with and without second cap:', result['registered_rows_with_and_without_cap'])
