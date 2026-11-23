from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GRID = HERE.parent / 'grid'
ABLATION = HERE.parent / 'per-goal-reward-ablation'
MEASURES = HERE.parent / 'trajectory-utility-measures'
sys.path.extend(map(str, (GRID, ABLATION, MEASURES)))

from registration import MEAN, RANDOM, S1
from schema import BLOCKS

STUDY = HERE.name
SHARES = (0.25, 0.75, 1.0)
MEASURES_BY_ARM = (MEAN, RANDOM)
PILOT_SECONDS = 40 * 3600
CONCURRENCY = 40
MINIMUM_STEPS = 300_000
SCOUTS = ('mazee2_alcs', 'mazee3_alcs')
ATTEMPT_SPREAD = 1.41


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def label(agent, share):
    return f'{agent}_tu{int(share * 100)}'


def registered():
    study = read(HERE / 'plan.json')
    for directory, key in ((ABLATION, 'ablation_sha256'), (MEASURES, 'measures_sha256')):
        if any(checksum(directory / name) != digest for name, digest in study[key].items()):
            raise ValueError(f'inherited registration differs: {directory.name}')
    base = read(ABLATION / 'plan.json')
    configs = {config['task']: config for config in base['configurations']}
    arms = {label(agent, share): (agent, share) for agent in MEASURES_BY_ARM for share in SHARES}
    if (len(configs) != 28 or study['seeds'] != list(range(42, 62)) or study['seeds'] != base['seeds']
            or study['per_pair_seeds'] != list(range(42, 47)) or study['per_pair_seeds'] != base['per_pair_seeds']
            or [(arm['label'], arm['agent'], arm['tu_share']) for arm in study['arms']] !=
            [(key, agent, {0.25: '1/4', 0.75: '3/4', 1.0: '1'}[share])
             for key, (agent, share) in arms.items()]
            or [(row['agent'], row['tu_share']) for row in study['references']] !=
            [(S1, '0'), (MEAN, '1/2'), (RANDOM, '1/2')]
            or study['new_runs'] != 3360 or study['second_test_cap']['steps'] != 100
            or study['pilot']['minimum_steps'] != MINIMUM_STEPS
            or study['pilot']['cpu_hour_limit_including_reservations'] * 3600 != PILOT_SECONDS
            or study['pilot']['max_running_jobs'] != CONCURRENCY or study['pilot']['seed'] != 42
            or study['pilot']['cells'] != 168):
        raise ValueError('invalid trajectory utility proportions registration')
    return study, base['preset'], configs, arms


def with_share(config, share, seed, targets):
    result = dict(config, targets=targets, second_test_cap=100,
                  trajectory_utility=dict(candidates=8, refresh_steps=2000, tu_share=share))
    if 42 <= seed <= 46:
        result['per_start_diagnostics'] = True
    else:
        result.pop('per_start_diagnostics', None)
    return result


def pilot_cells():
    _, preset, configs, arms = registered()
    cells = read(HERE / 'pilot-cells.json')
    if set(cells['tasks']) != set(configs):
        raise ValueError('pilot does not cover the registered mazes')
    runs = []
    for task, cfg in configs.items():
        limits = cells['tasks'][task]
        if set(limits) != set(arms):
            raise ValueError('pilot does not cover the registered arms')
        for name, (agent, share) in arms.items():
            spec = limits[name]
            target = spec['last_target']
            minutes = spec['limit_minutes']
            if target not in (MINIMUM_STEPS, cfg['targets'][-1]) or type(minutes) is not int or minutes < 3:
                raise ValueError('invalid pilot horizon or limit')
            runs.append(dict(run_id=f"{cfg['id']}_{name}_s42", task=task, agent=agent,
                             tu_share=share, label=name, seed=42, last_target=target,
                             limit_seconds=minutes * 60))
    if sum(run['limit_seconds'] for run in runs) != cells['reservation_seconds']:
        raise ValueError('pilot reservation does not match the registered jobs')
    return cells, preset, configs, runs


def configuration(run, configs):
    cfg = configs[run['task']]
    targets = [target for target in cfg['targets'] if target <= run['last_target']]
    return with_share(cfg, run['tu_share'], run['seed'], targets)
