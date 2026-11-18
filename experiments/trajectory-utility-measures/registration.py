from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ABLATION = HERE.parent / 'per-goal-reward-ablation'
sys.path.extend([str(HERE.parent / 'grid'), str(ABLATION)])

from protocol import OLD_TASKS, S1


MEAN = 'acs2tuher_mean_value'
TRIAL = 'acs2tuher_trial_update'
RANDOM = 'acs2tuher_random_goal'
ARMS = (MEAN, TRIAL, RANDOM)
STUDY = HERE.name


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def ablation():
    draft = read(HERE / 'plan.json')
    for name, digest in draft['ablation_registration_sha256'].items():
        if checksum(ABLATION / name) != digest:
            raise ValueError(f'ablation input differs from the registration: {name}')
    plan = read(ABLATION / 'plan.json')
    if (draft['arms'] != list(ARMS) or draft['reference']['agent'] != S1
            or len(plan['configurations']) != 28 or draft['seeds'] != plan['seeds']
            or draft['per_pair_seeds'] != plan['per_pair_seeds']):
        raise ValueError('study and ablation diverge')
    return draft, plan


def pilot_registration():
    study, inherited = ablation()
    registered = read(HERE / 'pilot-cells.json')
    pilot = study['pilot']
    if (study['stage'] != 'draft_before_cost_pilot' or pilot['seed'] != registered['seed']
            or (registered['candidates'], registered['refresh_steps']) != (pilot['K'], pilot['F'])
            or (pilot['K'], pilot['F'], pilot['seed'], pilot['minimum_steps']) != (8, 2000, 42, 300000)
            or (registered['max_running_jobs'], registered['max_cpu_seconds']) != (40, 72000)
            or registered['max_cpu_seconds'] != pilot['cpu_hour_limit_including_reservations'] * 3600):
        raise ValueError('pilot authorization differs')
    settings = registered['tasks']
    configs = {}
    runs = []
    for config in inherited['configurations']:
        task = config['task']
        cell = settings[task]
        last = cell['last_target']
        limits = cell['limit_minutes']
        if last not in (pilot['minimum_steps'], config['targets'][-1]) or set(limits) != set(ARMS):
            raise ValueError('pilot horizon or arms differ')
        configs[config['id']] = dict(
            config, targets=[point for point in config['targets'] if point <= last],
            per_start_diagnostics=True,
            trajectory_utility=dict(candidates=pilot['K'], refresh_steps=pilot['F']),
        )
        for arm in ARMS:
            name = f"{config['id']}_{arm}_s42"
            if type(limits[arm]) is not int or limits[arm] < 3:
                raise ValueError('invalid allocation limit')
            runs.append(dict(configuration_id=config['id'], agent=arm, seed=42, attempt_id=name + '_a01',
                             relative_directory=f'attempts/{name}_a01/measurement/{name}',
                             limit_seconds=limits[arm] * 60))
    seconds = sum(run['limit_seconds'] for run in runs)
    if (set(settings) != {config['task'] for config in configs.values()}
            or len(runs) != pilot['cells'] or len({run['attempt_id'] for run in runs}) != 84
            or not seconds == registered['reserved_seconds'] <= registered['max_cpu_seconds']):
        raise ValueError('invalid pilot grid or allocation budget')
    return registered, inherited['preset'], configs, runs
