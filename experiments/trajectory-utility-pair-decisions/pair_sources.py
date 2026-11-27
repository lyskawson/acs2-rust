from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENTS = REPO / 'experiments'
PER_PAIR = EXPERIMENTS / 'per-pair-diagnostics'
GRID = EXPERIMENTS / 'grid'
PROPORTIONS = EXPERIMENTS / 'trajectory-utility-proportions'
RUNS = Path.home() / 'Desktop/tu-runs'
OUTPUT = RUNS / 'trajectory-utility-pair-decisions'
TRUTH_BINARY = REPO / 'target/release/acs2-maze-truth'
STUDY = 'trajectory-utility-pair-decisions'

E1 = 'acs2er_episodic_per_goal_reward'
S1 = 'acs2her_shortened_per_goal_reward'
MEAN = 'acs2tuher_mean_value'
RANDOM = 'acs2tuher_random_goal'
LAMBDAS = (('1/4', 'p1of4'), ('1/2', 'p1of2'), ('3/4', 'p3of4'), ('1', 'p1of1'))
TU_ARMS = tuple(f'{agent}_{suffix}' for agent in (MEAN, RANDOM) for _, suffix in LAMBDAS)
ARMS = (E1, S1) + TU_ARMS
SEEDS = tuple(range(42, 62))
PAIR_SEEDS = tuple(range(42, 47))
GROUPS = ('no_aliasing', 'aliasing')
PYALCS_MAZES = ('maze4', 'maze5', 'maze6', 'maze7', 'woods14', 'mazema')
GRID_MANIFEST = '0da1da2946fe5e955ac24e8a84070c36352fce9448541453bebb322dc8fbcccb'
RECOVERY_MANIFEST = '25cf8c2453500f6f96995e26a3beb118593d998e2f07b6dd46ddca6ef1fd7da0'
FINAL_PLAN = 'db5c71750b28996460996d832c98cb2b8c26ff9ac6b9ca36862c2a44a3a1bdcc'
PER_PAIR_MODULES = ('protocol', 'analysis', 'post_hoc', 'decision_metrics')


def per_pair_definitions():
    saved = {name: sys.modules.pop(name) for name in PER_PAIR_MODULES if name in sys.modules}
    sys.path[:0] = [str(PER_PAIR), str(GRID)]
    try:
        analysis = importlib.import_module('analysis')
        decisions = importlib.import_module('decision_metrics')
        if Path(analysis.__file__).resolve().parent != PER_PAIR or Path(decisions.__file__).resolve().parent != PER_PAIR:
            raise ValueError('per-pair definitions loaded from another directory')
    finally:
        del sys.path[:2]
        for name in PER_PAIR_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return analysis, decisions


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'input checksum differs: {path}')
    return path


def json_lines(path):
    with Path(path).open('rb') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def proportions_modules():
    if str(PROPORTIONS) not in sys.path:
        sys.path.insert(0, str(PROPORTIONS))
    import grid_results
    import proportions
    return grid_results, proportions


def registered_configs():
    _, proportions = proportions_modules()
    study, _, configs, _ = proportions.registered()
    if tuple(study['seeds']) != SEEDS or tuple(study['per_pair_seeds']) != PAIR_SEEDS:
        raise ValueError('registered seeds differ')
    if sorted({config['group'] for config in configs.values()}) != sorted(GROUPS):
        raise ValueError('registered aliasing groups differ')
    return configs


def archive_of(path):
    path = Path(path).resolve()
    if RUNS not in path.parents:
        raise ValueError(f'source outside the archives: {path}')
    parts = path.relative_to(RUNS).parts
    if 'weryfikacja' in parts or STUDY in parts:
        raise ValueError(f'source is a copy, not the original archive: {path}')
    return parts[0], str(Path(*parts[1:]))


def expected_archive(task, arm):
    if arm in (E1, S1):
        return 'per-goal-reward' if task in PYALCS_MAZES else 'per-goal-reward-ablation'
    if arm.endswith('_p1of2'):
        return 'trajectory-utility-measures'
    return 'trajectory-utility-proportions'


def index_entries(path, digest, agents, rename):
    result = {}
    for entry in json.loads(verified(path, digest).read_bytes()):
        if entry['agent'] not in agents:
            continue
        key = entry['task'], rename(entry['agent']), entry['seed']
        if key in result:
            raise ValueError('duplicate archive index entry')
        result[key] = dict(path=Path(entry['source']), sha256=entry['sha256'], kind='index:' + path.parent.parent.name)
    return result


def select_sources(configs):
    grid_results, proportions = proportions_modules()
    final_path = PROPORTIONS / 'final-plan.json'
    if sha256(final_path) != FINAL_PLAN:
        raise ValueError('fixed-share final plan differs')
    final = proportions.read(final_path)
    archive = RUNS / 'trajectory-utility-proportions'
    measured, _, _ = grid_results.measured_sources(archive / 'pilot', archive / 'grid', GRID_MANIFEST, final,
                                                   archive / 'grid-recovery', RECOVERY_MANIFEST)
    sources = {key: dict(path=value['path'], sha256=value['sha256'], kind='fixed-share:' + value['source'])
               for key, value in measured.items()}
    sources |= index_entries(RUNS / 'per-goal-reward-ablation/final-evidence/local-full-index.json',
                             grid_results.ARCHIVE_INDEX, (E1, S1), lambda agent: agent)
    sources |= index_entries(RUNS / 'trajectory-utility-measures/registered-analysis/full-index.json',
                             final['costs']['archive_sha256']['full-index'], (MEAN, RANDOM),
                             lambda agent: proportions.label(agent, 0.5))
    expected = {(task, arm, seed) for task in configs for arm in ARMS for seed in SEEDS}
    if sources.keys() != expected:
        raise ValueError('selected sources differ from the registered arms, mazes and seeds')
    result = []
    for (task, arm, seed), source in sorted(sources.items()):
        name, relative = archive_of(source['path'])
        if name != expected_archive(task, arm):
            raise ValueError(f'source taken from the wrong archive: {task} {arm} {seed} {name}')
        result.append(dict(task=task, arm=arm, seed=seed, archive=name, path=relative,
                           sha256=source['sha256'], kind=source['kind']))
    return result


def source_path(source):
    return RUNS / source['archive'] / source['path']


def truth_model(task, cap, binary=TRUTH_BINARY):
    content = subprocess.check_output([str(binary), task, str(cap)])
    return content, hashlib.sha256(content).hexdigest()


def pair_universe(test, config):
    starts = test.get('starts')
    count = config['registration']['blocks']['test']['evaluation_pairs']
    if not starts or len(starts) != count or test['evaluation_pairs'] != count:
        raise ValueError('STOP: per-pair test records are missing or incomplete')
    weight = Fraction(config['registration']['blocks']['test']['pair_weight'])
    if weight != Fraction(1, count) or any(not math.isclose(record['weight'], float(weight), rel_tol=1e-12, abs_tol=0)
                                           for record in starts):
        raise ValueError('test pair weights are not equal within the maze')
    return [(record['start'], record['goal'], record['shortest_distance']) for record in starts]


def check_identity(rows, task, arm, seed, config):
    if [row['nominal_step'] for row in rows] != config['targets']:
        raise ValueError(f'checkpoints differ from registration: {task} {arm} {seed}')
    agent = arm if arm in (E1, S1) else arm.rsplit('_p', 1)[0]
    if str(GRID) not in sys.path:
        sys.path.append(str(GRID))
    from agent_names import canonical_run_name
    for row in rows:
        if (row['task'], canonical_run_name(row['agent']), row['seed'], row['cap']) != (task, agent, seed, config['cap']):
            raise ValueError(f'row identity differs: {task} {arm} {seed}')


def check_source(item):
    source, config = item
    path = verified(source_path(source), source['sha256'])
    rows = json_lines(path)
    check_identity(rows, source['task'], source['arm'], source['seed'], config)
    universe = None
    if source['seed'] in PAIR_SEEDS:
        universes = {json.dumps(pair_universe(row['evaluation_blocks']['test'], config)) for row in rows}
        if len(universes) != 1:
            raise ValueError('pair universe changes between checkpoints')
        universe = hashlib.sha256(universes.pop().encode()).hexdigest()
    return source['task'], source['arm'], source['seed'], universe


def check_sources(sources, configs, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check_source, [(source, configs[source['task']]) for source in sources], chunksize=4))
    universes = {}
    for task, _, seed, universe in results:
        if universe is not None:
            universes.setdefault(task, set()).add(universe)
    if any(len(values) != 1 for values in universes.values()) or universes.keys() != configs.keys():
        raise ValueError('test pair universes differ between arms or seeds')
    return {task: values.pop() for task, values in universes.items()}
