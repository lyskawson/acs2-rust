import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from protocol import ROOT, ARMS, ADMISSIBILITY_SHA, BASELINE_SHA, configurations
from grid_protocol import atomic_json

HER_SHA = 'e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444'


def catalog_sources(baseline, her, admissibility):
    spec = importlib.util.spec_from_file_location('admissibility_reference_catalog', ROOT / 'experiments/hindsight-admissibility/references.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline, her, admissibility = Path(baseline), Path(her), Path(admissibility)
    configs, seeds = configurations(), list(range(42, 47))
    result = module.catalog(baseline, baseline / 'plan.json', BASELINE_SHA, ['acs2'], configs, seeds)
    result.update(module.catalog(her / 'main-grid', her / 'approved-plan.json', HER_SHA,
                                 ['acs2er_episodic', 'acs2her'], configs, seeds))
    for config in configs:
        for arm in ('acs2her_nontrivial', 'acs2her_shortened'):
            result.update(module.catalog(admissibility / (config['id'] + '_' + arm),
                admissibility / 'approved-plan.json', ADMISSIBILITY_SHA, [arm], [config], seeds))
    if len(result) != 150:
        raise ValueError('frozen references require five arms, six mazes and seeds 42--46')
    return result


def freeze(root, sources):
    root = Path(root)
    index = []
    for (configuration_id, arm, seed), entry in sorted(sources.items()):
        relative = Path('references') / configuration_id / arm / f's{seed}.jsonl'
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = entry['path'] / 'rows.jsonl'
        shutil.copyfile(source, path)
        index.append(dict(configuration_id=configuration_id, agent=arm, seed=seed, file=str(relative),
            source=str(source), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            historical_cpu_seconds=entry['allocation_cpu_seconds'], charged_to_this_study=False))
    atomic_json(root / 'reference-index.json', index)
    return index


def read(root):
    root = Path(root)
    result = {}
    for entry in json.loads((root / 'reference-index.json').read_text()):
        path = root / entry['file']
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('frozen reference changed or escaped its root')
        key = entry['configuration_id'], entry['agent'], entry['seed']
        if key in result:
            raise ValueError('duplicate reference')
        result[key] = entry
    expected = {(config['id'], arm, seed) for config in configurations() for arm in ARMS for seed in range(42, 47)}
    if set(result) != expected:
        raise ValueError('frozen reference grid differs')
    return result
