import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from protocol import ROOT, REFERENCES, ADMISSIBILITY_SHA, BLOCKS, configurations
from grid_attempts import attempts, inspect_attempt
from grid_identity import compare_per_pair
from grid_protocol import atomic_json, load_plan
from collect import outside_repository

HER_SHA = 'e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444'
PAIR_SHA = 'ec83bffbba7ca0d14315138abf03d6b485900d6f3c63dcfa0172ebe74a53d8d5'


def freeze(output, her, admissibility, per_pair):
    output, her, admissibility, per_pair = outside_repository(output), Path(her), Path(admissibility), Path(per_pair)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('archived_reference_catalog', ROOT / 'experiments/hindsight-admissibility/references.py')
    catalog_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog_module)
    configs = configurations()
    sources = catalog_module.catalog(her / 'main-grid', her / 'approved-plan.json', HER_SHA,
                                     ['acs2er_episodic'], configs, list(range(42, 62)))
    for config in configs:
        sources.update(catalog_module.catalog(admissibility / (config['id'] + '_acs2her_shortened'),
            admissibility / 'approved-plan.json', ADMISSIBILITY_SHA, ['acs2her_shortened'], [config], list(range(42, 62))))
    success_sources = {key: entry['path'] / 'rows.jsonl' for key, entry in sources.items()}
    index = []
    def copy(kind, key, source):
        config, arm, seed = key
        relative = Path(kind) / config / arm / f's{seed}.jsonl'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        index.append(dict(kind=kind, configuration_id=config, agent=arm, seed=seed, file=str(relative),
            source=str(source), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), charged_to_this_study=False))
    for key, source in sorted(success_sources.items()):
        copy('success', key, source)
    load_plan(per_pair / 'approved-plan.json', PAIR_SHA)
    observed = set()
    for batch_name in ('pilot', 'replication'):
        batch = per_pair / batch_name
        plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        ledger = json.loads((batch / 'ledger.json').read_text())
        for entry in attempts(batch, ledger):
            key = entry['configuration_id'], entry['agent'], entry['seed']
            if entry['agent'] not in REFERENCES:
                continue
            config = next(config for config in plan['configurations'] if config['id'] == entry['configuration_id'])
            inspected, rows = inspect_attempt(entry, config)
            if inspected['status'] != 'complete' or key in observed:
                raise ValueError('per-pair reference needs one complete original attempt')
            compact = [json.loads(line) for line in success_sources[key].read_text().splitlines()]
            compare_per_pair(rows, compact, [key[2]], config['targets'], BLOCKS)
            copy('pairs', key, entry['path'] / 'rows.jsonl')
            observed.add(key)
    expected = {(config['id'], arm, seed) for config in configs for arm in REFERENCES for seed in range(42,47)}
    if observed != expected or len(index) != 300:
        raise ValueError('expected 240 success references and 60 pair references')
    atomic_json(output / 'reference-index.json', index)
    return dict(files=len(index), reused_runs=240, reused_pair_runs=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('out', 'her', 'admissibility', 'per-pair'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out, args.her, args.admissibility, args.per_pair)))
