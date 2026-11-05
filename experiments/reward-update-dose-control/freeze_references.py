import argparse
import hashlib
import json
from pathlib import Path
import shutil

from protocol import G, REFERENCES, configurations
from freeze_dose import derive
from collect import outside_repository
from grid_protocol import atomic_json


def freeze(output, archive):
    output, archive = outside_repository(output), Path(archive)
    probabilities = json.loads(Path(__file__).with_name('probabilities.json').read_text())
    if derive(archive) != probabilities:
        raise ValueError('STOP: original G dose differs from the fixed integer registration')
    verification = json.loads((archive / 'verification.json').read_text())
    if verification['plan_sha256'] != probabilities['source_plan_sha256']:
        raise ValueError('G source registration changed')
    output.mkdir(parents=True, exist_ok=False)
    index = []
    def copy(kind, config, arm, seed, source, expected):
        source = Path(source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('original reference checksum differs from G study provenance')
        relative = Path(kind) / config / arm / f's{seed}.jsonl'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        index.append(dict(kind=kind, configuration_id=config, agent=arm, seed=seed,
            file=str(relative), source=str(source), sha256=expected, charged_to_this_study=False))
    for entry in verification['runs_index']:
        for kind in ['success'] + (['pairs'] if entry['seed'] < 47 else []):
            copy(kind, entry['configuration_id'], G, entry['seed'], archive / entry['file'], entry['sha256'])
    for entry in json.loads((archive / 'reference-index.json').read_text()):
        copy(entry['kind'], entry['configuration_id'], entry['agent'], entry['seed'], entry['source'], entry['sha256'])
    keys = [(entry['kind'], entry['configuration_id'], entry['agent'], entry['seed']) for entry in index]
    expected = {(kind, config['id'], arm, seed) for kind, seeds in [('success', range(42,62)), ('pairs', range(42,47))]
                for config in configurations() for arm in REFERENCES for seed in seeds}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing, duplicate or extra original reference files')
    atomic_json(output / 'reference-index.json', index)
    return dict(files=len(index), success_runs=360, pair_runs=90, original_sources_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out, args.archive)))
