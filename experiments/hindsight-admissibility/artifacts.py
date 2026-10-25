import json
import hashlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))
from grid_protocol import load_plan


def verify_snapshot(study, child):
    path = Path(study) / 'operations/manifest.json'
    manifest = load_plan(path, child['operations_manifest_sha256'])
    for name, digest in manifest['files'].items():
        file = path.parent / name
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('operations file is outside its snapshot')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f'operations snapshot changed: {name}')
    return manifest

def reference_resolver(root, plan):
    declared = Path(plan['authorization_plan']).parent
    def resolve(value):
        path = Path(value)
        if not path.is_relative_to(declared):
            raise ValueError('identity reference is outside this admissibility study')
        return Path(root) / path.relative_to(declared)
    return resolve

def verify_launch(path, plan, attempt):
    launch = json.loads((path / 'launch.json').read_text())
    expected = dict(commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'],
                    partition='lem-cpu-normal', array_job_id=str(attempt['array_id']), array_task_id=str(attempt['task_id']))
    if any(launch.get(name) != value for name, value in expected.items()):
        raise ValueError('recorded admissibility launch identity differs from the frozen measurement')
    manifest = launch['manifest']
    config = next(item for item in plan['configurations'] if item['id'] == attempt['configuration_id'])
    expected = dict(configuration=config, agent=attempt['agent'], seed=attempt['seed'], attempt_id=attempt['attempt_id'],
                    measurement_commit=plan['measurement_commit'], expected_binary_sha256=plan['binary_sha256'],
                    expected_preset=plan['preset'], allocation_limit_seconds=attempt['limit_seconds'])
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError('resolved admissibility launch differs from its registered attempt')
