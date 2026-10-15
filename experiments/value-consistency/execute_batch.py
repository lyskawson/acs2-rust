import argparse
import hashlib
import json
from pathlib import Path
import sys

from execution import APPROVED_SHA256, MEASUREMENT_COMMIT, approved, batches, grid_manifest
from protocol import ROOT

sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import atomic_json, load_plan
from grid_scope import snapshot
from submit import execute, submit


def execute_batch(root, index, repository):
    root, repository = Path(root).resolve(), Path(repository).resolve()
    plan = approved(root / 'approved-plan.json')
    scope = load_plan(root / 'scope.json', hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest())
    if scope['authorization_sha256'] != APPROVED_SHA256 or scope['pilot_allocation_cpu_seconds'] != 16041:
        raise ValueError('study accounting authorization differs')
    expected_roots = [str(root / (config['id'] + '_' + agent)) for config, agent in batches(plan)]
    if scope['batch_roots'] != expected_roots:
        raise ValueError('accounting must cover every registered batch')
    manifest = grid_manifest(plan, index, root, root)
    operations = root / 'operations/manifest.json'
    manifest.update(authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
                    budget_scope=str(root / 'scope.json'), budget_scope_sha256=hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest(),
                    stop_path=str(root / 'STOP.json'), job_name='tu-value-consistency',
                    operations_commit=json.loads(operations.read_text())['commit'], operations_manifest=str(operations),
                    operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())
    state = snapshot(manifest, execute)
    if state['reserved_seconds'] != 0:
        raise ValueError('finish the outstanding twenty-job batch before submitting another')
    config, agent = batches(plan)[index]
    batch = root / (config['id'] + '_' + agent)
    if batch.exists():
        raise ValueError('batch already exists; verify its attempts instead of duplicating them')
    directory = root / 'manifests'
    directory.mkdir(exist_ok=True)
    path = directory / (batch.name + '.json')
    atomic_json(path, manifest)
    submit(batch, path, repository, root / 'approval.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=int)
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args()
    execute_batch(args.root, args.batch, args.repo)
