import argparse
import hashlib
import json
from pathlib import Path

from execution import APPROVED_SHA256, BINARY_SHA256, approved, batches, grid_manifest, study_scope
from grid_protocol import atomic_json, load_plan
from grid_scope import snapshot
from submit import execute, submit


def execute_batch(root, index, repository):
    root, repository = Path(root).resolve(), Path(repository).resolve()
    plan = approved(root / 'approved-plan.json')
    scope_path = root / 'scope.json'
    scope = json.loads(scope_path.read_text())
    if scope != study_scope(plan, root):
        raise ValueError('accounting must cover every registered batch and pilot allocation')
    if (root / 'STOP.json').exists():
        raise ValueError('shared STOP blocks further submissions')
    manifest = grid_manifest(plan, index, root, root)
    operations = root / 'operations/manifest.json'
    manifest.update(authorization_plan=str(root / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        budget_scope=str(scope_path), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        stop_path=str(root / 'STOP.json'), job_name='tu-hindsight-admissibility',
        operations_commit=json.loads(operations.read_text())['commit'], operations_manifest=str(operations),
        operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())
    state = snapshot(manifest, execute)
    if state['reserved_seconds']:
        raise ValueError('finish the outstanding twenty-job batch before submitting another')
    binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != BINARY_SHA256:
        raise ValueError('the approved pilot binary changed; rebuilding is not authorized')
    config, agent = batches(plan)[index]
    batch = root / (config['id'] + '_' + agent)
    if batch.exists():
        raise ValueError('batch exists; reconcile its attempts instead of duplicating it')
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
