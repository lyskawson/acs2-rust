import argparse
import hashlib
import json
from pathlib import Path

from protocol import ROOT
from execution import authorization_manifest
from grid_dispatch import dispatch
from grid_protocol import atomic_json


def execute(root, repository, purpose):
    root, repository = Path(root), Path(repository)
    if purpose == 'pilot':
        manifest = root / 'pilot-manifest.json'
        expected = authorization_manifest(json.loads(manifest.read_text()))
        actual = json.loads(manifest.read_text())
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError('pilot manifest changed')
        return dispatch(root / 'pilot', manifest, repository, None)
    registered = root / 'approved-plan.json'
    digest = hashlib.sha256(registered.read_bytes()).hexdigest()
    proposed = dict(stage='approved_grid', purpose='replication', authorization_plan=str(registered), authorization_sha256=digest)
    manifest = authorization_manifest(proposed, registered)
    operations = root / 'execution-operations/manifest.json'
    manifest.update(authorization_plan=str(registered), authorization_sha256=digest,
        job_name='tu-pair-diagnostics', operations_commit=json.loads(operations.read_text())['commit'],
        operations_manifest=str(operations), operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())
    path = root / 'replication-manifest.json'
    atomic_json(path, manifest)
    return dispatch(root / 'replication', path, repository, root / 'approval.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--purpose', choices=['pilot', 'replication'], required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.root, args.repo, args.purpose)))
