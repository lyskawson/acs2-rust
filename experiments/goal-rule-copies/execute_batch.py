import argparse
import hashlib
import json
from pathlib import Path

from protocol import ROOT
from execution import authorization_manifest, APPROVED_SHA256
from grid_dispatch import dispatch
from grid_protocol import atomic_json


def execute(root,repository,purpose):
    root=Path(root)
    registered=root/'approved-plan.json'
    proposed=dict(stage='approved_grid',purpose=purpose,authorization_plan=str(registered),authorization_sha256=APPROVED_SHA256)
    manifest=authorization_manifest(proposed,registered)
    operations=root/'execution-operations/manifest.json'
    manifest.update(authorization_plan=str(registered),authorization_sha256=APPROVED_SHA256,
        job_name='tu-goal-rule-copies',operations_commit=json.loads(operations.read_text())['commit'],
        operations_manifest=str(operations),operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest())
    path=root/'continuation'/(purpose+'-manifest.json')
    atomic_json(path,manifest)
    return dispatch(root/purpose,path,repository,root/'approval.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--purpose',choices=['diagnostics','compact'],required=True)
    args=parser.parse_args()
    print(json.dumps(execute(args.root,args.repo,args.purpose)))
