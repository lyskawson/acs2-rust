import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from protocol import ROOT
from execution import approved, continuation_scope, validate_inputs, APPROVED_SHA256
from archive import checksum, checked_path
from collect import outside_repository
from grid_protocol import atomic_json


def prepare(root, destination):
    root,destination=outside_repository(root),Path(destination)
    plan,approval=approved()
    if not destination.is_absolute() or (root/'execution-operations').exists() or (root/'continuation').exists():
        raise ValueError('execution already prepared or destination is not absolute')
    git=['/usr/bin/git','-C',str(ROOT)]
    if subprocess.check_output(git+['status','--porcelain'],text=True).strip():
        raise ValueError('freeze clean operations before preparation')
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    validate_inputs(root,plan)
    inventory=json.loads((root/'raw-checksums.json').read_text())
    for name,digest in inventory.items():
        checked_path(root,dict(file=name,sha256=digest))
    if (root/'registered-plan.json').read_bytes()!=Path(__file__).with_name('plan.json').read_bytes():
        raise ValueError('registered study copy differs from approved plan')
    shutil.copyfile(root/'raw-checksums.json',root/'pilot-inventory.json')
    for name in ('plan.json','approval.json'):
        shutil.copyfile(Path(__file__).with_name(name),root/('approved-plan.json' if name=='plan.json' else name))
    (root/'continuation').mkdir()
    atomic_json(root/'continuation/scope.json',continuation_scope(destination))
    files={}
    for name in subprocess.check_output(git+['ls-files','experiments'],text=True).splitlines():
        target=root/'execution-operations'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        files[name]=checksum(target)
    atomic_json(root/'execution-operations/manifest.json',dict(commit=commit,files=files))
    result=dict(plan_sha256=APPROVED_SHA256,phase_cpu_hour_limit=15,pilot_cpu_seconds=865,
        operations_commit=commit,pilot_inventory_sha256=checksum(root/'pilot-inventory.json'),
        measurement_commit=plan['measurement_commit'],binary_sha256=plan['binary_sha256'],measurement_rebuilt=False)
    atomic_json(root/'execution-preparation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.destination)))
