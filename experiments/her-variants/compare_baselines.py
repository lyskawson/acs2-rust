import argparse
import hashlib
import json
from pathlib import Path

from protocol import BASELINE_SHA256
from verify import verify


OMIT = {'commit','source_state','host','cpu_model','wall_seconds_train','wall_seconds_eval','wall_seconds_total'}


def fields(row):
    return {key:value for key,value in row.items() if key not in OMIT}


def compare(current, archived):
    verify(current)
    archived=Path(archived)
    if (archived/'plan.sha256').read_text().strip() != BASELINE_SHA256:
        raise ValueError('wrong archived baseline plan')
    def read(root):
        ledger=json.loads((root/'ledger.json').read_text())
        result={}
        for entry in ledger:
            for run in entry['runs']:
                if run['seed'] !=42: continue
                path=root/run['relative_directory']/'rows.jsonl'
                completion=json.loads(path.with_name('completion.json').read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != completion['sha256']['rows.jsonl']:
                    raise ValueError('baseline artifact checksum mismatch')
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    key=row['task'],row['agent'],row['nominal_step']
                    if key in result: raise ValueError('duplicate seed-42 baseline row')
                    result[key]=fields(row)
        return result
    expected,actual=read(archived),read(Path(current))
    if set(expected)!=set(actual): raise ValueError('baseline regression point set differs')
    for key in expected:
        if expected[key]!=actual[key]:
            differing=sorted(name for name in set(expected[key])|set(actual[key]) if expected[key].get(name)!=actual[key].get(name))
            raise ValueError(f'baseline fields changed: {key}: {differing}')
    return dict(rows=len(actual),compared_top_level_fields=sum(map(len,actual.values())),differences=0,
                archived_plan_sha256=BASELINE_SHA256,excluded=sorted(OMIT),purpose='seed-42 regression only; archived twenty-seed results are reused')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('current',type=Path)
    parser.add_argument('archived',type=Path)
    args=parser.parse_args()
    result=compare(args.current,args.archived)
    (args.current/'baseline-identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
