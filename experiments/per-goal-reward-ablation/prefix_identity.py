import hashlib
import json
from pathlib import Path
import struct

from cost_pilot_registration import TARGETS

NONLEARNING = {'host','cpu_model','wall_seconds_train','wall_seconds_eval','wall_seconds_total'}


def field_bits(value):
    if isinstance(value,float):return ['float64',struct.pack('>d',value).hex()]
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',value]
    if isinstance(value,str):return ['str',value]
    if value is None:return ['null']
    if isinstance(value,list):return ['list',[field_bits(v) for v in value]]
    if isinstance(value,dict):return ['object',{k:field_bits(v) for k,v in value.items()}]
    raise TypeError(type(value))


def learning_fields(row):
    return field_bits({k:v for k,v in row.items() if k not in NONLEARNING})


def compare(old,fresh):
    prefix=[row for row in old if row['nominal_step']<=30000]
    if [r['nominal_step'] for r in prefix]!=TARGETS or [r['nominal_step'] for r in fresh]!=TARGETS:
        raise ValueError('identity requires every complete registered 30k prefix checkpoint')
    for left,right in zip(prefix,fresh):
        if learning_fields(left)!=learning_fields(right):
            different=sorted(k for k in set(left)|set(right) if k not in NONLEARNING and field_bits(left.get(k))!=field_bits(right.get(k)))
            raise ValueError('fresh 30k run differs from stopped prefix at '+str(left['nominal_step'])+': '+str(different))
    return dict(checkpoints=TARGETS,learning_and_per_pair_fields_bit_identical=True,
                excluded_fields=sorted(NONLEARNING),performance_summary=False)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_cell(prior_root,run,fresh_file):
    prior_root=Path(prior_root);fresh_file=Path(fresh_file)
    matches=[prior_root/r['relative_directory']/'rows.jsonl'
        for batch in json.loads((prior_root/'ledger.json').read_text()) for r in batch['runs']
        if (r['configuration_id'],r['agent'])==(run['configuration_id'],run['agent'])]
    if len(matches)!=1:raise ValueError('old prefix identity source missing or duplicated')
    old=matches[0]
    manifest=json.loads((prior_root/'archive-checksums.json').read_text())
    old_sha=hashlib.sha256(old.read_bytes()).hexdigest()
    if manifest[str(old.relative_to(prior_root))]!=old_sha:raise ValueError('old prefix evidence changed')
    fresh_sha=hashlib.sha256(fresh_file.read_bytes()).hexdigest()
    completion=json.loads(fresh_file.with_name('completion.json').read_text())
    if completion['exit_code'] or completion['failure'] or completion['sha256']['rows.jsonl']!=fresh_sha:
        raise ValueError('fresh prefix identity run did not complete unchanged')
    return dict(configuration_id=run['configuration_id'],agent=run['agent'],seed=42,
        **compare(read(old),read(fresh_file)),prior_file=str(old),fresh_file=str(fresh_file),
        prior_sha256=old_sha,fresh_sha256=fresh_sha)


def verify(prior_root,fresh_root):
    prior_root,fresh_root=Path(prior_root),Path(fresh_root)
    prior=[run for batch in json.loads((prior_root/'ledger.json').read_text()) for run in batch['runs']]
    fresh={(run['configuration_id'],run['agent']):fresh_root/run['relative_directory']/'rows.jsonl'
        for batch in json.loads((fresh_root/'ledger.json').read_text()) for run in batch['runs']}
    checks=[]
    for run in prior:
        key=run['configuration_id'],run['agent']
        if key not in fresh:raise ValueError('fresh prefix identity cell missing')
        checks.append(verify_cell(prior_root,run,fresh[key]))
    if len(checks)!=18:raise ValueError('all 18 old prefixes require independent fresh checks')
    return dict(verified=True,cells=18,checkpoint_comparisons=72,checks=checks,success_summary=False)
