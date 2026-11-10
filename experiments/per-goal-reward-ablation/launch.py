import json
import os
from pathlib import Path
import subprocess
import sys
from execution import checksum,require_authorization,registered_runs
from protocol import configurations,recorded
from grid_protocol import atomic_json,load_plan,seconds,stop_phase
from grid_launch import allocation_fields
from schema import validate_sequence


def launch(root,assignment,index):
    root,assignment=Path(root),Path(assignment);stop=root/'STOP.json'
    if stop.exists():raise ValueError('study stopped')
    try:
        assigned=json.loads(assignment.read_text());plan=load_plan(root/'plan.json',assigned['plan_sha256']);require_authorization(plan)
        if root.resolve()!=Path(plan['result_root']).resolve():raise ValueError('node result root differs')
        expected={r['attempt_id']:r for r in registered_runs(plan)}
        if any(expected.get(r['attempt_id'])!=r for r in assigned['runs']):raise ValueError('assignment exceeds seed42 pilot')
        run=assigned['runs'][index]
        repo=Path(assigned['measurement_repository'])
        commit=subprocess.check_output(['/usr/bin/git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['/usr/bin/git','-C',str(repo),'status','--porcelain'],text=True).strip()
        if commit!=plan['measurement_commit'] or dirty or checksum(assigned['binary'])!=plan['binary_sha256']:raise ValueError('node source or binary identity differs')
        array,task=os.environ['SLURM_ARRAY_JOB_ID'],os.environ['SLURM_ARRAY_TASK_ID']
        if int(task)!=index:raise ValueError('array index differs')
        details=subprocess.check_output(['scontrol','show','job','-o',f'{array}_{task}'],text=True)
        fields=allocation_fields(details,os.environ['SLURM_JOB_ID'],array,task)
        limit=seconds(fields['TimeLimit']);timeout=limit-seconds(fields['RunTime'])-30
        if limit!=run['limit_seconds'] or timeout<60:raise ValueError('allocation limit differs or insufficient remaining time')
        output=(root/run['relative_directory']).parent;atomic_json(output.parent/'allocation-start.json',dict(fields=fields,response=details))
        config=recorded(next(c for c in configurations() if c['id']==run['configuration_id']),42)
        manifest=dict(configuration=config,agent=run['agent'],seed=42,attempt_id=run['attempt_id'],
            expected_cpu_model=plan['cpu_model'],expected_host=None,expected_binary_sha256=plan['binary_sha256'],
            expected_preset=plan['preset'],measurement_commit=commit,operations_commit=commit,
            stop_path=str(stop),first_row_timeout_seconds=60,process_timeout_seconds=timeout,allocation_limit_seconds=limit)
        path=output.parent/'manifest.json';atomic_json(path,manifest)
        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('collector.py')),'--manifest',str(path),
            '--binary',assigned['binary'],'--commit',commit,'--output',str(output),'--index','0'],check=True)
        rows=[json.loads(line) for line in (root/run['relative_directory']/'rows.jsonl').read_text().splitlines()]
        validate_sequence(rows,config,run['agent'],42,commit)
    except BaseException as error:
        stop_phase(stop,dict(error=str(error),job=os.environ.get('SLURM_JOB_ID'),index=index));raise


if __name__=='__main__':launch(sys.argv[1],sys.argv[2],int(sys.argv[3]))
