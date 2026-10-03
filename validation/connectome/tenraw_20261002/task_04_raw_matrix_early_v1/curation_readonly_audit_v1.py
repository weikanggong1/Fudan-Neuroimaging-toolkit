"""Stdlib-only immutable artifact audit; stdout JSON, no MRI/GPU calculation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

config=json.loads(Path(sys.argv[1]).read_bytes())
start=time.perf_counter(); rows=[]; manifests={}
for declared in config['bindings']:
    path=Path(declared['path']);before=path.stat(); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    after=path.stat(); actual=digest.hexdigest()
    rows.append({**declared,'actual_sha256':actual,'size_bytes':after.st_size,
                 'unchanged_during_read':(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                 'matches':actual==declared['sha256']})
    if 'actual_official_five_seed_manifest' in declared['roles']:
        data=json.loads(path.read_bytes())
        manifests[str(path)]={'sha256':actual,'state':data.get('state'),
            'execution_completed':data.get('execution_completed'),
            'seeds':data.get('seeds'),'case_id':data.get('raw_case_binding',{}).get('case_id'),
            'scope':data.get('input_scope',data.get('scope')),
            'planned_command_count':len(data.get('commands',[])),
            'completed_command_count':len(data.get('completed_commands',[])),
            'planned_completed_commands_match':bool(data.get('commands')) and len(data['commands'])==len(data.get('completed_commands',[])) and all(item.get('returncode')==0 and all(item.get(key)==value for key,value in plan.items()) for plan,item in zip(data['commands'],data.get('completed_commands',[])))}
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':socket.gethostname(),'python':sys.executable,'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
        'scope':config['scope'],'source_commit':config['source_commit'],'config_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bindings':rows,'official_manifest_metadata':manifests,'reports':config['reports'],
        'all_current_bound_bytes_match':all(x['matches'] and x['unchanged_during_read'] for x in rows),
        'all_four_official_contracts_completed':len(manifests)==4 and all(m['state']=='completed' and m['execution_completed'] is True and m['seeds']==[0,1,2,3,4] and m['planned_completed_commands_match'] for m in manifests.values()),
        'wall_seconds':time.perf_counter()-start,'MRI_solver_commands':0,'GPU_commands':0}
print(json.dumps(result,indent=2,allow_nan=False))
raise SystemExit(0 if result['all_current_bound_bytes_match'] and result['all_four_official_contracts_completed'] else 1)
