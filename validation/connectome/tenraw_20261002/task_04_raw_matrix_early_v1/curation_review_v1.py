"""Stdlib verification of frozen payloads and saved one-sided matrix decisions.

Read-only: no image reconstruction, matrix recomputation, MRI/GPU solver or source mutation.
"""
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

start=time.perf_counter()
root=Path(sys.argv[1]).resolve();source=Path(sys.argv[2]).resolve()
relative=Path('validation/connectome/tenraw_20261002/task_04_raw_matrix_early_v1')
p=root/relative;index=json.loads((p/'evidence_index.json').read_bytes())
commit=index['source_commit']
original_index=subprocess.check_output(['git','show',f'{commit}:{relative}/evidence_index.json'],cwd=source)
assert hashlib.sha256(original_index).hexdigest()==index['original_index_sha256']
source_checks=[];adopted=[]
for row in index['entries']:
    stored=(source/relative/row['artifact']).read_bytes()
    assert len(stored)==row['stored_size_bytes']
    assert hashlib.sha256(stored).hexdigest()==row['stored_sha256']
    raw=gzip.decompress(stored) if row['compression']=='gzip_lossless' else stored
    assert len(raw)==row['raw_size_bytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256']
    source_checks.append(row['artifact'])
    if row['publication']=='adopted':
        assert (p/row['artifact']).read_bytes()==stored
        adopted.append(row['artifact'])
assert len(source_checks)==index['original_index_entries']==83
config=json.loads((p/'actual/config.json').read_bytes())
source_equivalence=[]
for path,expected in config['source_files'].items():
    payload=(root/path).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==expected
    assert payload==(source/relative/'actual'/path).read_bytes()
    source_equivalence.append({'root_tool':path,'sha256':expected,'duplicate_snapshot_omitted':True})
summary_bytes=(p/'summary.json').read_bytes()
assert summary_bytes==subprocess.check_output(['git','show',f'{commit}:{relative}/summary.json'],cwd=source)
adopted.append('summary.json')
summary=json.loads(summary_bytes);reports=[];total=0;accepted_total=0
for entry in summary['verified_reports']:
    raw=gzip.decompress((p/(entry['file']+'.gz')).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    d=json.loads(raw);assert d['fnit_seeds']==[0] and d['official_seeds']==[0,1,2,3,4]
    assert d['matrix_envelope_status']=='failed'
    assert d['fnit_reproducibility_status']==d['population_envelope_status']=='not_assessed'
    assert len(d['profiles'])==8
    checked=0;accepted=0;by_profile={}
    for name,profile in d['profiles'].items():
        pairs=profile['pairwise'];assert len(pairs['official'])==10 and len(pairs['cross'])==5 and pairs['fnit']==[]
        profile_accepted=0;assert len(profile['ranges'])==6
        for metric,gate in profile['ranges'].items():
            official=[pair['metrics'][metric] for pair in pairs['official']]
            cross=[pair['metrics'][metric] for pair in pairs['cross']]
            assert official==gate['official_values'] and cross==gate['fnit_vs_official']
            assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in official+cross)
            if gate['criterion']=='>= official_min':
                limit=min(official);decisions=[v>=limit for v in cross]
            else:
                assert gate['criterion']=='<= official_max'
                limit=max(official);decisions=[v<=limit for v in cross]
            assert gate['threshold']==limit and gate['comparison_accepted']==decisions
            assert gate['official_min_max']==[min(official),max(official)]
            assert gate['accepted_count']==sum(decisions)
            assert gate['status']==('passed' if all(decisions) else 'failed')
            accepted+=sum(decisions);profile_accepted+=sum(decisions);checked+=len(decisions)
        assert profile_accepted==entry['profiles'][name]['accepted']
        by_profile[name]=profile_accepted
    assert checked==entry['total']==240 and accepted==entry['accepted']
    reports.append({'arm':entry['arm'],'case_id':entry['case_id'],'raw_sha256':entry['sha256'],
                    'accepted':accepted,'total':checked,'by_profile':by_profile,
                    'matrix_envelope_status':'failed','FNIT_self':'not_assessed','population':'not_assessed'})
    total+=checked;accepted_total+=accepted
remote=json.loads((p/'curation_actual_remote_read_v1.json').read_bytes())
assert remote['all_current_bound_bytes_match'] and remote['all_four_official_contracts_completed'] and len(remote['bindings'])==96
assert all(r['matches'] and r['unchanged_during_read'] for r in remote['bindings'])
assert remote['config_sha256']==hashlib.sha256((p/'curation_readonly_config_v1.json').read_bytes()).hexdigest()
assert remote['audit_source_sha256']==hashlib.sha256((p/'curation_readonly_audit_v1.py').read_bytes()).hexdigest()
tests=json.loads((p/'curation_actual_CPU_contract_tests_v1.json').read_bytes())
assert tests['returncode']==0 and '13 passed' in tests['output']
for path,expected in tests['source_sha256'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Frozen early payload integrity and replay of saved metrics/decisions; no new scientific matrix or population calculation.',
        'source_commit':commit,'original_index_sha256':index['original_index_sha256'],
        'original_83_payload_integrity':'passed','adopted_original_payloads':adopted,
        'original_83_stored_bytes':sum(r['stored_size_bytes'] for r in index['entries']),
        'extra_original_summary_binding':{'path':'summary.json','size_bytes':len(summary_bytes),'sha256':hashlib.sha256(summary_bytes).hexdigest(),'bound_to_source_commit':commit},
        'adopted_original_stored_bytes':len(summary_bytes)+sum(r['stored_size_bytes'] for r in index['entries'] if r['publication']=='adopted'),
        'existing_root_tools_byte_equivalence':source_equivalence,'reports':reports,
        'gate_decisions_replayed':total,'accepted':accepted_total,'all_reports_failed':len(reports)==7,
        'raw_wholechain_FNIT_self':'not_assessed','population':'not_assessed',
        'remote_actual_binding_count':len(remote['bindings']),'remote_actual_SHA':'all_match',
        'head_CPU_contract_tests':'13_passed','head_CPU_source_SHA':'all_match',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_seconds':time.perf_counter()-start,'MRI_solver_commands':0,'GPU_commands':0}
print(json.dumps(result,indent=2,allow_nan=False))
