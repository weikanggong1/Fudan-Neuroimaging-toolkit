"""Project immutable actual report metadata for concise ten-case delivery; preserve originals by SHA/path."""
import json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
R=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_01');A=R/'CPU_final10_independent_audit_v1';O=R/'CPU_final10_concise_metadata_v1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
while not (A/'audit.json').exists():time.sleep(30)
a=load(A/'audit.json');assert a['state']=='actual_ten_CPU_independently_verified_and_compared'
summary=load(R/'CPU_reference_explicit_ten_delivery_v1/summary.json');index={}
for s,row in summary['cases'].items():
 d=Path(row['official_case_directory']);p=d/'report.json';r=load(p);assert sha(p)==row['report_SHA256'];commands=[]
 for c in r['commands']:
  item={k:v for k,v in c.items() if k!='memory_samples'};item['memory_samples_retained_in_original_report']=True;item['original_memory_sample_count']=len(c.get('memory_samples',[]));commands.append(item)
 v={k:v for k,v in r.items() if k not in ('commands','source_CPU_stage_lineage')};v.update(commands=commands,projection_only=True,original_report_path=str(p),original_report_SHA256=sha(p),original_report_bytes=p.stat().st_size)
 if 'source_CPU_stage_lineage' in r:v['source_CPU_stage_lineage']={k:x for k,x in r['source_CPU_stage_lineage'].items() if k!='source_sha256'};v['source_CPU_stage_lineage']['source_sha256']=r['source_CPU_stage_lineage'].get('source_sha256',{})
 dest=O/s;dest.mkdir();save(dest/'report_execution_metadata.json',v)
 (dest/'completed_contract_verified.json').write_bytes((d/'completed_contract_verified.json').read_bytes())
 evidence=Path(row['evidence_directory']);(dest/'comparison.json').write_bytes((evidence/'comparison.json').read_bytes())
 index[s]={'original_report_path':str(p),'original_report_SHA256':sha(p),'projected_metadata_SHA256':sha(dest/'report_execution_metadata.json'),'contract_SHA256':sha(dest/'completed_contract_verified.json'),'comparison_SHA256':sha(dest/'comparison.json'),'execution_kind':row['execution_kind'],'raw_case_id':'sub-'+s}
assert len(index)==10
save(O/'index.json',{'observed_UTC':datetime.now(timezone.utc).isoformat(),'case_count':10,'cases':index,'independent_audit':str(A/'audit.json'),'independent_audit_SHA256':sha(A/'audit.json'),'original_reports_and_samples_not_modified':True,'tool_SHA256':sha(__file__)})
save(O/'SHA256_manifest.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()})
