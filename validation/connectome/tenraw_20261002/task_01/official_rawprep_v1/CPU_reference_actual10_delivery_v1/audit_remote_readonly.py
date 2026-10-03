"""Audit existing ten-case CPU evidence through an authenticated SSH session; no MRI solver."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE_PROGRAM = r"""import hashlib,json,socket,platform
from pathlib import Path
from datetime import datetime,timezone
E=json.loads(__EXPECTED_JSON__)
R=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_01')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
checks=[];rows={}
def check(label,actual,expected):
 if actual!=expected:raise AssertionError((label,actual,expected))
 checks.append(label)
for s,row in E['index']['cases'].items():
 p=Path(row['original_report_path']);r=load(p);d=p.parent;co=load(d/'completed_contract_verified.json');sr=E['summary']['cases'][s]
 check(s+':originalreportsha',sha(p),row['original_report_SHA256'])
 check(s+':contractsha',sha(d/'completed_contract_verified.json'),row['contract_SHA256'])
 check(s+':comparisonsha',sha(Path(sr['evidence_directory'])/'comparison.json'),row['comparison_SHA256'])
 check(s+':completed',r['completed'] and co['completed'],True)
 check(s+':CPU',r['EDDY_solver'],'cpu');check(s+':threads',r['threads'],8);check(s+':GPseed',r['gp_seed'],12345)
 check(s+':commandreturncodes',[c['returncode'] for c in r['commands']],[0]*len(r['commands']))
 check(s+':noGPU',any(c.get('GPU') for c in r['commands']),False)
 commands=[]
 for c in r['commands']:
  item={k:v for k,v in c.items() if k!='memory_samples'};item['memory_samples_retained_in_original_report']=True;item['original_memory_sample_count']=len(c.get('memory_samples',[]));commands.append(item)
 v={k:x for k,x in r.items() if k not in ('commands','source_CPU_stage_lineage')};v.update(commands=commands,projection_only=True,original_report_path=str(p),original_report_SHA256=sha(p),original_report_bytes=p.stat().st_size)
 if 'source_CPU_stage_lineage' in r:
  v['source_CPU_stage_lineage']={k:x for k,x in r['source_CPU_stage_lineage'].items() if k!='source_sha256'};v['source_CPU_stage_lineage']['source_sha256']=r['source_CPU_stage_lineage'].get('source_sha256',{})
 check(s+':projection_exact',v,E['projected'][s])
 actual_projection=hashlib.sha256((json.dumps(v,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
 check(s+':projectionsha',actual_projection,row['projected_metadata_SHA256'])
 for source,expected in r['source_sha256'].items():
  if str(source).startswith(str(R)) and str(source).endswith('.py'):check(s+':source:'+source,sha(source),expected)
 eddy=next(c for c in r['commands'] if c['stage']=='official_EDDY_CPU')
 check(s+':CPUcommandwall',eddy['wall_seconds'],sr['new_CPU8_EDDY_command_wall_seconds'])
 rows[s]={'original_report_path':str(p),'original_report_bytes':p.stat().st_size,'original_report_SHA256':sha(p),'projection_SHA256':actual_projection,'command_count':len(r['commands']),'original_memory_sample_counts':[len(c.get('memory_samples',[])) for c in r['commands']],'execution_kind':row['execution_kind'],'EDDY_command_wall_seconds':eddy['wall_seconds']}
D=R/'CPU_reference_explicit_ten_delivery_v1'
actual_index_checks={}
for name,expected in E['actual_index'].items():
 actual=sha(D/name);check('actualfinalindex:'+name,actual,expected);actual_index_checks[name]=actual
old=load(D/'SHA256_manifest.json');currentstatus=sha(D/'status.json')
check('staleoriginalstatusdifference',old['status.json']!=currentstatus,True)
check('actualstatuscomplete',load(D/'status.json')['state'],'actual_ten_verified_and_compared_CPU_delivery_collected')
diag_rows=[]
for item in E['diagnostic']['inputs']:
 p=Path(item['recorded_path']);expected=item['reported_SHA256']
 actual=sha(p);check('CON11input:'+p.name,actual,expected)
 diag_rows.append({'key':p.name,'path':str(p),'resolved_path':str(p.resolve()),'is_symlink':p.is_symlink(),'SHA256':actual})
print(json.dumps({'observed_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'python':platform.python_version(),'CPU_metadata_read_only':True,'MRI_or_GPU_solver_started':False,'check_count':len(checks),'checks':checks,'cases':rows,'actual_final_index':actual_index_checks,'original_collector_status_index_difference':{'recorded_SHA256':old['status.json'],'actual_SHA256':currentstatus},'CON11_EDDY_inputs':diag_rows},indent=2,allow_nan=False))
"""

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-root", type=Path, required=True,
                        help="Local curated CPU_reference_actual10_delivery_v1 directory")
    parser.add_argument("--diagnostic-root", type=Path, default=None,
                        help="Local CON11 raw-binding evidence; defaults to the delivery sibling")
    parser.add_argument("--host", default="gongwk@10.190.248.228",
                        help="Authenticated head host; all remote operations are metadata reads")
    parser.add_argument("--port", type=int, default=39516)
    parser.add_argument("--control-path", type=Path, required=True,
                        help="Existing SSH ControlMaster socket; no credential data are read")
    parser.add_argument("--output", type=Path, required=True,
                        help="New local JSON proof; refuses replacement of an existing file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    delivery = args.delivery_root.resolve(strict=True)
    diagnostic = args.diagnostic_root or delivery.parent / "CON11_Task2_actual_raw_binding_v1"
    index = load(delivery / "cases/index.json")
    source_index = load(delivery / "SOURCE_delivery_SHA256.json")
    used_files = ["cases/index.json", "summary.json",
                  "independent_audit/final_delivery_actual_SHA256_index.json"]
    used_files += [f"cases/{subject}/report_execution_metadata.json" for subject in index["cases"]]
    for name in used_files:
        if sha(delivery / name) != source_index[name]:
            raise ValueError(f"Local frozen evidence changed: {name}")
    if len(index["cases"]) != 10:
        raise ValueError("Exactly ten actual independent CPU references are required")
    expected = {
        "index": index,
        "summary": load(delivery / "summary.json"),
        "actual_index": load(delivery / "independent_audit/final_delivery_actual_SHA256_index.json"),
        "projected": {subject: load(delivery / "cases" / subject / "report_execution_metadata.json")
                      for subject in index["cases"]},
        "diagnostic": load(diagnostic / "actual_EDDY_input_bindings.json"),
    }
    remote_program = REMOTE_PROGRAM.replace("__EXPECTED_JSON__", repr(json.dumps(expected, allow_nan=False)))
    process = subprocess.run([
        "ssh", "-S", str(args.control_path), "-o", "ControlMaster=no",
        "-o", "BatchMode=yes", "-p", str(args.port), args.host, "/usr/bin/python3 -",
    ], input=remote_program, text=True, capture_output=True, check=True)
    proof = json.loads(process.stdout)
    proof["local_review_tool_SHA256"] = sha(Path(__file__))
    proof["local_expected_source_index_SHA256"] = sha(delivery / "SOURCE_delivery_SHA256.json")
    proof["local_expected_diagnostic_SHA256"] = sha(diagnostic / "actual_EDDY_input_bindings.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"check_count": proof["check_count"], "case_count": len(proof["cases"]),
                      "host": proof["host"], "output_SHA256": sha(args.output)}))


if __name__ == "__main__":
    main()
