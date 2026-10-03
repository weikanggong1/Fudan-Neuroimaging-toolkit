#!/usr/bin/env python3
"""等待十份真实完成报告，再复用冻结 v2 stage collector；只读，不运行 MRI。"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
CASES=('CON01','CON03','CON04','CON05','CON06','CON07','CON08','CON09','CON10','CON11')
COLLECTOR_SHA='703a27edbd6c29fc897228b6056f36233873dac004e37d05ded03090826a4026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--collector',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--poll-seconds',type=float,default=60);a=p.parse_args();r=a.reference_root.resolve();c=a.collector.resolve();o=a.output_root.resolve()
 for protected in [r,c.parent,Path(__file__).resolve().parent]:
  if o==protected or o.is_relative_to(protected) or protected.is_relative_to(o):raise ValueError('output overlaps protected source/producer/input')
 if o.exists():raise FileExistsError('new output only')
 if sha(c)!=COLLECTOR_SHA:raise ValueError('collector not frozen actual v2')
 o.mkdir(parents=True);own_before=sha(Path(__file__));paths={case:r/'cases'/f'sub-{case}'/('attempt-02' if case=='CON01' else 'attempt-01') for case in CASES}
 while True:
  ready=[]
  for case,pth in paths.items():
   f=pth/'report.corrected.public.json'
   if not f.exists():continue
   d=json.loads(f.read_text())
   if d['status']!='complete' or d['command_exit_code']!=0:continue
   if d['frames']!=180 or d['source_unchanged_during_run'] is not True or d['input_unchanged_during_run'] is not True or d['corrective_saved_file_guards_equal'] is not True:raise ValueError('complete reference contract failed')
   ready.append(case)
  (o/'waiting.public.json').write_text(json.dumps({'status':'waiting' if len(ready)!=10 else 'collecting','observed_utc':datetime.now(timezone.utc).isoformat(),'completed_cases':ready,'pending_cases':[x for x in CASES if x not in ready],'waiter_sha256':own_before,'collector_sha256':COLLECTOR_SHA},indent=2)+'\n')
  if len(ready)==10:break
  time.sleep(a.poll_seconds)
 before={case:{'report_sha256':sha(pth/'report.corrected.public.json'),'node_runtime_sha256':sha(pth/'node_runtime.public.json')} for case,pth in paths.items()};target=o/'reference_stages_all10.public.json';start=time.perf_counter()
 subprocess.run([sys.executable,str(c),'--reference-root',str(r),'--output',str(target),'--expected-cases',*CASES],check=True)
 report=json.loads(target.read_text())
 if report['status']!='complete_cohort' or set(report['cases'])!=set(CASES):raise ValueError('full ten-case actual set failed')
 for case,row in report['cases'].items():
  for key,value in before[case].items():
   if row[key]!=value:raise ValueError('stage source report/node binding mismatch')
 after={case:{'report_sha256':sha(pth/'report.corrected.public.json'),'node_runtime_sha256':sha(pth/'node_runtime.public.json')} for case,pth in paths.items()};guard={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'waiter_sha256':own_before,'waiter_sha256_after':sha(Path(__file__)),'collector_sha256':COLLECTOR_SHA,'collector_sha256_after':sha(c),'stage_report_sha256':sha(target),'source_report_node_sha256':before,'source_report_node_sha256_after':after,'source_report_node_unchanged':before==after,'collection_seconds':time.perf_counter()-start,'timing_scope':'Independent posthoc metadata collection only; no wait, node interval, or collection duration added to original MRI production clocks.'}
 if before!=after or sha(c)!=COLLECTOR_SHA or sha(Path(__file__))!=own_before:raise ValueError('metadata/source changed during collection')
 (o/'final_collection_guard.public.json').write_text(json.dumps(guard,indent=2)+'\n');print(json.dumps({'status':'complete','stage_report_sha256':sha(target),'guard_report_sha256':sha(o/'final_collection_guard.public.json')}))
if __name__=='__main__':main()
