"""收集只读报告，列出完成/等待状态；不启动计算，不把部分结果当成功。"""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
report={'overall_equivalence':'not_assessed','source_receipts':str(a.receipts),'modes':{},'cross_rows':[],'tail_rows':[]}
for mode in ('normalize','cross','tail','coordinates_fp32'):
 path=a.receipts/mode/'report.json'
 if not path.is_file():report['modes'][mode]={'state':'receipt_absent'};continue
 j=json.loads(path.read_text());report['modes'][mode]={'state':'complete' if j.get('complete') is True else 'running_or_waiting','source_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'commit':j.get('commit')}
 for case in j.get('cases',[]):
  if mode=='cross':
   for r in case.get('runs',[]):report['cross_rows'].append({'id':case['id'],'nu':r['combination'][0],'mask':r['combination'][1],'state':'complete' if 'norm_vs_official' in r else 'running_or_waiting','em_seconds':r.get('em_seconds_including_io'),'ca_seconds':r.get('ca_seconds_including_io'),'lta_max_abs':r.get('lta_max_abs_vs_official'),'norm_different':r.get('norm_vs_official',{}).get('different_voxels'),'ctrl_different':r.get('controls_vs_official',{}).get('different_voxels')})
  if mode=='tail':
   for r in case.get('stages',[]):report['tail_rows'].append({'id':case['id'],'name':r['name'],'seconds':r['seconds_including_io'],'different_voxels':r['comparison'].get('different_voxels'),'max_abs':r['comparison'].get('max_abs'),'p99_abs':r['comparison'].get('p99_abs')})
 if mode=='tail':
  samples=[];times=[];failed=0
  for s in j.get('gpu_samples',[]):
   times.append(s['time']);failed+=int('error' in s)
   for line in s.get('apps','').splitlines():
    fields=[v.strip() for v in line.split(',')]
    if len(fields)==3 and fields[0]==str(j['pid']):samples.append(int(fields[2])*1024**2)
  report['tail_gpu']={'scope':'observed_parent_PID_only_not_process_family_or_whole_pipeline','parent_pid':j['pid'],'peak_bytes':max(samples) if samples else None,'own_pid_samples':len(samples),'failed_samples':failed,'max_actual_gap_seconds':max((b-a for a,b in zip(times,times[1:])),default=None),'parent_child_aggregate':'not_available_no_ancestry_snapshots','no_zero_memory_claim':True}
(a.output/'collected.json').write_text(json.dumps(report,indent=2)+'\n')
for key in ('cross_rows','tail_rows'):
 rows=report[key]
 if rows:
  with open(a.output/(key+'.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(report['modes'],indent=2))
