"""从已保存的完整native日志提取科学指标第一差；不执行放置，不推断坐标或编译原因。"""
import argparse,csv,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--hemi',choices=('lh','rh'),default='lh');a=p.parse_args()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rounds(path):
 text=path.read_text();result=[]
 for match in re.finditer(r'Iteration (\d+) =+\n(.*?)(?=Iteration \d+ =+|\Z)',text,re.S):
  chunk=match.group(2); events=[];boundaries=[];thresholds={};steps=[]
  for line in chunk.splitlines():
   threshold=re.match(r'\s+(inside_hi|border_hi|border_low|outside_low|outside_hi|sigma)\s*=\s*([\d.e+-]+)',line)
   if threshold:thresholds[threshold[1]]=float(threshold[2])
   if line.startswith(('#SI#','mean border=','%')):boundaries.append(line)
   if re.match(r'^\d+: dt:',line) or line.startswith('rms =') or 'RMS increased, rejecting step' in line:events.append(line)
   step=re.match(r'^(\d+): dt: ([\d.]+), sse=([\d.e+-]+), rms=([\d.e+-]+)',line)
   if step: steps.append({'step':int(step[1]),'dt':float(step[2]),'sse':float(step[3]),'rms':float(step[4]),'raw':line})
  result.append({'outer_pass':int(match[1]),'thresholds':thresholds,'boundary_records':boundaries,'optimizer_events':events,'steps':steps,
                 'rejected_trials':chunk.count('RMS increased, rejecting step'),
                 'ended_on_reduction_limit':'maximum number of reductions reached' in chunk})
 return result
report={'scope':'read_only_current_same_complete_input_native_logs','overall_equivalence':'not_assessed',
 'hemisphere':a.hemi,'script_sha256':sha(__file__),'first_difference_is_printed_scalar_not_first_coordinate_difference':True,
 'FP32_library_or_compiler_cause':'unproven_hypothesis; executable/source/library/build differences require isolated operator evidence', 'stages':{}}
rows=[]
for kind in ('prewhite','white','pial'):
 # Permit the pial subreports to be extracted with their original relative paths.
 paths={backend:a.root/f'{a.hemi}_{kind}_{backend}' for backend in ('conda','official')}
 if not all((path/'report.json').is_file() for path in paths.values()):continue
 details={backend:json.loads((path/'report.json').read_text()) for backend,path in paths.items()}
 c,o=details['conda'],details['official']
 def argv(d):
  subject=str(Path(d['output']['path']).parents[1])
  return [token.replace(subject,'SUBJECT') for token in d['command'][1:]]
 inp=c['input_sha256']==o['input_sha256']
 if not inp or argv(c)!=argv(o) or c['threads']!=o['threads']:raise ValueError('incomparable inputs or parameters')
 traces={backend:rounds(path/'native.log') for backend,path in paths.items()}
 result={'input_sha256_identical':inp,'input_sha256':c['input_sha256'],
         'inputs_unchanged_after_run':{backend:d['input_sha256']==d['input_sha256_after'] for backend,d in details.items()},
         'normalized_argv_identical':True,'normalized_argv':argv(c),'threads':c['threads'],
         'report_sha256':{backend:sha(path/'report.json') for backend,path in paths.items()},
         'native_log_sha256':{backend:sha(path/'native.log') for backend,path in paths.items()},
         'program_sha256':{backend:d['program_sha256'] for backend,d in details.items()},
         'execution_script_sha256':{backend:d['script_sha256'] for backend,d in details.items()},
         'seconds':{backend:d['seconds_including_io'] for backend,d in details.items()},'rounds':[],
         'first_optimizer_print_difference':None,'first_boundary_print_difference':None,'first_target_threshold_difference':None}
 for cr,orr in zip(traces['conda'],traces['official']):
  index=cr['outer_pass'];entry={'outer_pass':index,'thresholds_identical':cr['thresholds']==orr['thresholds'],'boundary_records_identical':cr['boundary_records']==orr['boundary_records'],
          'conda_last_completed_step':cr['steps'][-1]['step'] if cr['steps'] else None,'official_last_completed_step':orr['steps'][-1]['step'] if orr['steps'] else None,
          'conda_rejected_trials':cr['rejected_trials'],'official_rejected_trials':orr['rejected_trials'],
          'conda_ended_on_reduction_limit':cr['ended_on_reduction_limit'],'official_ended_on_reduction_limit':orr['ended_on_reduction_limit']}
  result['rounds'].append(entry)
  if not entry['thresholds_identical'] and result['first_target_threshold_difference'] is None:result['first_target_threshold_difference']={'outer_pass':index,'conda':cr['thresholds'],'official':orr['thresholds']}
  for category,key in [('optimizer_events','first_optimizer_print_difference'),('boundary_records','first_boundary_print_difference')]:
   for event_index in range(max(len(cr[category]),len(orr[category]))):
    first=cr[category][event_index] if event_index<len(cr[category]) else None
    second=orr[category][event_index] if event_index<len(orr[category]) else None
    if first!=second and result[key] is None:result[key]={'outer_pass':index,'event_index':event_index,'conda':first,'official':second}
  cm={row['step']:row for row in cr['steps']};om={row['step']:row for row in orr['steps']}
  for step in sorted(set(cm)|set(om)):
   cc,oo=cm.get(step),om.get(step)
   rows.append([kind,index,step,cc['dt'] if cc else '',oo['dt'] if oo else '',cc['sse'] if cc else '',oo['sse'] if oo else '',cc['rms'] if cc else '',oo['rms'] if oo else '',(cc['sse']-oo['sse']) if cc and oo else ''])
 report['stages'][kind]=result
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
with a.output.with_suffix('.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['stage','outer_pass','completed_step','conda_dt','official_dt','conda_sse','official_sse','conda_rms','official_rms','printed_sse_difference']);writer.writerows(rows)
