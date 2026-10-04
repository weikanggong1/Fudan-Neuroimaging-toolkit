"""只读检查五项精度任务的后台进程、共享锁与已有回执。

输入 --root：本轮 gpucw1 验证根目录；输出 stdout JSON（UTC、PID、状态、SHA）。
先读固定 FNIT 服务器索引，再沿 --root 指定的已冻结任务路径读取。
不启动、暂停或终止任务，不读取影像像素、许可证内容或进程环境。
病例占位仅计为已开始；完成数量要求对应阶段比较字段存在。
路径或JSON损坏会抛异常；进程不存在与回执缺失分别显式记录。
示例：python inspect_background.py --root /path/to/accuracy_20261003
本工具是调度诊断，无独立原软件等价命令；空间与长度单位不适用。
"""
import argparse,json,pathlib,datetime,subprocess,hashlib,os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=pathlib.Path,required=True,help='本轮服务器验证根目录')
R=parser.parse_args().root
server_root=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
location_documents={}
for name in ('README.md','INDEX.md','INDEX.json'):
 p=server_root/name
 raw=p.read_bytes()
 raw.decode('utf-8')
 location_documents[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 if name=='INDEX.json':json.loads(raw)
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'tasks':{},'whole':{},'priority':{}}
out['server_root']=str(server_root)
out['location_documents']=location_documents
pids={'task_01':[32848,100248],'task_02':[90754,109621,74294],'task_03':[94174,98758,114339,115793],'task_04':[23673,34538,4972],'task_05':[78930]}
ps=subprocess.run(['ps','-eo','pid,ppid,stat,etime,pcpu,comm,args'],capture_output=True,text=True,check=True).stdout
pr={}
for l in ps.splitlines()[1:]:
 v=l.split(None,6)
 if len(v)==7:pr[int(v[0])]={'pid':int(v[0]),'ppid':int(v[1]),'stat':v[2],'elapsed':v[3],'cpu_percent':v[4],'comm':v[5],'command':v[6]}
locks=pathlib.Path('/proc/locks').read_text()
lock_path=pathlib.Path('/tmp/fnit-shared-benchmark.lock')
lock_inode=str(lock_path.stat().st_ino) if lock_path.exists() else None
out['benchmark_lock_inode']=lock_inode
out['benchmark_lock_lines']=[l for l in locks.splitlines() if lock_inode is not None and any(t.endswith(':'+lock_inode) for t in l.split())]
for k,ids in pids.items():
 out['tasks'][k]={'processes':[]}
 for pid in ids:
  row=pr.get(pid,{'pid':pid,'state':'absent'})
  if pid in pr:
   descendants={pid}
   for _ in range(8):descendants.update(x for x,v in pr.items() if v['ppid'] in descendants)
   row['children']=[{x:v[x] for x in ('pid','ppid','stat','elapsed','cpu_percent','comm','command')} for v in pr.values() if v['pid'] in descendants and v['pid']!=pid]
  out['tasks'][k]['processes'].append(row)
reports={
 'task_01':['task_01/pair_baseline_official_v1/ds000030_sub-10159/checkpoint.json','/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/evaluation/ds000030_sub-10171/checkpoint.json'],
 'task_02':['task_02/cohort_prefix/report.json'],
 'task_03':['task_03/frozen_v1/normalize/report.json','task_03/frozen_v1/cross/report.json','task_03/frozen_v1/tail/report.json','task_03/frozen_v1/coordinates_fp32/report.json','task_03/production_fix_abba_v1/report.json'],
 'task_04':['task_04/queue_status.json','task_04/different_meshes_sub01/report.json','task_04/seeded_sphere_sub01_lh_v1/receipt.json'],
 'task_05':['task_05/placement_receipt.json','task_05/receipt.json']}
for k,paths in reports.items():
 out['tasks'][k]['receipts']=[]
 for f in paths:
  p=R/f
  if not p.is_file():out['tasks'][k]['receipts'].append({'file':f,'present':False});continue
  b=p.read_bytes();j=json.loads(b)
  row={'file':f,'present':True,'sha256':hashlib.sha256(b).hexdigest(),'status':j.get('status'),'complete':j.get('complete'),'keys':list(j)}
  if 'cohort_prefix' in f:
   row['cases_completed']=len(j.get('cases',[]));row['ids']=[x.get('id') for x in j.get('cases',[])]
  elif 'cross/report' in f:
   row['combinations_completed']=sum('norm_vs_official' in r for c in j.get('cases',[]) for r in c.get('runs',[]))
  elif 'tail/report' in f:row['stages_completed']=sum(len(c.get('stages',[])) for c in j.get('cases',[]))
  elif 'coordinates_fp32' in f:
   row['cases_started']=len(j.get('cases',[]));row['cases_completed']=sum('norm' in c and 'controls' in c for c in j.get('cases',[]))
  elif 'queue_status' in f:
   row['commands']=[{a:v for a,v in x.items() if a in ['case','hemi','subject','hemisphere','kind','stage','status','exit_code','seconds']} for x in j.get('commands',[])]
  else: row['summary']={a:v for a,v in j.items() if not isinstance(v,(list,dict))}
  out['tasks'][k]['receipts'].append(row)
for f in ['priority_v1/status.json','priority_v1/completion.json']:
 p=R/f
 if p.is_file():out['priority'][f]=json.loads(p.read_text())
for role in ['baseline','official']:
 rows=[]
 for p in sorted((R/'configs_v1').glob(role+'_*.json')):
  c=json.loads(p.read_text());d=pathlib.Path(c['diagnostic_root']);q=d/'completion.json'
  row={'id':p.stem.removeprefix(role+'_'),'state':'not_started'}
  if q.is_file():
   value=json.loads(q.read_text());row.update(value);row['state']=value.get('execution_status','unknown')
  elif (d/'launch.json').is_file():row['state']='running'
  rows.append(row)
 out['whole'][role]=rows
desc={54960}
for _ in range(9):desc.update(x for x,v in pr.items() if v['ppid'] in desc)
out['official_active']=[v for v in pr.values() if v['pid'] in desc and (v['stat'].startswith('R') or v['comm'].startswith('mris') or v['comm'].startswith('mri_'))]
for x in [51563,54959,104896]:out.setdefault('queue_processes',[]).append(pr.get(x,{'pid':x,'state':'absent'}))
print(json.dumps(out))

