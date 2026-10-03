"""运行子命令并同步采样整个进程树显存，不记录外部进程命令行。"""
import argparse,csv,json,os,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--interval',type=float,default=.5);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
command=a.command[1:] if a.command[:1]==['--'] else a.command;proc=subprocess.Popen(command);stop=threading.Event();rows=[];failures=[]
def descendants(root):
 parents={}
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try: fields=(entry/'stat').read_text().rsplit(')',1)[1].split();parents[int(entry.name)]=int(fields[1])
  except (OSError,IndexError,ValueError):continue
 found={root};previous=set()
 while previous!=found:
  previous=found.copy();found.update(pid for pid,parent in parents.items() if parent in found)
 return found
def monitor():
 while not stop.is_set():
  tick=time.monotonic()
  try:
   own=descendants(proc.pid);raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
   ownbytes=0;externalbytes=0;unknown=False
   for line in csv.reader(raw.splitlines()):
    pid,uuid,memory=[x.strip() for x in line]
    if uuid!=a.gpu_uuid:continue
    try: value=int(memory)*1024**2
    except ValueError:unknown=True;continue
    if int(pid) in own:ownbytes+=value
    else:externalbytes+=value
   rows.append(dict(monotonic=tick,tree_gpu_bytes=None if unknown else ownbytes,external_gpu_bytes=externalbytes,pids=sorted(own),loadavg=os.getloadavg()))
  except Exception as e:failures.append(dict(monotonic=tick,error=str(e)))
  stop.wait(max(0,a.interval-(time.monotonic()-tick)))
thread=threading.Thread(target=monitor);thread.start();exitcode=proc.wait();stop.set();thread.join()
valid=[r['tree_gpu_bytes'] for r in rows if r['tree_gpu_bytes'] is not None]
report=dict(exit_code=exitcode,gpu_uuid=a.gpu_uuid,sampling_interval_seconds=a.interval,max_interval_seconds=max([rows[i]['monotonic']-rows[i-1]['monotonic'] for i in range(1,len(rows))],default=None),tree_peak_gpu_bytes=max(valid,default=None),samples=len(rows),failures=failures,rows=rows)
(a.output/'monitor.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(exitcode)
