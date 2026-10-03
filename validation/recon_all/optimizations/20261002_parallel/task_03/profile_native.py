"""冻结真实同输入，采样完整 GCA/WM C++ 阶段；输出只写任务目录。"""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--commit',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
mri=a.root/'serial_20261001'/a.case/'mri';binary=a.root/'serial_20261001/native_bundle/bin/mri_em_register';atlas=a.root/'assets/average/RB_all_2020-01-02.gca'
report={'commit':a.commit,'scope':'frozen_same_input_full_native_stage_profile_not_candidate_speedup','inputs':{str(p.name):sha(p) for p in (mri/'nu.mgz',mri/'brainmask.mgz',atlas)},'binary_sha256':sha(binary),'timing_includes_perf_overhead':True}
command=['perf','record','-F','99','-g','--call-graph','dwarf,8192','-o',str((a.output/'perf.data').resolve()),'--',str(binary),'-uns','3','-mask',str(mri/'brainmask.mgz'),str(mri/'nu.mgz'),str(atlas),str((a.output/'talairach.lta').resolve())]
env=dict(os.environ,FREESURFER_HOME=str(a.root/'assets'));tick=time.perf_counter()
with (a.output/'stdout.jsonl').open('w') as log:
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    (a.output/'pid').write_text(str(proc.pid))
    for line in proc.stdout:log.write(json.dumps({'elapsed':time.perf_counter()-tick,'line':line.rstrip()})+'\n');log.flush()
    status=proc.wait()
report['seconds']=time.perf_counter()-tick;report['returncode']=status
if status==0:
    report['output_sha256']=sha(a.output/'talairach.lta')
    reference=mri/'transforms/talairach.lta';report['frozen_lta_sha256']=sha(reference)
subprocess.run(['perf','report','--stdio','--no-children','--percent-limit','0.5','-i',str(a.output/'perf.data')],stdout=(a.output/'perf_report.txt').open('w'),stderr=subprocess.STDOUT)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if status:raise SystemExit(status)
