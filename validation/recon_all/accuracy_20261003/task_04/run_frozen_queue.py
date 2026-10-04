"""服务器隔离诊断队列；逐阶段取得并释放全局资源锁。"""
from __future__ import annotations
import fcntl, hashlib, json, os, pathlib, subprocess, sys, time
BASE=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ROOT=BASE/'accuracy_20261003/task_04'
PYTHON=BASE/'fnit_main_env/bin/python'
SOURCE=BASE/'accuracy_20261003/baseline_runtime_816e5610'
FS=pathlib.Path('/public/software/apps/Freesurfer/8.2.0-1')
COMMIT='816e5610417a4c587caf321049438a9554139016'
CHECKPOINTS={'sub01':BASE/'parallel_20261002/whole_sub01_candidate_8d750e2','sub02':BASE/'parallel_20261002/whole_sub02_candidate_8d750e2_retry_v3'}

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    license_path=os.environ.get('FS_LICENSE')
    if not license_path or not pathlib.Path(license_path).is_file():raise RuntimeError('Set FS_LICENSE to the existing private license path; license content is never read by this driver')
    os.chdir(ROOT);env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',FREESURFER_HOME=str(FS))
    receipts=[];report={'pid':os.getpid(),'started':time.time(),'code_commit':COMMIT,'kind':'same-input isolated stage replay','checkpoint_code':'8d750e2 frozen previous whole outputs; not 816e5610 continuous chain','overall_equivalence':'not_assessed','status':'running','commands':receipts}
    (ROOT/'queue_status.json').write_text(json.dumps(report,indent=2)+'\n')
    jobs=[]
    for subject,checkpoint in CHECKPOINTS.items():
        for hemi in ('lh','rh'):
            # 首先验证同pretess输入生成的raw quad和main component。
            code='255' if hemi=='lh' else '127';out=ROOT/'frozen_v2'/subject/hemi/'tessellation';out.mkdir(parents=True,exist_ok=True)
            raw=out/(hemi+'.orig.raw.quad');nofix=out/(hemi+'.orig.nofix')
            jobs.extend([(subject,hemi,'tessellate','official',[str(FS/'bin/mri_tessellate'),str(checkpoint/'mri'/('filled-pretess'+code+'.mgz')),code,str(raw)],out),(subject,hemi,'extract','official',[str(FS/'bin/mris_extract_main_component'),str(raw),str(nofix)],out)])
            for stage in ('remesh','sphere','register'):
                for kind in ('official','fnit'):
                    out=ROOT/'frozen_v2'/subject/hemi/stage/kind
                    if kind=='official':cmd=[str(PYTHON),str(ROOT/'run_reference.py'),'--checkpoint',str(checkpoint),'--reference-home',str(FS),'--assets',str(BASE/'assets'),'--output-root',str(out),'--hemisphere',hemi,'--stage',stage]
                    else:cmd=[str(PYTHON),str(ROOT/'run_stage.py'),'--source-root',str(SOURCE),'--checkpoint',str(checkpoint),'--output-root',str(out),'--assets',str(BASE/'assets'),'--commit',COMMIT,'--stage',stage,'--hemisphere',hemi,'--device','cuda:0','--gpu-uuid','GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e','--threads','4']
                    jobs.append((subject,hemi,stage,kind,cmd,out))
    for subject,hemi,stage,kind,cmd,out in jobs:
        out.mkdir(parents=True,exist_ok=True);item={'subject':subject,'hemisphere':hemi,'stage':stage,'kind':kind,'command':cmd,'program_sha256':sha(cmd[0]),'status':'waiting_lock','log':str(out/(stage+'_'+kind+'.log'))};receipts.append(item);(ROOT/'queue_status.json').write_text(json.dumps(report,indent=2)+'\n')
        with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);item['start']=time.time();item['status']='running'
            with open(item['log'],'w') as log:
                child=subprocess.Popen(cmd,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;(ROOT/'queue_status.json').write_text(json.dumps(report,indent=2)+'\n');item['exit_code']=child.wait()
            item['seconds']=time.time()-item['start'];item['status']='complete' if item['exit_code']==0 else 'failed'
            fcntl.flock(lock,fcntl.LOCK_UN)
        (ROOT/'queue_status.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='complete' if all(r['status']=='complete' for r in receipts) else 'failed';report['ended']=time.time();(ROOT/'queue_status.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
