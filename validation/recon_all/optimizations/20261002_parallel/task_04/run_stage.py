"""任务4冻结同输入阶段复现；调用者用共用flock包裹本进程。"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,subprocess,sys,threading,time,traceback


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('source_root','checkpoint','output_root','assets','commit'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--stage',choices=('remesh','sphere','register'),required=True)
    p.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--dirty-sha256',default=None)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();root=pathlib.Path(a.source_root).resolve();sys.path.insert(0,str(root/'src'))
    out=pathlib.Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    # 测量边界包含FNIT与数值库导入、校验、冷JIT、H2D/D2H和写文件。
    t0=time.perf_counter();report={'status':'running','args':vars(a),'pid':os.getpid(),'load_before':os.getloadavg(),'affinity':sorted(os.sched_getaffinity(0)), 'tolerance_declared':{'remesh_coordinates':'exact','ordered_faces':'exact','sphere_coordinates':'exact','sphere_dt_trajectory':'exact','registration_coordinates':'exact','rigid_objective_absolute':1e-9},'whole_equivalence':'not_assessed'}
    report['source_sha256']={str(p.relative_to(root)):sha(p) for p in sorted((root/'src/fnit/recon_all').glob('*.py'))}
    samples=[];done=threading.Event()
    def monitor():
        while not done.is_set():
            tick=time.perf_counter()
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                tree={os.getpid()}
                pairs=[]
                for item in pathlib.Path('/proc').iterdir():
                    if not item.name.isdigit():continue
                    try:
                        st=(item/'stat').read_text().split(') ')[1].split();pairs.append((int(item.name),int(st[1])))
                    except (OSError,IndexError,ValueError):pass
                while True:
                    new={pid for pid,ppid in pairs if ppid in tree};old=len(tree);tree|=new
                    if len(tree)==old:break
                own,external=0,0
                for line in raw.strip().splitlines():
                    uuid,pid,mib=[x.strip() for x in line.split(',')]
                    if uuid!=a.gpu_uuid:continue
                    size=int(mib)*1024*1024
                    if int(pid) in tree:own+=size
                    else:external+=size
                samples.append({'elapsed':tick-t0,'tree_gpu_bytes':own,'external_gpu_bytes':external,'load':os.getloadavg()})
            except Exception as e:samples.append({'elapsed':tick-t0,'error':str(e)})
            done.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        import numba,torch,numpy as np
        numba.set_num_threads(a.threads);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        report['runtime']={'torch':torch.__version__,'numba':numba.__version__,'numpy':np.__version__,'numba_threads':numba.get_num_threads(),'torch_threads':torch.get_num_threads(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'autocast':False,'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')}}
        surf=pathlib.Path(a.checkpoint)/'surf';h=a.hemisphere;output=out/(h+'.'+a.stage)
        if a.stage=='remesh':
            from fnit.recon_all.mris_remesh_python import remesh_surface
            inputs={'premesh':surf/(h+'.orig.premesh')};report['input_sha256']={k:sha(v) for k,v in inputs.items()}
            tick=time.perf_counter();remesh_surface(inputs['premesh'],output,iterations=3);report['stage']={'total_seconds_including_io':time.perf_counter()-tick}
        elif a.stage=='sphere':
            from fnit.recon_all.sphere_standard_run import run_standard_sphere
            inputs={k:surf/(h+'.'+k) for k in ('inflated','smoothwm')};report['input_sha256']={k:sha(v) for k,v in inputs.items()}
            report['stage']=run_standard_sphere(inputs['inflated'],inputs['smoothwm'],output,finish_device='cpu',averaging_device=a.device)
        else:
            from fnit.recon_all.mris_register_run import run_register_sphere
            inputs={k:surf/(h+'.'+k) for k in ('sphere','smoothwm','sulc')};inputs['atlas']=pathlib.Path(a.assets)/'average'/(h+'.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif');report['input_sha256']={k:sha(v) for k,v in inputs.items()}
            report['stage']=run_register_sphere(inputs['sphere'],inputs['smoothwm'],inputs['sulc'],inputs['atlas'],output,overlap_device='cpu',averaging_device=a.device)
        if a.stage!='remesh':
            torch.cuda.synchronize(torch.device(a.device));report['torch_peak']={'allocated':torch.cuda.max_memory_allocated(a.device),'reserved':torch.cuda.max_memory_reserved(a.device)}
        report.update(status='complete',output_sha256=sha(output),command_wall_seconds=time.perf_counter()-t0)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc(),command_wall_seconds=time.perf_counter()-t0)
    finally:
        done.set();thread.join(6);report['load_after']=os.getloadavg();report['gpu_samples']=samples
        good=[s for s in samples if 'tree_gpu_bytes' in s]
        report['gpu_memory']={'sampling_interval_seconds':.25,'max_sampling_gap_seconds':max((b['elapsed']-a['elapsed'] for a,b in zip(samples,samples[1:])),default=None),'failures':len(samples)-len(good),'tree_peak_bytes':max((s['tree_gpu_bytes'] for s in good),default=None),'external_peak_bytes':max((s['external_gpu_bytes'] for s in good),default=None),'budget_bytes':20000000000,'scope':'same-instant complete process tree on target UUID; NVML MiB resolution'}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ('status','command_wall_seconds','error','gpu_memory')},indent=2),flush=True)
    if report['status']!='complete':raise SystemExit(1)

if __name__=='__main__':main()
