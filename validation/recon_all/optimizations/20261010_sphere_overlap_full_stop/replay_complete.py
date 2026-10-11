"""冻结实际注册清理输入并比较旧/新完整停止语义；不修改生产或参考输入。

显式传FNIT私有根、不可变new2源码及新候选overlap文件。实际运行完整
注册以observer保存清理调用输入；输入不是已经清理的sphere.reg。
随后同一输入CPU旧/新与新GPU回放，报告严格末态；无任意新增轮次。
"""
import argparse,hashlib,importlib.util,json,os,sys,time,threading,traceback
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):
    import numpy as np
    if hasattr(a,'detach'): a=a.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True)
    p.add_argument('--candidate-module',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); entered=time.perf_counter()
    if a.output.exists(): raise ValueError('output must be a new directory')
    a.output.mkdir(parents=True)
    r={'scope':'frozen same-input actual registration cleanup replay; not raw T1 whole',
       'status':'running','threads':a.threads,'affinity':sorted(os.sched_getaffinity(0)),
       'script_sha256':sha(__file__),'candidate_module_sha256':sha(a.candidate_module),
       'runs':[],'overall_metric_equivalence':'not_assessed'}
    save=lambda:atomic_json(a.output/'report.json',r);save()
    try:
        import numpy as np,torch,nibabel.freesurfer.io as fsio
        from numba import set_num_threads
        from fnit.recon_all import mris_register_smoothwm_run as smooth
        from fnit.recon_all.mris_register_run import run_register_sphere
        from fnit.recon_all.mris_register_overlap import remove_overlap_sphere as old
        from fnit.recon_all.profiling import ProcessTreeDeviceSampler
        torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);set_num_threads(a.threads)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        r.update(torch_version=torch.__version__,numpy_version=np.__version__,
                 device=a.device,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 cuda_allocator_cache_disabled=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')=='1',
                 old_source_sha256=sha(sys.modules[old.__module__].__file__))
        source=a.fnit_root/'runs/recon_python_profile_20261010_v4/sub06-candidate/subject/surf'
        inputs={name:source/f'lh.{name}' for name in ['sphere','smoothwm','sulc']}
        inputs['atlas']=a.fnit_root/'assets/recon_resources_20261009_v1/assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif'
        r['inputs_sha256']={k:sha(v) for k,v in inputs.items()}
        captured={}
        def observe(x,f,**kwargs):
            captured.update(positions=x.detach().cpu().numpy().copy(),
                            faces=f.detach().cpu().numpy().copy(),kwargs=kwargs)
            np.savez(a.output/'actual_cleanup_input.npz',positions=captured['positions'],
                     faces=captured['faces'],start_iteration=kwargs.get('start_iteration',0))
            captured['checkpoint_sha256']=sha(a.output/'actual_cleanup_input.npz')
            began=time.perf_counter();out,h=old(x,f,**kwargs)
            captured.update(old_output=out.detach().cpu().numpy().copy(),old_history=h,
                            old_seconds=time.perf_counter()-began)
            np.save(a.output/'old_cleanup_output.npy',captured['old_output'])
            return out,h
        smooth.remove_overlap_sphere=observe
        sampler=ProcessTreeDeviceSampler(device=a.device,parent_pid=os.getpid(),interval=.2)
        torch.cuda.init();gpu_uuid=str(torch.cuda.get_device_properties(a.device).uuid);sampler.uuid=gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-"+gpu_uuid
        stop=threading.Event()
        def sample():
            while not stop.wait(.2):sampler.sample_if_due(force=True)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        torch.cuda.synchronize(a.device);started=time.perf_counter()
        full=run_register_sphere(**inputs,output=a.output/'old_registered.sphere',
            overlap_device='cpu',averaging_device=a.device)
        torch.cuda.synchronize(a.device)
        r['old_complete_registration_seconds_including_IO']=time.perf_counter()-started
        r['old_registration']=full
        r['actual_cleanup_input_sha256']=captured['checkpoint_sha256']
        r['actual_cleanup_start_iteration']=captured['kwargs']['start_iteration']
        r['old_actual_cleanup_seconds']=captured['old_seconds']
        r['old_actual_cleanup_history']=captured['old_history']
        spec=importlib.util.spec_from_file_location('fnit.recon_all.overlap_diagnostic_candidate',a.candidate_module)
        candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
        for dev in ['cpu',a.device]:
            xyz=torch.from_numpy(captured['positions']).to(dev);faces=torch.from_numpy(captured['faces']).to(dev)
            diagnostics={}
            if dev!='cpu':torch.cuda.synchronize(dev)
            start=time.perf_counter()
            out,h=candidate.remove_overlap_sphere(xyz,faces,**captured['kwargs'],diagnostics=diagnostics)
            if dev!='cpu':torch.cuda.synchronize(dev)
            wall=time.perf_counter()-start
            values=out.cpu().numpy();delta=np.abs(values.astype(np.float64)-captured['old_output'])
            row={'device':dev,'full_cleanup_seconds_including_diagnostics':wall,
                 'history_exact_to_old':h==captured['old_history'],
                 'coords_exact_to_old':bool(np.array_equal(values,captured['old_output'])),
                 'coordinate_elements_differing':int(np.count_nonzero(delta)),
                 'coordinate_max_mm':float(delta.max(initial=0)),
                 'coordinate_p99_mm':float(np.percentile(delta,99)),
                 'coords_sha256':array_sha(values),'diagnostics':diagnostics}
            r['runs'].append(row);save()
            try:
                candidate._finish_overlap_report(out,faces,h,diagnostics=None,require_unfolded=True,
                    count=diagnostics['final_negative_float32'],iteration=diagnostics['final_iteration'],
                    start_iteration=captured['kwargs']['start_iteration'],max_iterations=1000,
                    stop_reason=diagnostics['stop_reason'],minimum_negative=diagnostics['minimum_negative_float32'],
                    minimum_iteration=diagnostics['minimum_iteration'],dt=diagnostics['final_dt'],
                    max_neighbors=diagnostics['marked_expansion_rings'])
                row['strict_gate']='passed'
            except candidate.SphereOverlapQualityError:
                row['strict_gate']='failed_quality'
        for label,path in [
            ('normal79',a.fnit_root/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub06-candidate/subject/surf/lh.sphere'),
            ('official_frozen',a.fnit_root/'datasets/benchmark_references_official_pair_20261009_v2/subjects/ds000114_sub-06/surf/lh.sphere')]:
            vertices,faces=fsio.read_geometry(path);diagnostics={}
            x=torch.from_numpy(vertices.astype(np.float32));f=torch.from_numpy(faces.astype(np.int64))
            oldout,oldh=old(x,f);out,h=candidate.remove_overlap_sphere(x,f,diagnostics=diagnostics,require_unfolded=True)
            r['runs'].append({'case':label,'input_sha256':sha(path),'history_exact_to_old':h==oldh,
                'coords_exact_to_old':bool(torch.equal(out,oldout)),'diagnostics':diagnostics});save()
        stop.set();thread.join(2);sampler.sample_if_due(force=True)
        r['gpu_process_sampler']=sampler.report()
        r['allocated_peak_bytes']=torch.cuda.max_memory_allocated(a.device)
        r['reserved_peak_bytes']=torch.cuda.max_memory_reserved(a.device)
        r['status']='complete_replay_with_failed_quality_input_preserved'
    except Exception:
        r.update(status='failed',error=traceback.format_exc())
    r['harness_wall_seconds_including_imports_checks_compute_IO']=time.perf_counter()-entered;save()
    if r['status']=='failed':raise RuntimeError(r['error'])

if __name__=='__main__':main()
