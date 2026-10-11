"""独立真实检查点/完整球面配准回归；不续跑 recon-all、不调用原版程序。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time, traceback
from pathlib import Path
STARTED=time.perf_counter()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['checkpoint','full'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--baseline-source',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--checkpoint-function',choices=['corner','face'],default='corner')
    p.add_argument('--subject',type=Path)
    p.add_argument('--atlas',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hemi',default='lh')
    p.add_argument('--device',default='cuda:5')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'scope':'frozen real same-input stage; not whole pipeline','mode':a.mode,
            'source_manifest_sha256':sha(a.source/'STAGE_SOURCE_BINDING.json'),
            'device':a.device,'threads':a.threads,'affinity':sorted(os.sched_getaffinity(0)),
            'status':'running','normal_vertex_zero_sum_policy':'explicit ValueError; source five-round jitter branch remains unimplemented',
            'original_FS_stage_programs':[]}
    try:
        import numpy as np, torch
        import nibabel.freesurfer.io as fsio
        from fnit.recon_all.profiling import ProcessTreeDeviceSampler
        from fnit.recon_all import mris_register_nonlinear as current
        torch.set_num_threads(a.threads)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        report.update(torch_version=torch.__version__,numpy_version=np.__version__,
                      precision={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                                 'autocast':torch.is_autocast_enabled(),'dtype':'float32 with existing FP64 intermediates','fp16_bf16':False},
                      allocator_cache_disabled=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')=='1',
                      selected_module_sha256=sha(current.__file__))
        if a.mode=='checkpoint':
            import importlib.util
            old_path=a.baseline_source/'src/fnit/recon_all/mris_register_nonlinear.py'
            spec=importlib.util.spec_from_file_location('fnit.recon_all._old_registration_replay',old_path)
            old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
            data=np.load(a.checkpoint)
            pos=torch.from_numpy(data['arg0']);faces=torch.from_numpy(data['arg1'])
            if a.checkpoint_function=='corner':
                tick=time.perf_counter();before=old.sphere_vertex_normals(pos,faces);old_seconds=time.perf_counter()-tick
                tick=time.perf_counter();after=current.sphere_vertex_normals(pos,faces);new_seconds=time.perf_counter()-tick
                old_array,new_array=before.numpy(),after.numpy()
                ordinary=np.isfinite(old_array).all(axis=1)
                assert int((~ordinary).sum())==4
                assert int((~np.isfinite(old_array)).sum())==12
                reproduced=np.array_equal(old_array,data['output0'],equal_nan=True)
            else:
                tick=time.perf_counter();old_areas,before=old.face_area_normals(pos,faces,signed_sphere=True);old_seconds=time.perf_counter()-tick
                tick=time.perf_counter();new_areas,after=current.face_area_normals(pos,faces,signed_sphere=True);new_seconds=time.perf_counter()-tick
                old_array,new_array=before.numpy(),after.numpy()
                first=pos[faces[:,1]]-pos[faces[:,0]];second=pos[faces[:,2]]-pos[faces[:,0]]
                cross=torch.stack((first[:,1]*second[:,2]-first[:,2]*second[:,1],first[:,2]*second[:,0]-first[:,0]*second[:,2],first[:,0]*second[:,1]-first[:,1]*second[:,0]),dim=1)
                squared=cross[:,0]*cross[:,0]+cross[:,1]*cross[:,1]+cross[:,2]*cross[:,2]
                length=squared.double().sqrt().float().numpy()
                ordinary=np.isfinite(old_array).all(axis=1)&(length>=np.finfo(np.float32).eps)
                tiny=length<np.finfo(np.float32).eps
                report.update(zero_cross_faces=int((length==0).sum()),FZERO_faces=int(tiny.sum()),normal_faces_checked=int(ordinary.sum()),
                    finite_small_cross_changed_faces=int((np.any(old_array!=new_array,axis=1)&np.isfinite(old_array).all(axis=1)&tiny).sum()),
                    areas_exact=bool(np.array_equal(old_areas.numpy(),new_areas.numpy())))
                reproduced=np.array_equal(old_areas.numpy(),data['output0'],equal_nan=True)&np.array_equal(old_array,data['output1'],equal_nan=True)
                assert report['zero_cross_faces']==2 and int((~np.isfinite(old_array)).sum())==6
                assert report['areas_exact']
            valid=np.isfinite(old_array).all(axis=1)
            delta=np.abs(old_array[ordinary].astype(np.float64)-new_array[ordinary].astype(np.float64))
            report.update(checkpoint_function=a.checkpoint_function,checkpoint_sha256=sha(a.checkpoint),baseline_module_sha256=sha(old_path),
                input_shapes={'positions':list(pos.shape),'faces':list(faces.shape)},
                old_nonfinite_elements=int((~np.isfinite(old_array)).sum()),old_nonfinite_vertices=int((~valid).sum()),
                new_nonfinite_elements=int((~np.isfinite(new_array)).sum()),
                previously_finite_vertices=int(valid.sum()),previously_finite_changed_elements=int((old_array[valid]!=new_array[valid]).sum()),
                ordinary_changed_elements=int((old_array[ordinary]!=new_array[ordinary]).sum()),
                ordinary_max_abs_error=float(delta.max(initial=0)),ordinary_p99_abs_error=float(np.percentile(delta,99)),
                checkpoint_old_output_reproduced=bool(reproduced),old_cpu_seconds=old_seconds,new_cpu_seconds=new_seconds,
                new_normal_norm_min=float(np.linalg.norm(new_array.astype(np.float64),axis=1).min()),
                new_normal_norm_max=float(np.linalg.norm(new_array.astype(np.float64),axis=1).max()))
            assert report['new_nonfinite_elements']==0 and report['ordinary_changed_elements']==0
            assert report['checkpoint_old_output_reproduced']
        else:
            from fnit.recon_all.mris_register_run import run_register_sphere
            module_root=a.source/'src/fnit/recon_all'
            module_paths=['mris_register_run.py','mris_register_nonlinear.py','mris_register_sulc_run.py',
                          'mris_register_smoothwm_run.py','mris_register_average_numba.py','mris_register_average_gpu.py']
            report['module_sha256']={name:sha(module_root/name) for name in module_paths}
            inputs={name:a.subject/'surf'/f'{a.hemi}.{name}' for name in ['sphere','smoothwm','sulc']}
            inputs['atlas']=a.atlas
            report['input_sha256_before']={name:sha(path) for name,path in inputs.items()}
            torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
            sampler=ProcessTreeDeviceSampler(device=a.device,parent_pid=os.getpid(),interval=.5)
            stop=threading.Event()
            def sample():
                while not stop.is_set():
                    sampler.sample_if_due(force=True);stop.wait(.5)
            sampler.sample_if_due(force=True);thread=threading.Thread(target=sample,daemon=True);thread.start()
            tick=time.perf_counter()
            try:
                report['registration']=run_register_sphere(sphere=inputs['sphere'],smoothwm=inputs['smoothwm'],
                    sulc=inputs['sulc'],atlas=inputs['atlas'],output=a.output/'sphere.reg',
                    overlap_device='cpu',averaging_device=a.device)
                torch.cuda.synchronize(a.device)
                report['complete_stage_API_wall_seconds']=time.perf_counter()-tick
            finally:
                stop.set();thread.join();sampler.sample_if_due(force=True)
                report['gpu_process_sampler']=sampler.report()
                report['allocated_peak_bytes']=torch.cuda.max_memory_allocated(a.device)
                report['reserved_peak_bytes']=torch.cuda.max_memory_reserved(a.device)
            report['input_sha256_after']={name:sha(path) for name,path in inputs.items()}
            assert report['input_sha256_before']==report['input_sha256_after']
            v,f=fsio.read_geometry(str(a.output/'sphere.reg'));iv,iff=fsio.read_geometry(str(inputs['sphere']))
            tri=v[f].astype(np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);centers=tri.mean(axis=1)
            signed=np.einsum('ij,ij->i',cross,centers)
            edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
            unique,counts=np.unique(edges,axis=0,return_counts=True)
            report['qc']={'finite_coordinates':bool(np.isfinite(v).all()),'vertices':len(v),'faces':len(f),
                'same_vertex_count':len(v)==len(iv),'ordered_faces_identical':bool(np.array_equal(f,iff)),
                'euler':int(len(v)-len(unique)+len(f)),'boundary_edges':int((counts==1).sum()),
                'nonmanifold_edges':int((counts>2).sum()),'zero_area_faces_double':int((np.linalg.norm(cross,axis=1)==0).sum()),
                'negative_radial_faces_double':int((signed<0).sum()),
                'radius_min_mm':float(np.linalg.norm(v,axis=1).min()),'radius_max_mm':float(np.linalg.norm(v,axis=1).max()),
                'source_overlap_negative_counts':report['registration']['smoothwm_pass']['negative_counts']}
            assert report['qc']['finite_coordinates'] and report['qc']['ordered_faces_identical']
            report['output_sha256']=sha(a.output/'sphere.reg')
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['harness_wall_seconds_including_imports_checks_compute_IO_QC']=time.perf_counter()-STARTED
        save(a.output/'report.json',report)

if __name__=='__main__': main()
