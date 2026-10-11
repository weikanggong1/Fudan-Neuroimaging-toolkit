"""冻结同输入、自有连续 white/pial 链；参考仅在独立目录执行。"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time, traceback
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.place_surface_observer import PlacementObserver
from fnit.recon_all.surface_placement_python import placement_options
from fnit.recon_all.thread_budget import thread_budget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, data):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def compare(first,second):
    a,f=nib.freesurfer.read_geometry(str(first));b,g=nib.freesurfer.read_geometry(str(second))
    ordered=a.shape==b.shape and np.array_equal(f,g)
    delta=np.linalg.norm(a-b,axis=1) if ordered else None
    return dict(ordered_faces=ordered,different_elements=int(np.count_nonzero(a!=b)) if ordered else None,
                max_mm=float(delta.max()) if ordered else None,p99_mm=float(np.percentile(delta,99)) if ordered else None,
                mean_mm=float(delta.mean()) if ordered else None,first_sha256=sha(first),second_sha256=sha(second))


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hemi=args.hemisphere
    from fnit.recon_all import place_white_preaparc_python as white,place_pial_python as pial
    from fnit.recon_all.place_final_white_python import place_final_white
    from fnit.recon_all.autodet_gwstats_python import write_autodet_stats
    from fnit.recon_all.white_preaparc_conda import run_white_preaparc
    from fnit.recon_all.final_white_conda import run_final_white
    from fnit.recon_all.pial_t1_conda import run_pial_t1
    report=dict(schema_version=1,base_commit=args.base_commit,candidate_commit=args.candidate_commit,script_sha256=sha(__file__),status='running',scope='frozen matched-input surface chain; no raw T1 or annotation generation',
        threads=args.threads,cpu_affinity=sorted(os.sched_getaffinity(0)),device=args.device,
        pial_cleanup_constraint_mode=args.cleanup_constraint_mode,
        source_sha256={},input_sha256={},stages=[],overall_equivalence='not_assessed',operator_threshold='not_assessed',
        native_programs_in_candidate=[],reference_program_sha256={name:sha(path) for name,path in [('conda',args.reference_binary),('official',args.official_binary)] if path},
        precision=dict(matmul_tf32=True,cudnn_tf32=True,autocast=False,low_precision=False),
        timing_scope='stage includes loading, transfers, JIT, computation, observation and writes; independent copy and reference separate')
    receipt=args.output/'report.private.json';publish(receipt,report)
    dest=args.output/'candidate';inputs=['mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz']
    if args.start_stage=='preaparc':inputs += [f'surf/{hemi}.orig',f'surf/{hemi}.orig.premesh']
    else:inputs += [f'surf/{hemi}.white',f'surf/autodet.gw.stats.{hemi}.dat']
    inputs += [f'label/{hemi}.{name}' for name in ('cortex.label','cortex+hipamyg.label')]
    if args.start_stage=='preaparc' or args.reference_binary or args.official_binary:
        inputs.append(f'label/{hemi}.aparc.annot')
    report['input_sha256']={name:sha(args.subject/name) for name in inputs}
    tick=time.perf_counter()
    for name in inputs:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.subject/name,target)
    report['input_copy_seconds']=time.perf_counter()-tick
    options=placement_options(device=args.device)
    if args.target_search_backend:options['target_search_backend']=args.target_search_backend
    stages=['preaparc','white','pial'] if args.start_stage=='preaparc' else ['pial']
    outputs={'preaparc':'white.preaparc','white':'white','pial':'pial.T1'}
    references={'preaparc':run_white_preaparc,'white':run_final_white,'pial':run_pial_t1}
    tick=time.perf_counter()
    memory=None
    try:
        with thread_budget(threads=args.threads):
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            if args.device.startswith('cuda:'):
                torch.empty(1,device=args.device);torch.cuda.synchronize(args.device)
                from fnit.recon_all.place_surface_benchmark_memory import PlacementMemorySampler
                memory=PlacementMemorySampler(uuid=args.gpu_uuid).start()
            for name in stages:
                stage_input=dest/f'surf/{hemi}.{dict(preaparc="orig",white="white.preaparc",pial="white")[name]}'
                row=dict(stage=name,stage_input_sha256=sha(stage_input),references={});report['stages'].append(row);publish(receipt,report)
                # Reference clones contain candidate's actual input for this stage; their outputs never return to candidate.
                for label,binary in [('conda',args.reference_binary),('official',args.official_binary)]:
                    if binary is None:continue
                    reference=args.output/f'{name}-{label}';shutil.copytree(dest,reference)
                    started=time.perf_counter()
                    with (args.output/f'{name}-{label}.private.log').open('w') as log:
                        code='import sys;from fnit.recon_all.'+dict(preaparc='white_preaparc_conda',white='final_white_conda',pial='pial_t1_conda')[name]+' import '+references[name].__name__+' as run;run(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],threads=int(sys.argv[5]))'
                        reference_environment=dict(os.environ)
                        if args.reference_license:reference_environment['FS_LICENSE']=str(args.reference_license)
                        child=subprocess.run([os.sys.executable,'-c',code,str(reference),hemi,str(binary),str(args.assets),str(args.threads)],env=reference_environment,stdout=log,stderr=subprocess.STDOUT)
                    row['references'][label]=dict(exit_code=child.returncode,wall_seconds=time.perf_counter()-started)
                    if child.returncode:raise RuntimeError('independent reference failed')
                if name=='preaparc':
                    write_autodet_stats(dest/'mri/brain.finalsurfs.mgz',dest/'mri/wm.mgz',dest/f'surf/{hemi}.orig.premesh',dest/f'surf/autodet.gw.stats.{hemi}.dat',hemi)
                output=dest/f'surf/{hemi}.{outputs[name]}'
                target_module=pial if name=='pial' else white
                if args.device.startswith('cuda:'):torch.cuda.reset_peak_memory_stats(args.device);torch.cuda.synchronize(args.device)
                started=time.perf_counter();original_popen=subprocess.Popen
                candidate_started=started
                def deny(*a,**k):raise RuntimeError('candidate attempted external process')
                try:
                    subprocess.Popen=deny
                    with PlacementObserver(stage_module=target_module,directory=args.output/f'{name}-observation') as observer:
                        if name=='preaparc':result=white.place_white_preaparc(subject_dir=dest,hemi=hemi,output=output,output_volume=dest/'mri/mrisps.wpa.mgz',max_steps=400,trace_callback=observer.callback,**options)
                        elif name=='white':result=place_final_white(subject_dir=dest,hemi=hemi,output=output,output_volume=dest/'mri/mrisps.white.mgz',max_steps=400,trace_callback=observer.callback,**options)
                        else:result=pial.place_pial_t1(subject=dest,hemisphere=hemi,output=output,max_steps=400,trace_callback=observer.callback,profile=True,cleanup_constraint_mode=args.cleanup_constraint_mode,**options)
                finally:subprocess.Popen=original_popen
                if args.device.startswith('cuda:'):torch.cuda.synchronize(args.device)
                row.update(result=result,wall_seconds=time.perf_counter()-started,output_sha256=sha(output))
                if args.device.startswith('cuda:'):row.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
                for label in row['references']:row['references'][label]['geometry']=compare(args.output/f'{name}-{label}/surf/{hemi}.{outputs[name]}',output)
                publish(receipt,report)
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if hasattr(error, 'cleanup_diagnostics'):
            report['failed_cleanup'] = error.cleanup_diagnostics
        if 'candidate_started' in locals():
            row['wall_seconds_to_failed_candidate_gate']=time.perf_counter()-candidate_started
            if args.device.startswith('cuda:'):
                torch.cuda.synchronize(args.device)
                row.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                           torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        if hasattr(error,'intersection_coordinates'):np.save(args.output/'failed_cleanup.private.npy',error.intersection_coordinates);report['failed_stage']=getattr(error,'partial_stage',None)
        raise
    finally:
        if memory is not None:report['nvml_memory']=memory.finish()
        report['total_compute_diagnostic_seconds']=time.perf_counter()-tick
        report['input_unchanged']=report['input_sha256']=={name:sha(args.subject/name) for name in inputs}
        root=Path(white.__file__).parents[1]
        report['source_sha256']={str(Path(module.__file__).relative_to(root)):sha(module.__file__) for module in list(os.sys.modules.values()) if getattr(module,'__name__','').startswith('fnit.recon_all') and getattr(module,'__file__',None) and str(module.__file__).endswith('.py')}
        publish(receipt,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hemisphere',choices=['lh','rh'],required=True);parser.add_argument('--start-stage',choices=['preaparc','pial'],default='preaparc')
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--base-commit',required=True);parser.add_argument('--candidate-commit',required=True);parser.add_argument('--reference-license',type=Path);parser.add_argument('--target-search-backend',choices=['serial','parallel'])
    parser.add_argument('--reference-binary',type=Path);parser.add_argument('--official-binary',type=Path);parser.add_argument('--assets',type=Path)
    parser.add_argument('--cleanup-constraint-mode',choices=['source_rip','cortex_pinned_union'],default='source_rip')
    run(parser.parse_args())
if __name__=='__main__':main()
