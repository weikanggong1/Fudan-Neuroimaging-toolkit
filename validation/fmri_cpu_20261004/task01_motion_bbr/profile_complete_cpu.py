"""完整默认真实180帧的函数级CPU诊断；同CPU锁内，计时不作正式速度claim。"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--job-index',type=int)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    while True:
        current=json.loads(Path(plan['candidate_status']).read_text())
        if current['status']=='completed':break
        if current['status']=='failed':raise RuntimeError('Candidate must be investigated before extra profile')
        time.sleep(15)
    if args.job_index is None:
        for index in range(len(plan['jobs'])):
            subprocess.run([sys.executable,str(Path(__file__)),'--plan',str(args.plan),'--job-index',str(index)],check=True)
        return
    sys.path.insert(0,str(Path(__file__).parent));import benchmark_baseline as baseline
    for job in (plan['jobs'][args.job_index],):
        options=baseline.arguments(job['command'][2:])
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]=str(options.threads)
        os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,baseline.parse_cpus(options.cpu_list))
        with options.cpu_lock.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            import torch
            torch.set_num_threads(options.threads)
            if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
            sys.path.insert(0,str(options.source))
            import fnit.mcflirt.core as core
            import fnit.mcflirt._cost_cpu as ordered
            import fnit.mcflirt.sampling as sampling
            import fnit.fmri.spatial as spatial
            import nibabel as nib
            stats={};stack=ExitStack()
            def wrap(module,name,label):
                original=getattr(module,name)
                stats[label]={'calls':0,'wall_seconds':0.0}
                def call(*a,**kw):
                    start=time.perf_counter()
                    try:return original(*a,**kw)
                    finally:
                        stats[label]['wall_seconds']+=time.perf_counter()-start
                        stats[label]['calls']+=1
                stack.enter_context(patch.object(module,name,call))
            for module,name,label in ((core,'_isotropic_reference','prepare_reference_isotropic'),
                  (core,'_centre_of_gravity','centre_of_gravity'),(core.FSLMotionNormCorr,'__init__','cost_setup'),
                  (core.FSLMotionNormCorr,'__call__','cost_total_nested'),(ordered,'motion_cost_rows','ordered_numba_sampling_nested'),
                  (core.RigidAffineComposer,'__call__','rigid_composition'),(core,'fsl_coordinate_optimize','coordinate_optimizer_total_nested'),
                  (spatial,'apply_motion_warp','final_motion_resampling_total_nested'),
                  (sampling,'_coordinates','final_coordinate_generation_nested'),(sampling,'_cubic_coefficients','final_cubic_prefilter_nested'),
                  (sampling,'_sample_cubic','final_cubic_sampling_nested'),(sampling,'sample_motion_frame','final_frame_sampling_total_nested'),
                  (nib,'save','normal_nifti_save')):wrap(module,name,label)
            options.output_dir.mkdir(parents=True,exist_ok=False)
            config=json.loads(options.case_json.read_text())
            with stack:
                report=baseline._execute(options,config[options.case],config,options.output_dir/'result')
            report.update(diagnostic_full_real_profile=True,nested_clocks_must_not_be_summed=True,
                          function_wall=stats,normal_full_frames=180,
                          explicitly_no_gaussian_blur_stage='MCFLIRT reference isotropic helper has no prefilter in this implementation')
            (options.output_dir/'profile.safe.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'threads':options.threads,'api_seconds':report['api_seconds'],
                 'largest_function_wall':sorted(stats.items(),key=lambda pair:pair[1]['wall_seconds'],reverse=True)[:8]}),flush=True)

if __name__=='__main__':main()
