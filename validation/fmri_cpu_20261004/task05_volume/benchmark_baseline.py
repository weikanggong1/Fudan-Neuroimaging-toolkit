"""完整真实 volume/helper CPU benchmark；不进入生产运行依赖。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from unittest.mock import patch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def cpus(value):
    return {int(v) for v in value.split(',')}


def publish(path, record):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='冻结工作树根目录')
    p.add_argument('--source-revision', required=True)
    p.add_argument('--reference-tools-root', type=Path,
                   help='独立官方包装工具目录；默认使用本仓库 validation/fmri')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-list', required=True)
    p.add_argument('--cpu-lock', type=Path, required=True)
    p.add_argument('--threads', type=int, choices=(1,8), required=True)
    p.add_argument('--function', choices=('volume','feat','temporal','stc','mask','sampling_reference','legacy_motion_warp'), required=True)
    p.add_argument('--backend', choices=('fnit','official'), default='fnit')
    p.add_argument('--registration-backend', choices=('fnirt','synthmorph'), default='fnirt')
    p.add_argument('--warm-repeats', type=int, default=0)
    p.add_argument('--cutoff-seconds', type=float, default=100.)
    p.add_argument('--target-median', type=float, default=10000.)
    p.add_argument('--remove-mean', action='store_true')
    p.add_argument('--voxel-chunk', type=int, default=8192)
    p.add_argument('--slice-time-reference', type=float, default=.5)
    p.add_argument('--ignore', type=int, default=0)
    p.add_argument('--interpolation', choices=('linear','spline'), default='spline')
    p.add_argument('--pipeline-stc', action='store_true')
    p.add_argument('--no-anatomical-cache', action='store_true')
    return p


class StageObserver:
    """在原函数外计真实完整调用；不替换任何算法或输入。"""
    def __init__(self, output):
        self.output = output
        self.stack = ExitStack()
        self.records = []

    def wrap(self, module, name, label):
        original = getattr(module, name)
        def invoke(*args, **kwargs):
            row = {'stage': label, 'status': 'running', 'start_unix': time.time()}
            self.records.append(row)
            publish(self.output / 'stages.safe.json', self.records)
            start = time.perf_counter()
            try:
                value = original(*args, **kwargs)
            except BaseException:
                row.update(status='failed', wall_seconds=time.perf_counter()-start)
                publish(self.output / 'stages.safe.json', self.records)
                raise
            row.update(status='complete', wall_seconds=time.perf_counter()-start)
            publish(self.output / 'stages.safe.json', self.records)
            return value
        self.stack.enter_context(patch.object(module, name, invoke))

    def __enter__(self):
        import fnit.fmri.end_to_end as e2e
        import fnit.fmri.pipeline as feat
        import fnit.fmri.bbr as bbr
        import fnit.fmri._anatomical as anatomical
        import fnit.fmri.normalization as normalization
        for name in ('prepare_anatomical','run_feat_core','run_aroma_pipeline','_resample_final_volume'):
            self.wrap(e2e, name, name)
        self.wrap(bbr, 'register_bbr', 'register_bbr')
        self.wrap(feat, 'grand_mean_scale', 'grand_mean_scale')
        self.wrap(feat, 'gaussian_highpass', 'gaussian_highpass')
        self.wrap(anatomical, 'register_t1_to_mni', 'register_t1_to_mni')
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new task-local output directory')
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, cpus(args.cpu_list))
    if os.sched_getaffinity(0) != cpus(args.cpu_list):
        raise RuntimeError('CPU allocation mismatch')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # official fMRIPrep 的并发采样同样把BLAS子线程限制为1，避免嵌套池超预算。
    if args.function == 'volume':
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(args.source / 'src'))
    imports_start = time.perf_counter()
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    import_seconds = time.perf_counter()-imports_start
    config = json.loads(args.inputs.read_text())
    reference_root = (args.reference_tools_root or
                      Path(__file__).resolve().parents[2] / 'fmri').resolve()
    config['reference_tools_root'] = str(reference_root)
    preparation_start=time.perf_counter()
    if args.backend=='official':
        image_hash=sha256(config['fmriprep_image'])
        if image_hash!=config['expected_fmriprep_sha256']:raise ValueError('Frozen official container SHA-256 mismatch')
        if args.function=='volume':
            copied_cache=args.output/'templateflow'
            shutil.copytree(config['official_templateflow_cache'],copied_cache,symlinks=False)
            config['official_templateflow_cache']=str(copied_cache)
    source_before = {str(p.relative_to(args.source)):sha256(p) for p in (args.source / 'src/fnit').rglob('*.py')}
    input_paths = {'raw_BOLD':config['public180']['bold'], 'raw_T1w':config['public180']['t1w']}
    if args.function == 'temporal':
        input_paths = {k:config['temporal490'][k] for k in ('official_corrected','brain_mask')}
    elif args.function == 'sampling_reference':
        input_paths = {k:config[k] for k in ('sampling_t1','sampling_moving','sampling_t1_mask')}
    elif args.function == 'volume':
        for key in ('mni_template','mni_brain_mask','synthstrip_weights'):
            input_paths[key] = config['public180']['pipeline'][key]
        if args.registration_backend == 'synthmorph':input_paths['synthmorph_weights'] = config['synthmorph_weights']
    elif args.function == 'legacy_motion_warp':
        input_paths={k:config['legacy490'][k] for k in ('bold','reference')}
    elif args.function == 'mask':input_paths={'reference':config['public180']['reference']}
    input_hashes={key:{'bytes':Path(value).stat().st_size,'sha256':sha256(value)} for key,value in input_paths.items()}
    metadata_sha256=hashlib.sha256(json.dumps(config['public180']['metadata'],sort_keys=True).encode()).hexdigest()
    report = {'function':args.function,'backend':args.backend,'source_revision':args.source_revision,
              'cpu_threads':args.threads,'cpu_affinity':sorted(cpus(args.cpu_list)),
              'import_seconds':import_seconds,'adapter_sha256':sha256(__file__),
              'source_sha256':source_before,'input_sha256':input_hashes,'raw_public_metadata_sha256':metadata_sha256,'records':[],'default_slice_timing':False,
              'outside_timer_preparation_seconds':time.perf_counter()-preparation_start,
              'software':{'torch':torch.__version__,'numpy':np.__version__,'nibabel':nib.__version__}}
    if args.backend == 'official':
        report['reference_tools_sha256'] = {
            name: sha256(reference_root / name)
            for name in ('native_exec.py', 'e2e_latest/make_original_sampling_reference.py')}
    import fcntl
    args.cpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.cpu_lock.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for repeat in range(args.warm_repeats+1):
            out = args.output / f'repeat_{repeat}'
            out.mkdir()
            row = {'repeat':repeat,'call_type':'first_call' if repeat==0 else 'warm_same_process_fnit_call' if args.backend=='fnit' else 'fresh_official_process_with_workflow_cache' if args.function=='volume' and repeat>0 else 'fresh_official_process'}
            if args.backend == 'fnit':
                with StageObserver(out) as observer:
                    start = time.perf_counter()
                    metadata = fnit_call(args, config, out, repeat)
                    row['api_wall_seconds_including_io'] = time.perf_counter()-start
                row['stages'] = observer.records
            else:
                start = time.perf_counter()
                metadata = official_call(args, config, out, repeat)
                row['official_wall_seconds_including_io'] = time.perf_counter()-start
            row.update(metadata)
            report['records'].append(row)
            publish(args.output / 'report.safe.json', report)
    report.update(status='complete',source_unchanged=all(sha256(args.source / p)==h for p,h in source_before.items()),
                  timing_scope='Complete real helper/API with normal I/O; imports, source/input hashes and numerical comparison are separately counted. Stage clocks are nested and cannot be added to API.')
    publish(args.output / 'report.safe.json', report)
    print(json.dumps({'status':report['status'],'function':args.function,'records':len(report['records'])}))


def fnit_call(args, config, out, repeat):
    import nibabel as nib
    import numpy as np
    p = config['public180']
    if args.function == 'volume':
        from fnit.fmri import fMRIVolume_pipeline
        kwargs = dict(p['pipeline'])
        kwargs.update(device='cpu',registration_backend=args.registration_backend,
                      slice_timing=args.pipeline_stc,slice_time_reference=args.slice_time_reference,
                      reuse_anatomical=not args.no_anatomical_cache,overwrite=repeat>0)
        if args.registration_backend == 'synthmorph':
            kwargs.pop('fnirt_config',None)
            kwargs['synthmorph_weights'] = config['synthmorph_weights']
        root = args.output / 'derivatives'
        result = fMRIVolume_pipeline(derivatives_root=root, **kwargs)
        return {'pipeline_seconds':result.timing_seconds,
                'anatomical_cache_reused':json.loads(result.metadata.read_text()).get('FNIT',{}).get('Configuration',{}).get('anatomical_cache',{}).get('reused'),
                'all_four_volume_outputs_exist':all(getattr(result,k).is_file() for k in ('clean_native','clean_mni','preproc_t1w','preproc_mni')),
                'complete_frames':p['bold_shape'][3], 'registration_backend':args.registration_backend,
                'scope':'Original raw BIDS T1/BOLD; complete preproc+clean and default ICA/AROMA, no surface.'}
    if args.function == 'feat':
        from fnit.fmri.pipeline import run_feat_core
        kwargs={k:p['pipeline'][k] for k in ('bids_root','subject','session','task','synthstrip_weights') if k in p['pipeline']}
        result=run_feat_core(output_dir=out/'feat',device='cpu',**kwargs)
        return {'intensity_factor':result.intensity_factor,'complete_frames':p['bold_shape'][3]}
    if args.function == 'temporal':
        from fnit.feat.temporal import scale_nifti,highpass_nifti
        c=config['temporal490']
        scaled=out/'scaled.nii.gz'
        factor=scale_nifti(c['official_corrected'],c['brain_mask'],scaled,target_median=args.target_median)
        highpass_nifti(scaled,out/'highpass.nii.gz',cutoff_seconds=args.cutoff_seconds,
                      tr_seconds=c['tr'],device='cpu',voxel_chunk=args.voxel_chunk,preserve_mean=not args.remove_mean)
        return {'intensity_factor':factor,'complete_frames':490,'preserve_mean':not args.remove_mean,
                'scope':'Fixed complete real MCFLIRT output; grand-mean scaling then Gaussian highpass.'}
    if args.function == 'stc':
        from fnit.fmri.slice_timing import slice_timing_correct
        _,details=slice_timing_correct(p['bold'],out/'stc.nii.gz',p['metadata'],
            reference_fraction=args.slice_time_reference,ignore=args.ignore,device='cpu')
        return dict(details,complete_frames=p['bold_shape'][3])
    if args.function == 'mask':
        from fnit.fmri.mask import epi_brain_mask
        reference = p['reference']
        if reference is None:
            raise ValueError('A verified complete public reference is required for mask fixture')
        result=epi_brain_mask(nib.load(reference))
        nib.save(result,out/'mask.nii.gz')
        return {'mask_voxels':int(np.count_nonzero(np.asarray(result.dataobj))),'scope':'Declared FNIT Otsu helper; no independent official Otsu CLI claimed.'}
    if args.function == 'sampling_reference':
        from fnit.fmri.sampling_reference import native_bold_sampling_reference
        native_bold_sampling_reference(config['sampling_t1'],config['sampling_moving'],config['sampling_t1_mask'],out/'sampling_reference.nii.gz')
        return {'scope':'Complete fixed T1 and mask at native real BOLD voxel sizes.'}
    if args.function == 'legacy_motion_warp':
        from fnit.fmri.spatial import apply_motion_warp
        c=config['legacy490'];raw=nib.load(c['bold']);target=nib.load(c['reference'])
        matrices=np.stack([np.loadtxt(Path(c['official_matrices'])/f'MAT_{t:04d}') for t in range(raw.shape[3])])
        result=apply_motion_warp(raw,target,matrices,device='cpu',interpolation=args.interpolation,
                                 batch_size=8,mcflirt=False)
        nib.save(result,out/'legacy_motion.nii.gz')
        return {'complete_frames':raw.shape[3],'mcflirt_constant_boundary':False,
                'scope':'Legacy FSL-scaled-mm complete motion-only helper; differs from MCFLIRT Constant/extraslice protocol.'}
    raise ValueError('Unsupported FNIT helper')


def original(command, out, environment=None):
    # 独立参考工具不依赖生产源码快照包含 validation/。
    sys.path.insert(0,os.environ['FNIT_BENCHMARK_REFERENCE_TOOLS'])
    from native_exec import run_traced
    with (out/'native.private.log').open('wb') as log:
        process,evidence=run_traced(command,trace_path=out/'native_exec.private.trace',
                                   env=environment,stdout=log,stderr=subprocess.STDOUT)
    if not evidence['original_process_accepted']:
        raise RuntimeError('Native process has no successful actual-child exit evidence')
    return {'native_exit_code':process.returncode,'native_exit_accepted':True}


def official_call(args, config, out, repeat):
    import numpy as np
    p=config['public180']
    os.environ['FNIT_BENCHMARK_REFERENCE_TOOLS']=config['reference_tools_root']
    if args.function == 'stc':
        meta=p['metadata'];timings=np.asarray(meta['SliceTiming'],dtype=float)
        direction=meta.get('SliceEncodingDirection','k')
        if direction not in ('k','k-'):
            raise ValueError('This official Tshift adapter requires the real k-axis fixture')
        if direction.endswith('-'):timings=timings[::-1]
        np.savetxt(out/'timings.1D',timings[None],fmt='%.15g')
        target=float(np.round(timings.min()+args.slice_time_reference*np.ptp(timings),3))
        command=[config['singularity'],'exec','--cleanenv','--bind','/cwStorage,/home1,/public','--env',f'OMP_NUM_THREADS={args.threads},MKL_NUM_THREADS={args.threads},OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS={args.threads}',config['fmriprep_image'],config['afni_tshift'],
                 '-Fourier','-TR',str(meta['RepetitionTime'])+'s','-tzero',str(target),'-ignore',str(args.ignore),
                 '-tpattern','@'+str(out/'timings.1D'),'-prefix',str(out/'stc.nii.gz'),p['bold']]
        return dict(original(command,out),complete_frames=p['bold_shape'][3],StartTime=target)
    if args.function == 'sampling_reference':
        script=Path(config['reference_tools_root'])/'e2e_latest/make_original_sampling_reference.py'
        command=[config['singularity'],'exec','--cleanenv','--bind','/cwStorage,/home1,/public','--env',f'OMP_NUM_THREADS={args.threads},MKL_NUM_THREADS={args.threads},OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS={args.threads}',config['fmriprep_image'],'/app/.pixi/envs/fmriprep/bin/python',str(script),
            '--fixed',config['sampling_t1'],'--moving',config['sampling_moving'],'--mask',config['sampling_t1_mask'],
            '--work',str(out/'sampling_work'),'--output',str(out/'sampling_reference.nii.gz'),'--report',str(out/'sampling_reference.safe.json')]
        with (out/'native.private.log').open('wb') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Original NiWorkflows sampling reference failed')
        return {'native_exit_code':result.returncode,'scope':'Original NiWorkflows grid generation; identical full T1/WM mask and complete real BOLD geometry.'}
    if args.function == 'volume':
        cache=Path(config['official_templateflow_cache'])
        work=args.output/'work';work.mkdir(exist_ok=True)
        home=args.output/'runtime_home';home.mkdir(exist_ok=True)
        command=[config['singularity'],'exec','--cleanenv','--bind',f'/cwStorage,/home1,/public,{cache}:/reference-templateflow','--home',f'{home}:/home/reference','--env',f'TEMPLATEFLOW_HOME=/reference-templateflow,OMP_NUM_THREADS={args.threads},MKL_NUM_THREADS={args.threads},OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS={args.threads},ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={args.threads}',config['fmriprep_image'],
            '/app/.pixi/envs/fmriprep/bin/fmriprep',p['pipeline']['bids_root'],str(args.output/'derivatives'),
            'participant','--participant-label',p['pipeline']['subject'],'--fs-no-reconall',
            '--output-spaces','T1w','MNI152NLin6Asym:res-2','--nthreads',str(args.threads),
            '--omp-nthreads',str(args.threads),'--work-dir',str(args.output/'work'),
            '--skip-bids-validation','--notrack','--no-msm','--random-seed','0','--dummy-scans','0','--slice-time-ref',str(args.slice_time_reference),'--resource-monitor','--stop-on-first-crash','--ignore','fieldmaps']
        if not args.pipeline_stc:command+=['slicetiming']
        return dict(original(command,out),complete_frames=p['bold_shape'][3],
                    scope='Independent official raw-BIDS volume-only preproc; no AROMA/highpass/scaling/clean, no recon-all/surface. Output scope differs from FNIT complete API.')
    if args.function == 'temporal':
        c=config['temporal490'];fsl=Path(config['fsl_root'])
        env=dict(os.environ,FSLDIR=str(fsl),FSLOUTPUTTYPE='NIFTI_GZ')
        env['PATH']=str(fsl/'bin')+':'+env.get('PATH','')
        env['LD_LIBRARY_PATH']=str(fsl/'lib')+':'+env.get('LD_LIBRARY_PATH','')
        # 保留FSL精确p50；stdout是私有采样值，不出公共报告。
        sys.path.insert(0,config['reference_tools_root'])
        from native_exec import run_traced
        stats,stats_evidence=run_traced([str(fsl/'bin/fslstats'),c['official_corrected'],'-k',c['brain_mask'],'-p','50'],trace_path=out/'percentile_exec.private.trace',env=env,capture_output=True,text=True)
        if not stats_evidence['original_process_accepted']:
            raise RuntimeError('Official percentile has no successful actual-child exit evidence')
        tokens=stats.stdout.split()
        if not tokens:raise RuntimeError('Official fslstats produced no percentile')
        median=float(tokens[-1])
        if not np.isfinite(median) or median<=0:raise ValueError('Invalid official percentile')
        factor=args.target_median/median
        scaled=out/'scaled.nii.gz';mean=out/'mean.nii.gz'
        result=original([str(fsl/'bin/fslmaths'),c['official_corrected'],'-mul',str(factor),str(scaled)],out,env)
        sigma=args.cutoff_seconds/(2*c['tr'])
        if not args.remove_mean:
            mean_dir=out/'mean_command';mean_dir.mkdir()
            original([str(fsl/'bin/fslmaths'),str(scaled),'-Tmean',str(mean)],mean_dir,env)
        filter_dir=out/'filter_command';filter_dir.mkdir()
        command=[str(fsl/'bin/fslmaths'),str(scaled),'-bptf',str(sigma),'-1']
        if not args.remove_mean:command+=['-add',str(mean)]
        command +=[str(out/'highpass.nii.gz')]
        original(command,filter_dir,env)
        return dict(result,complete_frames=490,intensity_factor=factor,preserve_mean=not args.remove_mean)
    raise ValueError('No independent official adapter for this helper; use a declared frozen FNIT/reference gate')


if __name__=='__main__':
    main()
