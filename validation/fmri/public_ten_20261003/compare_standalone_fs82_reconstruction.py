"""额外同例原生 FreeSurfer 8.2 与正式 FNIT 重建的只读比较。

不冒用 fMRIPrep 的 ITK 变换：独立单 T1 冷 recon-all 的 raw→orig/001
网格与保存命令证明 scanner frame，强度变化另记。标签为 raw 网格 NN，
曲面为 own orig.affine @ inv(vox2ras_tkr)，不拟合配准，不判等价。
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REVISION = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
CORE_SHA = 'b5a0bdc7a60a9e5671cfe0f0938d61c7e72031c6ee9054815e5e4a15a6441e07'
COLD_RUNNER_SHA = 'a19408d5b928227aa458b2d9480772c1b3820836508046461fbe3f9b0c49fc54'
LATE_VALIDATOR_SHA = '8ba881d05112cc952d120f123e7cca8f48ea9a43db2fd8478e7f6dbcfe8080e8'
OWNED_OUTPUT = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def reject_overlap(output, protected):
    output = output.expanduser().resolve()
    for path in protected:
        path = Path(path).expanduser().resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('new output overlaps a protected input or source')
    if output.exists() or output.is_symlink():
        raise FileExistsError('independent output already exists')
    return output


def read_ready(args):
    candidate_report = args.candidate_case/'report/report.public.json'
    cold_report = args.standalone_case/'report/report.public.json'
    if not candidate_report.is_file() or not cold_report.is_file():
        return None
    candidate, original_cold = load(candidate_report), load(cold_report)
    cold = original_cold
    late_report = getattr(args, 'late_report', None)
    if late_report is not None:
        required = (late_report, args.late_files, args.late_source_manifest, args.late_validator)
        if not all(path.is_file() for path in required):
            return None
        cold = validate_late(args, original_cold, cold_report)
        cold_report = late_report
    for report in (candidate, cold):
        if report.get('status') in ('failed', 'interrupted', 'invalid_provenance'):
            raise ValueError('an original whole run failed; no successful comparison inferred')
    accepted_cold_status = 'complete_saved_outputs_verified_late' if late_report is not None else 'complete'
    if candidate.get('status') != 'complete' or cold.get('status') != accepted_cold_status:
        return None
    if candidate.get('source_revision') != REVISION or cold.get('source_revision') != REVISION:
        raise ValueError('completed run does not use frozen formal1128')
    if (candidate.get('source_unchanged_during_run') is not True
            or candidate.get('raw_inputs_unchanged') is not True
            or candidate.get('provenance_guards_passed') is not True
            or candidate.get('input_sha256') != candidate.get('input_sha256_after')
            or candidate.get('frames') != 180
            or candidate.get('subject') != args.case_id or cold.get('subject') != args.case_id):
        raise ValueError('formal original source/input guards failed')
    if (cold.get('backend') != 'freesurfer' or cold.get('volume_executed') is not False
            or cold.get('reconstruction_reused') is not False
            or (original_cold if late_report is not None else cold).get('cold_reconstruction_preexisting') is not False
            or cold.get('actual_reconstruction_device') != 'cpu'
            or cold.get('frame_count') != 180
            or any(cold.get(key) is not True for key in
                   ('readonly_input_guards_equal','frozen_source_guards_equal','binding_guard_equal'))):
        raise ValueError('standalone cold complete run contract failed')
    if candidate.get('input_sha256',{}).get('t1w') != cold.get('raw_t1w_sha256'):
        raise ValueError('different original raw T1 inputs')
    if (candidate.get('input_sha256',{}).get('bold') != cold.get('raw_bold_sha256')
            or candidate.get('repetition_time') != cold.get('tr_seconds')):
        raise ValueError('different whole-run raw BOLD SHA or measured TR')
    return candidate_report, cold_report, candidate, cold


def validate_late(args, original, original_path):
    """Accept only the fixed reporter-only failure with independent saved QC.

    This never promotes the original failed driver or invents its missing API
    clock. The reference is the exact same saved cold subject, additionally
    validated after the serialization failure.
    """
    late = load(args.late_report)
    failure = late.get('original_driver_failure', {})
    if (late.get('status') != 'complete_saved_outputs_verified_late'
            or late.get('original_driver_status') != 'failed'
            or original.get('status') != 'failed' or original.get('error_type') != 'TypeError'
            or original.get('runner_sha256') != COLD_RUNNER_SHA
            or failure.get('error_type') != 'TypeError'
            or failure.get('message') != 'Object of type PosixPath is not JSON serializable'
            or failure.get('phase') != 'save report/files.private.json after full API return and saved-output checks'
            or failure.get('original_report_sha256') != sha(original_path)
            or failure.get('original_runner_sha256') != COLD_RUNNER_SHA):
        raise ValueError('late branch does not match the exact preserved reporter-only failure')
    if any(original.get(key) is not True for key in
           ('readonly_input_guards_equal','frozen_source_guards_equal','binding_guard_equal')):
        raise ValueError('original cold source/raw/native/configuration guard failed')
    if any(late.get(key) is not True for key in
           ('original_source_guard_passed','late_input_guards_equal','late_source_guards_equal')):
        raise ValueError('late original/source/input guards failed')
    if (not late.get('late_input_sha256_before')
            or late['late_input_sha256_before'] != late.get('late_input_sha256_after')):
        raise ValueError('late saved-input SHA guards changed or are absent')
    timing = late.get('saved_timing_seconds', {})
    total_before_publication = timing.get('total_before_publication')
    if (any(key not in late for key in ('full_api_seconds','returned_api_total_seconds'))
            or late['full_api_seconds'] is not None or late['returned_api_total_seconds'] is not None
            or isinstance(total_before_publication, bool)
            or not isinstance(total_before_publication, (int, float))
            or not math.isfinite(total_before_publication) or total_before_publication < 0
            or late.get('original_driver_failure_wall_seconds') != original.get('driver_through_saved_output_validation_seconds')
            or 'total' in late.get('saved_timing_seconds', {})):
        raise ValueError('late branch invents a missing whole API clock or changes the failed outer clock')
    failure_log = original_path.parent/'failure.private.txt'
    if (sha(failure_log) != failure.get('failure_log_sha256')
            or failure_log.read_text().splitlines()[-1] != 'TypeError: Object of type PosixPath is not JSON serializable'):
        raise ValueError('original private failure log differs from the identified exception')
    binding = load(args.standalone_case/'binding.private.json')
    if (sha(Path(binding['paths']['runner'])) != COLD_RUNNER_SHA
            or sha(args.late_validator) != LATE_VALIDATOR_SHA
            or late.get('late_validator_sha256') != LATE_VALIDATOR_SHA
            or sha(args.late_source_manifest) != late.get('source_manifest_sha256')
            or load(args.late_source_manifest) != binding['source_hashes_before']
            or late.get('actual_source_hashes_before') != binding['source_hashes_before']
            or late.get('actual_source_hashes_after') != binding['source_hashes_before']
            or sha(args.late_files) != late.get('late_private_filemap_sha256')):
        raise ValueError('late validator/source/typed file map is not bound to the actual original run')
    files = load(args.late_files)
    if (files.get('late_generated_filemap') is not True
            or files.get('original_private_filemap_saved') is not False
            or files.get('original_failed_report_sha256') != sha(original_path)
            or files.get('original_binding_sha256') != sha(args.standalone_case/'binding.private.json')):
        raise ValueError('late file map provenance differs from the preserved original run')
    if (files.get('configuration') != load(args.standalone_case/'config.private.json')
            or sha(args.standalone_case/'config.private.json') != late.get('configuration_sha256')
            or sha(Path(files['result']['metadata'])) != late.get('metadata_sha256')
            or sha(Path(files['result']['qc_report'])) != late.get('qc_sha256')
            or sha(Path(files['result']['recon_all']).parent/'fnit-surface-reconstruction.json') != late.get('reconstruction_manifest_sha256')):
        raise ValueError('late saved metadata/QC/reconstruction/configuration differs')
    for key in ('dtseries', 'left', 'right'):
        check = {'left':'L', 'right':'R'}.get(key, key)
        if sha(Path(files['result'][key])) != late['outputs'][check]['sha256']:
            raise ValueError('late full saved time-series bytes differ from strict QC')
    return late


def original_frame(subject, raw_path, metadata):
    """Same scanner frame requires saved single-source command and raw grid.

    Equal pixels are measured, not assumed. A bias-corrected image may retain
    this grid/frame; it is never described as raw-identical in that branch.
    """
    import nibabel as nib
    import numpy as np
    raw = nib.as_closest_canonical(nib.load(str(raw_path)))
    original = nib.as_closest_canonical(nib.load(str(subject/'mri/orig/001.mgz')))
    if raw.ndim != 3 or original.shape != raw.shape or not np.allclose(raw.affine,original.affine,rtol=0,atol=1e-4):
        raise ValueError('standalone single-input original and raw scanner grids disagree; no identity frame inferred')
    records = [record for record in metadata.get('commands',[]) if Path(record['argv'][0]).name == 'recon-all']
    if len(records) != 1:
        raise ValueError('standalone exactly one saved cold recon-all invocation required')
    argv = records[0]['argv']
    if (argv.count('-i') != 1 or '-all' not in argv
            or Path(argv[argv.index('-i')+1]).resolve() != raw_path.resolve()
            or metadata['request']['source_sha256'] != sha(raw_path)):
        raise ValueError('saved cold command is not single same original raw T1')
    a,b = (np.asarray(image.dataobj,dtype=np.float64) for image in (raw,original))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('raw/original pixels are nonfinite')
    same = bool(np.allclose(a,b,rtol=1e-5,atol=1e-3))
    exact = bool(np.array_equal(a,b))
    return {'status':'same_scanner_frame_provenance_verified',
            'canonical_grid_shape':[int(value) for value in raw.shape],
            'canonical_affine_max_abs':float(np.abs(raw.affine-original.affine).max()),
            'pixel_allclose':same,'pixel_allclose_rtol':1e-5,'pixel_allclose_atol':1e-3,
            'pixel_array_equal':exact,'pixel_max_abs':float(np.abs(a-b).max()),
            'orig001_is_raw_identical':exact,
            'raw_sha256':sha(raw_path),'orig001_sha256':sha(subject/'mri/orig/001.mgz'),
            'single_source_recon_all_program_sha256':records[0]['sha256'],
            'saved_command_contract':'Exactly one cold recon-all -all -i this SHA-bound raw T1; matching original scanner grid. No additional workflow affine or fitted registration.',
            'intensity_scope':'exactly equal raw pixel values' if exact else 'tolerance-matching raw pixels; exact pixel identity is not claimed' if same else 'scanner grid matches; original intensities differ, so pixel identity is not claimed'}


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-case','standalone-case','source','helpers','output-root','lut'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=60)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--quality-timeout',type=float,default=180)
    parser.add_argument('--pair-budget',type=int,default=20_000_000)
    for name in ('late-report','late-files','late-source-manifest','late-validator'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    late_paths = (args.late_report,args.late_files,args.late_source_manifest,args.late_validator)
    if any(path is not None for path in late_paths) and not all(path is not None for path in late_paths):
        parser.error('the four explicit --late-* paths must be provided together')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA before starting this CPU-only posthoc')
    if args.threads < 1 or args.threads > 4 or not 1 <= args.poll_seconds <= 60:
        raise ValueError('CPU threads must be 1–4 and poll wait at most60s')
    if args.pair_budget < 1 or args.quality_timeout <= 0:
        raise ValueError('quality candidate/time budgets must be positive')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]=str(args.threads)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    output = reject_overlap(args.output_root,[args.candidate_case,args.standalone_case,args.source,args.helpers,args.lut.parent])
    if args.late_report is not None:
        output = reject_overlap(output,[path.parent for path in late_paths])
    while True:
        ready = read_ready(args)
        if ready is not None:
            break
        if not args.wait:
            raise ValueError('both original whole runs must be complete')
        time.sleep(args.poll_seconds)
    candidate_report,cold_report,candidate,cold = ready
    candidate_files_path = candidate_report.parent/'files.private.json'
    cold_files_path = args.late_files if args.late_report is not None else cold_report.parent/'files.private.json'
    candidate_files,cold_files = load(candidate_files_path),load(cold_files_path)
    subjects = {'reference':Path(cold_files['result']['recon_all']),
                'candidate':Path(candidate_files['recon_all'])}
    adapter_path = subjects['reference'].parent/'fnit-surface-reconstruction.json'
    adapter = load(adapter_path)
    raw = Path(adapter['request']['source_t1w'])
    raw_sha = sha(raw)
    if (adapter.get('status') != 'complete' or adapter['request']['backend'] != 'freesurfer'
            or adapter['request']['device'] != 'cpu'
            or adapter['request']['options'].get('fsnative_to_t1w') is not None
            or adapter['request']['source_sha256'] != raw_sha
            or adapter.get('reused',False) or adapter['subject_dir'] != str(subjects['reference'])
            or cold['raw_t1w_sha256'] != raw_sha):
        raise ValueError('saved standalone subject/request/raw contract failed')
    output = reject_overlap(output,[raw.parent,*subjects.values()])
    source_manifest_path = candidate_report.parent/'source.private.json'
    source_manifest = load(source_manifest_path)
    if (sha(source_manifest_path) != candidate.get('source_sha256') or not source_manifest
            or any(sha(args.source/name) != value for name,value in source_manifest.items())):
        raise ValueError('actual frozen candidate source differs from its whole-run source manifest')
    binding_path = args.standalone_case/'binding.private.json'
    binding = load(binding_path)
    raw_dataset_root=Path(binding['paths']['public_data_manifest']).parent.resolve()
    if not raw.resolve().is_relative_to(raw_dataset_root):
        raise ValueError('raw T1 is outside its actually bound public BIDS dataset root')
    output=reject_overlap(output,[raw_dataset_root])
    if (binding['hashes_before'] != cold.get('readonly_inputs_before')
            or binding['hashes_before'] != cold.get('readonly_inputs_after')
            or binding['source_hashes_before'] != source_manifest
            or any(sha(Path(binding['paths'][name])) != value for name,value in binding['hashes_before'].items())
            or any(sha(args.source/name) != value for name,value in binding['source_hashes_before'].items())):
        raise ValueError('standalone original binding/native/raw/source guards no longer match saved complete run')
    candidate_adapter_path=subjects['candidate'].parent/'fnit-surface-reconstruction.json'
    candidate_reconstruction_path=candidate_report.parent/'reconstruction.private.json'
    candidate_metadata_path=Path(candidate_files['metadata'])
    candidate_adapter=load(candidate_adapter_path)
    if (candidate_adapter != load(candidate_reconstruction_path)
            or candidate_adapter != load(candidate_metadata_path)['FNIT']['Reconstruction']
            or candidate_adapter.get('status') != 'complete'
            or candidate_adapter['request']['backend'] != 'fnit'
            or candidate_adapter['request']['source_sha256'] != raw_sha
            or candidate_adapter['subject_dir'] != str(subjects['candidate'])):
        raise ValueError('candidate subject does not match saved formal reconstruction provenance')
    core_path = args.helpers/'compare_reconstruction.py'
    if sha(core_path) != CORE_SHA:
        raise ValueError('frozen reconstruction metric helper changed')
    for role,manifest in [('reference',adapter),('candidate',candidate_adapter)]:
        for name,value in manifest['files'].items():
            path=(subjects[role]/name).resolve(strict=True)
            if (not path.is_relative_to(subjects[role].resolve()) or not path.is_file()
                    or path.stat().st_size != value['size'] or sha(path) != value['sha256']):
                raise ValueError(role+' saved closure changed')
    effective = adapter['request']['effective']
    for name in ('command','mris_expand'):
        if sha(effective[name]) != effective[name+'_sha256']:
            raise ValueError('standalone original program bytes changed')
    paths = {'candidate_report':candidate_report,'candidate_files':candidate_files_path,
             'candidate_source_manifest':source_manifest_path,'standalone_report':cold_report,
             'candidate_adapter':candidate_adapter_path,'candidate_reconstruction':candidate_reconstruction_path,
             'candidate_surface_metadata':candidate_metadata_path,
             'standalone_files':cold_files_path,'standalone_adapter':adapter_path,
             'raw_t1w':raw,'lut':args.lut,'standalone_binding':binding_path}
    paths.update({'standalone_original_guard/'+name:Path(path) for name,path in binding['paths'].items()})
    if args.late_report is not None:
        paths.update({'standalone_original_failed_report':args.standalone_case/'report/report.public.json',
                      'standalone_original_failure_log':args.standalone_case/'report/failure.private.txt',
                      'late_source_manifest':args.late_source_manifest,'late_validator':args.late_validator})
    for role,subject in subjects.items():
        for directory in ('mri','surf','label','stats','scripts'):
            for path in sorted((subject/directory).rglob('*')):
                if path.is_file():
                    paths[f'{role}/{path.relative_to(subject)}'] = path
    paths.update({'original_program/'+name:Path(effective[name]) for name in ('command','mris_expand')})
    build_stamp = Path(effective['command']).parent.parent/'build-stamp.txt'
    if build_stamp.is_file():
        paths['original_build_stamp'] = build_stamp
    modules = {'wrapper':Path(__file__),**{'metric_helper/'+path.name:path for path in args.helpers.iterdir() if path.is_file()}}
    modules.update({'fnit_source/'+name:args.source/name for name in source_manifest})
    before = {name:sha(path) for name,path in paths.items()}
    code_before = {name:sha(path) for name,path in modules.items()}
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('standalone_fs82_metric_core',core_path)
    core = importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    import numpy as np,nibabel.freesurfer.io as fs,socket
    frame_proof = original_frame(subjects['reference'],raw,adapter)
    candidate_proof = core.candidate_raw_identity(SimpleNamespace(raw_t1w=raw,candidate=subjects['candidate']),candidate)
    matrices,frames = {},{}
    for role,subject in subjects.items():
        matrices[role],frames[role] = core.frame(subject)
    # All previous steps are read-only; reject before this first output write.
    output.mkdir(parents=True,exist_ok=False)
    OWNED_OUTPUT = output
    distance,stats,quality = core.configure(args.source,args.helpers,args.threads,output/'numba_cache')
    import torch
    report_path = output/'report.public.json'
    report = {'case_id':args.case_id,'status':'running','equivalence':'not_assessed',
              'scope':'Independent standalone original FreeSurfer8.2 vs formal FNIT reconstruction posthoc; excludes both MRI/surface clocks and never replaces fMRIPrep7.3 ten-case reference.',
              'created_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
              'threads':args.threads,'actual_torch_threads':torch.get_num_threads(),'device':'cpu',
              'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),
              'host_load_average_at_posthoc_start':list(os.getloadavg()),'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'reference_software':{'FreeSurfer_build':build_stamp.read_text().strip() if build_stamp.is_file() else 'not_assessed'},
              'candidate_source_revision':REVISION,'input_sha256':before,'code_sha256':code_before,
              'coordinate_frame':{'space':'original raw T1 scanner RAS','reference':frames['reference'],'candidate':frames['candidate'],
                  'standalone_single_T1_frame_proof':frame_proof,'candidate_raw_identity_proof':candidate_proof,
                  'method':'Each own orig.affine @ inv(vox2ras_tkr), no workflow affine is present in the standalone single-T1 invocation; no new fitted registration.'},
              'quality_budget':{'seconds_per_worker':args.quality_timeout,'maximum_pair_budget':args.pair_budget,'incomplete_is_not_pass':True},'surfaces':{}}
    if args.late_report is not None:
        report['standalone_saved_output_validation_boundary'] = {
            'status':'independently verified saved outputs after preserved reporter-only failure',
            'original_driver_status':'failed','original_driver_report_sha256':before['standalone_original_failed_report'],
            'late_report_sha256':before['standalone_report'],'late_private_filemap_sha256':before['standalone_files'],
            'original_driver_failure_wall_seconds':cold['original_driver_failure_wall_seconds'],
            'original_full_api_seconds':None,'returned_api_total_seconds':None,
            'saved_timing_seconds':cold['saved_timing_seconds'],
            'scope':'Late QC is separate from original failed outer clock and from this extra posthoc; no complete original driver or recovered continuous API clock claimed.'}
    core.save(report_path,report)
    for hemi in ('lh','rh'):
        report['surfaces'][hemi] = {}
        for name in ('white','pial'):
            tick=time.perf_counter();rv,rf=fs.read_geometry(str(subjects['reference']/'surf'/f'{hemi}.{name}'));cv,cf=fs.read_geometry(str(subjects['candidate']/'surf'/f'{hemi}.{name}'))
            rv=core.map_vertices(rv,matrices['reference']);cv=core.map_vertices(cv,matrices['candidate'])
            report['surfaces'][hemi][name]={'reference_vertices':len(rv),'candidate_vertices':len(cv),'reference_faces':len(rf),'candidate_faces':len(cf),
                'reference_to_candidate_triangle':distance._summary(distance._point_to_mesh(rv,cv,cf)),
                'candidate_to_reference_triangle':distance._summary(distance._point_to_mesh(cv,rv,rf)),
                'method':'Exact vertex-sampled bidirectional point-to-full-triangle distance; not continuous Hausdorff; no fitted alignment.',
                'seconds':time.perf_counter()-tick}
            core.save(report_path,report)
    identity=np.eye(4)
    report['label_dice']=core.label_metrics(subjects['reference'],subjects['candidate'],raw,args.lut,identity,identity)
    report['roi_stats']=core.roi_metrics(subjects['reference'],subjects['candidate'],stats)
    for row in report['roi_stats'].values():
        if row.get('status')=='measured':
            row['reason']='Standalone original FS8.2 and FNIT preserve original native stats definitions; zero references have no relative division. Numerical agreement is measured, equivalence is not assessed.'
    core.save(report_path,report)
    report['quality']={}
    for role,subject in subjects.items():
        # Separate role paths prevent equal subject-parent names from overwriting diagnostics.
        quality_args=SimpleNamespace(source=args.source,helpers=args.helpers,threads=args.threads,
            quality_timeout=args.quality_timeout,pair_budget=args.pair_budget,
            case_id=args.case_id+'-'+role,output=output/role/'report.public.json')
        report['quality'][role]=core.surface_quality(subject,quality,quality_args)
        core.save(report_path,report)
    after={name:sha(path) for name,path in paths.items()};code_after={name:sha(path) for name,path in modules.items()}
    report.update(inputs_unchanged=before==after,code_unchanged=code_before==code_after,input_sha256_after=after,code_sha256_after=code_after,
        cuda_initialized=torch.cuda.is_initialized(),posthoc_wall_seconds=time.perf_counter()-started,
        timing_scope='Independent extra CPU-only saved-reconstruction comparison; excluded from both pipeline timing and the formal ten-case reference statistics.')
    report['status']='measured' if report['inputs_unchanged'] and report['code_unchanged'] else 'failed_input_or_code_changed'
    if report['status']=='measured' and any(row.get('status')!='measured' for chain in report['quality'].values() for row in chain.values()):
        report['status']='partially_measured'
    core.save(output/'files.private.json',{name:str(path) for name,path in paths.items()})
    core.save(report_path,report)
    if report['status'].startswith('failed'):
        raise RuntimeError('input or source changed during posthoc')
    print(json.dumps({'case_id':args.case_id,'status':report['status'],'report_sha256':sha(report_path),'extra_seconds':report['posthoc_wall_seconds']}))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        # Never write a rejected/pre-existing directory. Preserve a failure only
        # after this invocation successfully created its isolated output.
        if OWNED_OUTPUT is not None:
            path=OWNED_OUTPUT/'report.public.json'
            report=load(path) if path.is_file() else {}
            report.update(status='failed_posthoc_exception',failure_type=type(error).__name__,
                          failure_details='preserved private traceback; no complete or quality pass inferred')
            (OWNED_OUTPUT/'failure.private.txt').write_text(traceback.format_exc())
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise
