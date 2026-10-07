"""Read eight saved first/warm jobs and compare complete outputs; zero new APIs."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from volume_numeric import sha, compare_arrays, compare_nifti

ROLES = ('clean_native', 'clean_mni', 'preproc_t1w', 'preproc_mni', 'mask_mni',
         't1_brain', 'bbr_matrix', 'motion_pull', 'mni_pull', 'bold_reference', 'metadata')
CONFIG_KEYS = ('registration_backend', 'fnirt_config', 'ica_n_components', 'ica_max_iter',
               'aroma_mode', 'regress_wm', 'regress_csf', 'regress_motion', 'motion_model',
               'bandpass', 'global_signal', 'highpass_cutoff_seconds', 'slice_timing',
               'slice_time_reference', 'device', 'batch_size', 'motion_iterations',
               'bold_reference_strategy', 'motion_algorithm', 'motion_output',
               'n_splits', 'random_state', 'brain_extraction', 'mni_template',
               'mni_brain_mask', 'fast_config', 'reuse_anatomical', 'bbr_execution',
               'fnirt_execution', 'mni_interpolation', 'preproc_interpolation',
               'preproc_coordinate_precision', 'matmul_allow_tf32', 'cudnn_allow_tf32', 'weights')


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError('Pinned saved evidence changed')
    return json.loads(Path(path).read_text())


def verify_source(source):
    root = Path(source['root'])
    actual = {p.relative_to(root).as_posix(): sha(p) for p in (root/'src').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != source['files_sha256'] or digest != source['tree_sha256']:
        raise ValueError('Complete scientific source closure differs')


def load_binding(path, digest):
    binding = read(path, digest)
    if socket.gethostname().split('.')[0] != binding['node']:
        raise ValueError('Collect on the CPU host with the shared physical-core evidence')
    if (len(binding['jobs']) != 8 or binding['complete_frames'] != 180
            or binding['default_slice_timing'] is not False
            or set(binding['cpu_pairs']) != {'fnirt_cpu1', 'fnirt_cpu8', 'synthmorph_cpu1', 'synthmorph_cpu8'}):
        raise ValueError('Four complete first/warm source pairs required')
    for row in binding['cpu_topology']:
        topology=Path(f"/sys/devices/system/cpu/cpu{row['logical_cpu']}/topology")
        if (int((topology/'physical_package_id').read_text())!=row['socket']
                or int((topology/'core_id').read_text())!=row['core']):
            raise ValueError('Actual physical CPU topology differs from final freeze')
    for source in binding['source_bindings'].values():
        verify_source(source)
    for path, info in binding['input_metadata'].items():
        if Path(path).stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Complete shared raw/template/model input changed')
    for path_key, sha_key in (('input_path', 'input_json_sha256'),
                              ('benchmark_path', 'benchmark_sha256'),
                              ('cpu_worker_path', 'cpu_worker_sha256'),
                              ('cpu_controller_path', 'cpu_controller_sha256')):
        read_path = Path(binding[path_key])
        if sha(read_path) != binding[sha_key]:
            raise ValueError('Frozen adapter/worker/input configuration changed')
    return binding


def status(binding):
    rows = []
    for name in sorted(binding['cpu_pairs']):
        path = Path(binding['run_root'])/('queue_'+name)/'controller_status.private.json'
        value = read(path) if path.is_file() else {}
        rows.append({'queue': name, 'status': value.get('status', 'not_started'),
                     'completed_API_calls': value.get('complete_API_calls', 0),
                     'jobs': [{'source_kind': x['source_kind'], 'status': x['status']}
                              for x in value.get('jobs', [])]})
    return {'queues': rows, 'complete_API_target': 16,
            'complete_API_calls': sum(x['completed_API_calls'] for x in rows),
            'status': 'execution_complete_numeric_not_collected' if all(x['status'] == 'complete' for x in rows)
                      else 'execution_pending', 'new_API_calls': 0}


def scientific_configuration(path, repeat, backend):
    metadata = read(path)
    f = metadata['FNIT']; config = f['Configuration']
    if (f['RegistrationBackend'] != backend or config['device'] != 'cpu'
            or config['slice_timing'] is not False or config['bold_reference_strategy'] != 'robust'
            or config['anatomical_cache']['reused'] is not (repeat == 1)
            or f['Denoising']['Completed'] is not True
            or config['ica_max_iter'] != 500 or config['n_splits'] != 1000
            or config['random_state'] != 0 or config['motion_iterations'] != [1, 1, 1]):
        raise ValueError('Actual complete robust/default/STC-OFF first/warm CPU protocol differs')
    canonical = {key: config[key] for key in CONFIG_KEYS}
    canonical['confound_projection'] = config.get('confound_projection', 'orthogonal')
    if canonical['confound_projection'] != 'orthogonal':
        raise ValueError('Preserve existing orthogonal GPU/CPU pipeline settings')
    return {'canonical': canonical, 'sources': metadata['Sources'], 'TR': metadata['RepetitionTime'],
            'anatomical_cache_reused': repeat == 1,
            'historical_metadata_defaults': [] if 'confound_projection' in config else ['confound_projection=orthogonal']}


def image_check(path, role, tr):
    import nibabel as nib
    import numpy as np
    image = nib.load(str(path), keep_file_open=True)
    if role in ROLES[:4]:
        units = image.header.get_xyzt_units()
        if (image.ndim != 4 or image.shape[-1] != 180 or str(image.get_data_dtype()) != 'float32'
                or units != ('mm', 'sec') or not np.isclose(image.header.get_zooms()[3], tr, rtol=0, atol=1e-6)):
            raise ValueError('Complete float32 BOLD physical space/time axis differs')
    elif role == 'official_native_preproc':
        # The independent official output may retain its original stored
        # precision. Its full frame count and physical time axis still need
        # the same explicit check before it is described as validated.
        units = image.header.get_xyzt_units()
        if (image.ndim != 4 or image.shape[-1] != 180
                or units != ('mm', 'sec') or not np.isclose(image.header.get_zooms()[3], tr, rtol=0, atol=1e-6)):
            raise ValueError('Complete official native BOLD physical space/time axis differs')
    finite = True
    chunks = ([(slice(None),)*(image.ndim-1)+(slice(i,i+8),)
               for i in range(0,image.shape[-1],8)] if image.ndim == 4 else [Ellipsis])
    for selection in chunks:
        finite = finite and bool(np.isfinite(np.asanyarray(image.dataobj[selection])).all())
    if not finite:
        raise ValueError('Nonfinite saved scientific output')
    result = {'shape': list(image.shape), 'dtype': str(image.get_data_dtype()), 'all_finite': True,
              'spatial_axis_codes': list(nib.aff2axcodes(image.affine)),
              'time_unit': image.header.get_xyzt_units()[1],
              'binary_header_sha256': hashlib.sha256(image.header.binaryblock).hexdigest()}
    if role == 'official_native_preproc':
        result.update(physical_time_axis_checked=True,
                      TR_seconds=float(image.header.get_zooms()[3]),
                      dtype_policy='Original stored dtype reported; no float32 conversion or requirement')
    return result


def load_job(binding, binding_sha, job, controller_row):
    guard_path = Path(job['output_root'])/'guard.private.json'
    guard = read(guard_path, controller_row['guard_sha256'])
    source = binding['source_bindings'][job['source_kind']]
    if (controller_row.get('status') != 'complete' or controller_row.get('returncode') != 0
            or guard.get('status') != 'complete' or guard['binding_sha256'] != binding_sha
            or guard['worker_sha256'] != binding['cpu_worker_sha256']
            or guard['job_name'] != job['name'] or guard['source_kind'] != job['source_kind']
            or guard['registration_backend'] != job['registration_backend']
            or guard['source_tree_sha256'] != source['tree_sha256'] or guard['git_revision'] != source['git_revision']
            or guard['source_unchanged'] is not True or guard['input_unchanged'] is not True
            or guard['actual_cpu_threads'] != job['threads'] or guard['actual_cpu_affinity'] != job['affinity']
            or guard['complete_frames'] != 180 or guard['first_and_warm_complete_API_calls'] != 2
            or len(guard['calls']) != 2):
        raise ValueError('Actual complete matched-source/budget execution guard differs')
    report = read(guard['mature_report_path'], guard['mature_report_sha256'])
    expected_py = {key: value for key, value in source['files_sha256'].items()
                   if key.startswith('src/fnit/') and key.endswith('.py')}
    if (report['status'] != 'complete' or report['source_unchanged'] is not True
            or report['source_sha256'] != expected_py or report['adapter_sha256'] != binding['benchmark_sha256']
            or report['cpu_threads'] != job['threads'] or report['cpu_affinity'] != job['affinity']
            or report['source_revision'] != source['git_revision'] or len(report['records']) != 2):
        raise ValueError('Recorded actually checked Python sources/API adapter differ')
    inputs = read(binding['input_path']); raw = inputs['public180']
    for label, key in (('raw_BOLD','bold'), ('raw_T1w','t1w')):
        if report['input_sha256'][label] != binding['input_metadata'][raw[key]]:
            raise ValueError('Completed API did not hash the complete pinned raw input')
    calls = []
    for repeat, observation in enumerate(guard['calls']):
        if (observation['repeat'] != repeat or observation['actual_Torch_threads'] != job['threads']
                or observation['actual_Torch_interop_threads'] != 1
                or observation['actual_affinity'] != job['affinity']
                or set(observation['output_metadata']) != set(ROLES)):
            raise ValueError('Saved first/warm complete outputs/budget differ')
        record = report['records'][repeat]
        seconds = guard['API_wall_seconds'][repeat]
        if not math.isfinite(seconds) or seconds <= 0 or seconds != record['api_wall_seconds_including_io']:
            raise ValueError('Actual normal-I/O API clock differs')
        outputs = observation['output_metadata']
        for information in outputs.values():
            if (Path(information['path']).stat().st_size != information['bytes']
                    or sha(information['path']) != information['sha256']):
                raise ValueError('Preserved complete output differs')
        config = scientific_configuration(outputs['metadata']['path'], repeat, job['registration_backend'])
        checks = {role: image_check(info['path'], role, config['TR']) for role,info in outputs.items()
                  if info['path'].endswith(('.nii','.nii.gz'))}
        calls.append({'repeat': repeat, 'outputs': outputs, 'configuration': config, 'checks': checks,
                      'API_seconds': seconds, 'stages': record['stages'],
                      'outside_API_output_preservation_seconds': observation['outside_API_output_preservation_seconds']})
    return {'job': job, 'guard': guard, 'calls': calls, 'controller_row': controller_row,
            'guard_sha256': sha(guard_path), 'report_sha256': guard['mature_report_sha256']}


def compare_calls(first, second, compare_settings=True):
    import numpy as np
    rows = {}
    for role in ROLES:
        if role == 'metadata': continue
        left, right = first['outputs'][role], second['outputs'][role]
        if left['path'].endswith(('.nii','.nii.gz')):
            row = compare_nifti(left['path'],right['path'])
        else:
            load = (lambda path: np.load(path,allow_pickle=False)) if role == 'motion_pull' else np.loadtxt
            row = compare_arrays(load(left['path']),load(right['path']))
            if row.get('shape_equal') is not True or list(load(left['path']).shape) != ([180,4,4] if role == 'motion_pull' else [4,4]):
                raise ValueError('Every affine/frame must be preserved')
        row.update(baseline_file_sha256=left['sha256'],candidate_file_sha256=right['sha256'])
        rows[role] = row
    a,b = first['configuration'],second['configuration']
    equal = (a['canonical'] == b['canonical'] and a['sources'] == b['sources'] and a['TR'] == b['TR'])
    return {'output_comparisons': rows,'scientific_configuration_equal':equal,
            'all_values_dtype_grids_headers_exact': all(x['accepted'] for x in rows.values()),
            'strict_numeric_gate': (equal or not compare_settings) and all(x['accepted'] for x in rows.values()),
            'precision_scope':'Every saved value of ten scientific outputs,180 frames,full grids; no image resampling. Configuration excludes source identity and cache fingerprint, which legitimately differ.'}


def stage_rows(stages):
    # Keep nested events separately; never sum into an API or process clock.
    return [{'stage': row['stage'], 'status': row['status'], 'wall_seconds': row.get('wall_seconds')}
            for row in stages]


def collect(binding, digest):
    loaded = {}; controllers = {}; timings = []
    for queue_name, pair in sorted(binding['cpu_pairs'].items()):
        controller_path = Path(binding['run_root'])/('queue_'+queue_name)/'controller_status.private.json'
        controller = read(controller_path)
        if (controller['status'] != 'complete' or controller['complete_API_calls'] != 4
                or controller['binding_sha256'] != digest or controller['cpu_affinity'] != pair['affinity']):
            raise ValueError('Complete four-call physical-core queue is still pending')
        controllers[queue_name] = {'sha256':sha(controller_path)}
        jobs = [j for j in binding['jobs'] if j['queue_name'] == queue_name]
        if len(jobs) != 2 or {j['source_kind'] for j in jobs} != {'baseline','candidate'}:
            raise ValueError('One baseline/candidate pair per budget required')
        for job in jobs:
            if job['affinity'] != pair['affinity'] or job['threads'] != pair['threads']:
                raise ValueError('Old/new must retain identical physical cores')
            row = next(x for x in controller['jobs'] if x['name'] == job['name'])
            data = load_job(binding,digest,job,row); loaded[(queue_name,job['source_kind'])] = data
            for call in data['calls']:
                timings.append({'queue':queue_name,'source_kind':job['source_kind'],'repeat':call['repeat'],
                                'call_type':'cold_complete_API' if call['repeat']==0 else 'warm_anatomical_cache_API',
                                'API_seconds_including_normal_IO':call['API_seconds'],
                                'outside_API_preservation_seconds':call['outside_API_output_preservation_seconds'],
                                'stage_wall_records':stage_rows(call['stages']),
                                'output_checks':call['checks'], 'guard_sha256':data['guard_sha256'],
                                'execution_report_sha256':data['report_sha256']})
    pairs = []; cache_pairs = []
    for queue in sorted(binding['cpu_pairs']):
        old,new = loaded[(queue,'baseline')],loaded[(queue,'candidate')]
        for repeat in range(2):
            rows = compare_calls(old['calls'][repeat],new['calls'][repeat])
            pairs.append(dict(queue=queue,repeat=repeat,**rows))
        for source_kind, data in (('baseline',old),('candidate',new)):
            rows = compare_calls(data['calls'][0],data['calls'][1],compare_settings=False)
            cache_pairs.append(dict(queue=queue,source_kind=source_kind,**rows))
    private = {'jobs': {queue+'_'+kind: data for (queue,kind),data in loaded.items()}}
    process_rows = [{'queue':queue,'source_kind':kind,
                     'process_wall_seconds':data['controller_row']['process_wall_seconds_including_lease_imports_hashes_IO'],
                     'process_user_seconds':data['controller_row']['process_user_seconds'],
                     'process_system_seconds':data['controller_row']['process_system_seconds'],
                     'maximum_process_RSS_kib':data['guard']['maximum_process_RSS_kib']}
                    for (queue,kind),data in loaded.items()]
    public = {'schema_version':1,'status':'complete_scientific_comparison','complete_API_calls':16,
              'new_API_calls':0,'matched_pairs':8,'cache_preservation_pairs':8,
              'source_revisions':{key:value['git_revision'] for key,value in binding['source_bindings'].items()},
              'source_tree_sha256':{key:value['tree_sha256'] for key,value in binding['source_bindings'].items()},
              'binding_sha256':digest,'controller_evidence':controllers,'timings':timings,'process_timings':process_rows,
              'old_new_comparisons':pairs,'cold_warm_comparisons':cache_pairs,
              'strict_old_new_numeric_gate':all(x['strict_numeric_gate'] for x in pairs),
              'strict_cache_output_gate':all(x['strict_numeric_gate'] for x in cache_pairs),
              'same_old_new_physical_cores':True,'same_raw_input_hashes':True,
              'API_clock_scope':'Complete raw input,normal compute and final saves; imports,source/input hashes,leases and saved-output preservation excluded. First/warm reported separately.',
              'stage_clock_scope':'Existing nested observer host wall clocks; never summed. No new timed pipeline executed.',
              'process_clock_scope':'Frozen controller observed child process wall through saved guard read; includes two APIs, imports, hashes, lease waits and outside-API output copies. It is distinct from one complete API.',
              'privacy':'Only anonymous roles,hashes and scalar metrics; paths,subjects,input arrays and images remain private.'}
    return public,private


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--binding',type=Path,required=True); parser.add_argument('--binding-sha256',required=True)
    parser.add_argument('--status-only',action='store_true'); parser.add_argument('--output',type=Path)
    parser.add_argument('--collector-queue',choices=('fnirt_cpu1','synthmorph_cpu1'),default='fnirt_cpu1')
    args=parser.parse_args(); os.umask(0o077)
    binding=read(args.binding,args.binding_sha256)
    if args.status_only:
        print(json.dumps(status(binding))); return
    if args.output is None: parser.error('--output is required for collection')
    if args.output.exists(): raise FileExistsError('Preserve previous saved comparisons; choose a new output')
    if status(binding)['complete_API_calls'] != 16:
        raise RuntimeError('Wait for all four complete queues before saved-output collection')
    pair=binding['cpu_pairs'][args.collector_queue]
    if pair['threads'] != 1 or len(pair['affinity']) != 1:
        raise ValueError('Collector must reuse one existing authorized CPU1 queue')
    os.sched_setaffinity(0,set(pair['affinity']))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    leases=[]
    for key,mode in (('queue_lock',fcntl.LOCK_EX),('shared_group_lease',fcntl.LOCK_SH)):
        lease=Path(pair[key])
        if not lease.is_file(): raise FileNotFoundError('Reuse an existing completed CPU1 lease')
        handle=lease.open('a+'); fcntl.flock(handle,mode); leases.append(handle)
    binding=load_binding(args.binding,args.binding_sha256)
    args.output.mkdir(parents=True); started=time.perf_counter()
    save(args.output/'status.private.json',{'status':'collecting_saved_complete_outputs','new_API_calls':0})
    public,private=collect(binding,args.binding_sha256)
    load_binding(args.binding,args.binding_sha256)
    public.update(collector_sha256=sha(__file__),numeric_helper_sha256=sha(Path(__file__).with_name('volume_numeric.py')),
                  saved_output_comparison_seconds=time.perf_counter()-started)
    save(args.output/'comparison.public.json',public); save(args.output/'inventory.private.json',private)
    save(args.output/'status.private.json',{'status':'complete','new_API_calls':0,
         'complete_API_calls':16,'strict_old_new_numeric_gate':public['strict_old_new_numeric_gate'],
         'strict_cache_output_gate':public['strict_cache_output_gate'],'report_sha256':sha(args.output/'comparison.public.json')})
    print(json.dumps(read(args.output/'status.private.json')))


if __name__=='__main__': main()
