"""CPU-only byte and parsed-array audit of actual original/recovery artifacts.

No tolerance, matrix recomputation, scalar sampling or tracking. Metadata bytes
are compared and reported too; path/time records remain distinct artifacts.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
ORIGIN=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1/formal_accuracy_raw_v1')
CONFIG=ORIGIN.parent/'formal_frozen_v1/accuracy_configuration.json'
CONFIG_SHA='f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'.cpu.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def equal_array(left,right):
    import numpy as np
    equal_shape=left.shape==right.shape;equal_dtype=left.dtype==right.dtype
    result={'original_shape':list(left.shape),'recovery_shape':list(right.shape),'original_dtype':str(left.dtype),
            'recovery_dtype':str(right.dtype),'shape_equal':equal_shape,'dtype_equal':equal_dtype}
    if not equal_shape or not equal_dtype: result['array_bits_equal']=False; return result
    # Exact stored bits: NaN payload and signed zero differences are retained.
    bits=left.tobytes(order='C')==right.tobytes(order='C'); result['array_bits_equal']=bits
    result['value_equal_including_nan']=bool(np.array_equal(left,right,equal_nan=True)) if left.dtype.kind in 'fc' else bool(np.array_equal(left,right))
    if left.dtype.kind in 'biufc':
        mask=(left!=right)
        if left.dtype.kind in 'fc': mask&=~(np.isnan(left)&np.isnan(right))
        result['unequal_value_count']=int(np.count_nonzero(mask))
        finite=np.isfinite(left)&np.isfinite(right)
        result['max_absolute_error_on_finite']=float(np.max(np.abs(left[finite].astype(np.float64)-right[finite].astype(np.float64)))) if np.any(finite) else None
    return result

def parsed(path,left,right):
    import nibabel as nib
    import numpy as np
    name=path.name
    if name.endswith(('.nii','.nii.gz','.mgz')):
        a,b=nib.load(left),nib.load(right)
        fields={'image':equal_array(np.asanyarray(a.dataobj),np.asanyarray(b.dataobj)),
                'affine':equal_array(np.asarray(a.affine),np.asarray(b.affine))}
        return {'kind':'MRI_array_and_affine','fields':fields,'all_array_bits_equal':all(v['array_bits_equal'] for v in fields.values())}
    if name.endswith('.npz'):
        with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
            keys_equal=set(a.files)==set(b.files)
            fields={key:equal_array(a[key],b[key]) for key in a.files if key in b.files}
            return {'kind':'NPZ_exact_saved_arrays','original_keys':a.files,'recovery_keys':b.files,'keys_equal':keys_equal,
                    'fields':fields,'all_array_bits_equal':keys_equal and all(v['array_bits_equal'] for v in fields.values())}
    if name.endswith('.tck'):
        a,b=nib.streamlines.load(left,lazy_load=False).streamlines,nib.streamlines.load(right,lazy_load=False).streamlines
        # Public saved streamline iteration excludes unused internal capacity.
        # Point order plus every track length uniquely preserves all boundaries.
        points_a=np.concatenate(list(a),axis=0) if len(a) else np.empty((0,3),dtype=np.float32)
        points_b=np.concatenate(list(b),axis=0) if len(b) else np.empty((0,3),dtype=np.float32)
        lengths_a=np.asarray([len(x) for x in a],dtype=np.int64);lengths_b=np.asarray([len(x) for x in b],dtype=np.int64)
        fields={'points':equal_array(points_a,points_b),'lengths':equal_array(lengths_a,lengths_b)}
        return {'kind':'TCK_all_saved_points_and_boundaries','original_tracks':len(a),'recovery_tracks':len(b),'fields':fields,
                'all_array_bits_equal':all(v['array_bits_equal'] for v in fields.values())}
    if name.endswith('.json'):
        a,b=json.loads(Path(left).read_bytes()),json.loads(Path(right).read_bytes())
        return {'kind':'JSON_metadata_exact_structure','parsed_equal':a==b,'scientific_arrays_assessed':False}
    if name.endswith(('.csv','.txt','.bvec','.bval')) or '.eddy_' in name:
        try:
            a=np.loadtxt(left,delimiter=',' if name.endswith('.csv') else None)
            b=np.loadtxt(right,delimiter=',' if name.endswith('.csv') else None)
        except ValueError:
            return {'kind':'text_exact_bytes_only','scientific_arrays_assessed':False}
        values=equal_array(a,b)
        return {'kind':'numeric_text_exact_parsed_array','fields':{'values':values},'all_array_bits_equal':values['array_bits_equal']}
    return {'kind':'other_exact_bytes_only','scientific_arrays_assessed':False}

def verify_exports(analyzer,wall):
    import nibabel as nib
    import numpy as np
    for name in ('geometry.npz','wm_fod_normalized.nii.gz','fa.nii.gz','brain_mask.nii.gz','five_tissue.nii.gz','gmwmi.nii.gz','track_metrics.npz','tracks.tck','dwi_to_t1_world.csv'):
        analyzer.verified_export(wall,name)
    tracks=analyzer.verified_export(wall,'tracks.tck'); scalars=analyzer.verified_export(wall,'track_metrics.npz')
    saved=list(nib.streamlines.load(tracks,lazy_load=False).streamlines)
    endpoints=np.stack([np.stack((x[0],x[-1])) for x in saved]) if saved else np.empty((0,2,3),dtype=np.float32)
    with np.load(scalars,allow_pickle=False) as values:
        require(endpoints.dtype==values['endpoints'].dtype and endpoints.tobytes()==values['endpoints'].tobytes(),'TCK endpoint saved bits differ from returned arrays')
        require(all(len(values[k])==len(saved) for k in ('weights','lengths','mean_fa','endpoints')),'scalar/track count mismatch')
    return {'tracks':len(saved),'TCK_sha256':sha(tracks),'scalar_sha256':sha(scalars),'endpoint_bits_verified':True,'all_scalar_counts_verified':True}

def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU comparison must explicitly hide GPUs')
    require(sha(CONFIG)==CONFIG_SHA,'original frozen config changed')
    config=json.loads(CONFIG.read_bytes()); clone=json.loads((ROOT/'recovery_configuration.json').read_bytes())
    plan=json.loads((ROOT/'frozen_recovery_plan.json').read_bytes()); terminal=json.loads((ROOT/'status.json').read_bytes())
    require(terminal['status']=='GPU_execution_completed_CPU_comparison_pending','selected scientific recovery is not actually complete')
    sys.path.insert(0,str(Path(config['worker_script']).parent))
    driver=importlib.import_module('benchmark_connectome_accuracy_cohort');driver.verify_tools(config)
    analyzer=importlib.import_module('analyze_connectome_accuracy_cohort')
    require(Path(analyzer.__file__).resolve()==Path(config['worker_script']).parent/'analyze_connectome_accuracy_cohort.py','wrong frozen analyzer import')
    cases=driver.validate_plan(config,driver.bound(config['raw_manifest']),driver.bound(config['input_bindings']))
    bindings=driver.bound(config['input_bindings'])
    report={'status':'running','scope':'strict CPU audit; all original outputs retained and recovery used only for memory measurement eligibility',
            'identity':{'host':socket.gethostname(),'python':sys.executable,'numpy_version':importlib.import_module('numpy').__version__,'nibabel_version':importlib.import_module('nibabel').__version__},
            'original_configuration_sha256':CONFIG_SHA,'recovery_configuration_sha256':sha(ROOT/'recovery_configuration.json'),
            'plan_sha256':sha(ROOT/'frozen_recovery_plan.json'),'GPU_terminal_sha256':sha(ROOT/'status.json'),
            'comparison_helper_sha256':sha(__file__),'frozen_analyzer_sha256':sha(analyzer.__file__),'cases':{},'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    for item in plan['explicit_selection']:
        case_id=item['case_id']; anatomy=bindings['cases'][case_id]['anatomy']['directory']
        gpu0,wall0,gpu_path0=analyzer.completed_run(config,cases[case_id],'candidate',anatomy)
        gpu1,wall1,gpu_path1=analyzer.completed_run(clone,cases[case_id],'candidate',anatomy)
        before=json.loads((ROOT/f'{case_id}_original_before.json').read_bytes())
        after=json.loads((ROOT/f'{case_id}_original_after.json').read_bytes()); require(before==after,'original before/after preservation audit failed')
        origin=ORIGIN/'candidate'/case_id; recovery=ROOT/'candidate'/case_id
        files0=before['scientific_output_inventory'];files1={str(p.relative_to(recovery)):p for d in ('connectome','returned_result') for p in (recovery/d).rglob('*') if p.is_file()}
        require(set(files0)==set(files1),'original/recovery output file coverage differs')
        records={}
        for relative,record in files0.items():
            left,right=origin/relative,files1[relative]
            require(sha(left)==record['sha256'],'original output changed since immutable before receipt')
            byte1=sha(right)
            records[relative]={'original_sha256':record['sha256'],'recovery_sha256':byte1,'bytes_equal':record['sha256']==byte1,
                               'original_size_bytes':left.stat().st_size,'recovery_size_bytes':right.stat().st_size,
                               'parsed':parsed(Path(relative),left,right)}
        numeric=[r['parsed']['all_array_bits_equal'] for r in records.values() if 'all_array_bits_equal' in r['parsed']]
        case={'original_GPU_sha256':sha(gpu_path0),'recovery_GPU_sha256':sha(gpu_path1),'original_wall_sha256':sha(origin/'raw_bids_wall.json'),
              'recovery_wall_sha256':sha(recovery/'raw_bids_wall.json'),'original_memory':gpu0['memory_budget'],'recovery_memory':gpu1['memory_budget'],
              'original_CLI_seconds':gpu0['raw_dwi_cli_total_runtime_seconds'],'independent_recovery_CLI_seconds':gpu1['raw_dwi_cli_total_runtime_seconds'],
              'time_ratio':'not_calculated: different monitor backend and shared GPU load; not a speed comparison',
              'original_export':verify_exports(analyzer,wall0),'recovery_export':verify_exports(analyzer,wall1),
              'file_count':len(records),'bytes_equal_files':sum(r['bytes_equal'] for r in records.values()),
              'byte_different_paths':[p for p,r in records.items() if not r['bytes_equal']],
              'parsed_array_artifact_count':len(numeric),'all_scientific_array_bits_equal':all(numeric),
              'array_different_paths':[p for p,r in records.items() if r['parsed'].get('all_array_bits_equal') is False],
              'original_input_FS_source_reports_preserved':True,'artifacts':records}
        report['cases'][case_id]=case;atomic(ROOT/'science_comparison.json',report)
    require(sha(CONFIG)==CONFIG_SHA,'original config changed during CPU audit')
    report.update(status='CPU_comparison_completed',all_scientific_array_bits_equal=all(c['all_scientific_array_bits_equal'] for c in report['cases'].values()),
                  end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic(ROOT/'science_comparison.json',report)
    print(json.dumps({'status':report['status'],'array_bits_equal':report['all_scientific_array_bits_equal'],
                      'cases':{k:{n:v[n] for n in ('file_count','bytes_equal_files','array_different_paths','recovery_memory','independent_recovery_CLI_seconds')} for k,v in report['cases'].items()}},indent=2))

if __name__=='__main__': run()
