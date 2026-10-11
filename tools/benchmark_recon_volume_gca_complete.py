"""同一自产nu/brainmask的首方向、完整Python和隔离原生GCA参考。"""
import argparse,hashlib,json,os,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--nu',type=Path,required=True);p.add_argument('--mask',type=Path,required=True)
p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=('python','native-reference'),required=True)
p.add_argument('--binary',type=Path);p.add_argument('--device',default='cuda:0')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
import numpy as np
import torch
from fnit.recon_all.assets import ASSET_FILES
from fnit.recon_all.mri_em_register_python import register_t1
from fnit.recon_all.ca_normalize_python import read_voxel_lta
size,digest,*_=ASSET_FILES['average/RB_all_2020-01-02.gca']
if a.atlas.stat().st_size!=size or sha(a.atlas)!=digest:raise ValueError('GCA resource verification failed')
report={'scope':'frozen_own_complete_GCA_phase_not_whole','mode':a.mode,
        'input_sha256':{'nu':sha(a.nu),'brainmask':sha(a.mask),'GCA':digest},
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'strict_matrix_tolerance_declared':0,
        'overall_equivalence':'not_assessed','target_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'timing_boundary':'native subprocess spawn/load/IO' if a.mode=='native-reference' else 'Python file API load/transfer/JIT/IO excluding interpreter and imports'}
output=a.output/'talairach.lta'
try:
    if a.mode=='native-reference':
        if a.binary is None:raise ValueError('explicit independent reference binary is required')
        report['binary_sha256']=sha(a.binary)
        tick=time.perf_counter()
        with (a.output/'native.stdout.private.log').open('w') as log:
            subprocess.run([str(a.binary),'-uns','3','-mask',str(a.mask),str(a.nu),str(a.atlas),str(output)],check=True,stdout=log,stderr=subprocess.STDOUT)
        report['seconds_including_spawn_load_IO']=time.perf_counter()-tick
    else:
        import inspect
        source=Path(inspect.getfile(register_t1)).parent
        report['source_sha256']={name:sha(source/name) for name in ('mri_em_register.py','mri_em_register_python.py','mri_em_register_em_candidate.py','mri_em_register_score_gpu.py','mri_em_register_optimizer.py','mri_em_register_complete.py')}
        if (source/'mri_em_register_complete.py').exists():report['source_sha256']['mri_em_register_complete.py']=sha(source/'mri_em_register_complete.py')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
        tick=time.perf_counter()
        api=register_t1(a.nu,a.atlas,a.mask,output,device=a.device,search_backend='torch',
                        reduce_on_device=True,inverse_backend='torch',candidate_chunk=64,sample_chunk=8192)
        torch.cuda.synchronize(a.device)
        report['seconds_including_spawn_load_IO']=time.perf_counter()-tick
        report['api']={k:v for k,v in api.items() if k!='output'}
        report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(a.device)
        report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved(a.device)
        report['actual_precision']={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False}
        report['nvml_own_memory_bytes']=None;report['nvml_attribution']='container mapping not established; Torch counters are own allocator only'
except Exception as error:
    report.update(execution_complete=False,error_type=type(error).__name__)
    (a.output/'metrics.public.json').write_text(json.dumps(report,indent=2)+'\n')
    raise
try:
    report['matrix']=read_voxel_lta(output).tolist();report['output_sha256']=sha(output)
    report['execution_complete']=True
    (a.output/'metrics.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
except Exception as error:
    report.pop('api',None)
    report.update(execution_complete=False,error_type=type(error).__name__,report_failure=True)
    (a.output/'metrics.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise
