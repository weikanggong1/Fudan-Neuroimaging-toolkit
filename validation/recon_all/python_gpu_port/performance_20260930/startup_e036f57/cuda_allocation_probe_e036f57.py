"""区分 CUDA 空分配、填充、复制及模块加载；仅诊断，不改变 FNIT 实现。"""
import json
import os
import platform
from pathlib import Path
import subprocess
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
directory = root / 'volume_parity_20260930'
records = []
for index, (cached, loader, operation) in enumerate([
        (False, None, 'empty'), (False, None, 'ones'), (True, None, 'ones'),
        (False, 'EAGER', 'ones'), (False, 'LAZY', 'ones'), (True, 'EAGER', 'copy')]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_LAUNCH_BLOCKING='1',
               PYTHONPATH=str(directory / 'performance_e036f57_code/src'))
    env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING', None)
    env.pop('CUDA_MODULE_LOADING', None)
    if not cached:
        env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    if loader:
        env['CUDA_MODULE_LOADING'] = loader
    code = ('import json,torch; from fnit.recon_all.native_free import run_recon_all_python; '
            'torch.set_num_threads(4); torch.cuda.init(); '
            'print("free_before",torch.cuda.mem_get_info(0),flush=True); ')
    code += {'empty': 'x=torch.empty(1,dtype=torch.float32,device="cuda:0"); ',
             'ones': 'x=torch.ones(1,dtype=torch.float32,device="cuda:0"); ',
             'copy': 'x=torch.ones(1,dtype=torch.float32).to("cuda:0"); '}[operation]
    code += 'torch.cuda.synchronize("cuda:0"); print("succeeded",torch.cuda.mem_get_info(0),flush=True)'
    gpu_before = subprocess.check_output(['nvidia-smi', '--id=1',
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    started = time.perf_counter()
    result = subprocess.run([str(root / 'fnit_main_env/bin/python'), '-c', code],
                            env=env, text=True, capture_output=True)
    records.append({'index': index, 'cache_enabled': cached, 'module_loading': loader,
                    'operation': operation, 'command': code, 'seconds': time.perf_counter()-started,
                    'exit_code': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr, 'gpu_before': gpu_before})
    print(json.dumps(records[-1]), flush=True)
report = {'host': platform.node(), 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'physical_gpu': 1, 'precision': 'float32', 'cuda_launch_blocking': True,
          'source_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68', 'records': records}
(directory / 'cuda_allocation_probe_e036f57.json').write_text(json.dumps(report,indent=2)+'\n')
