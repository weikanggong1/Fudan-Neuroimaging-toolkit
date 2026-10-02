"""同 GPU、同 allocator 环境的启动诊断，不运行重建，不改动生产结果。"""
import json
import os
import platform
from pathlib import Path
import subprocess
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
directory = root / 'volume_parity_20260930'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTORCH_NO_CUDA_MEMORY_CACHING='1',
           PYTHONPATH=str(directory / 'performance_e036f57_code/src'))
records = []
for index, threads in enumerate([None, 4, None, 4, None, 4]):
    code = 'import json,torch; from fnit.recon_all.native_free import run_recon_all_python; '
    if threads is not None:
        code += f'torch.set_num_threads({threads}); '
    code += ('torch.cuda.init(); x=torch.ones(1,dtype=torch.float32,device="cuda:0"); '
             'torch.cuda.synchronize("cuda:0"); '
             'print(json.dumps({"threads":torch.get_num_threads(),"free_total_bytes":'
             'torch.cuda.mem_get_info(0),"value":x.item()}))')
    gpu_before = subprocess.check_output(['nvidia-smi', '--id=1',
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    started = time.perf_counter()
    result = subprocess.run([str(root / 'fnit_main_env/bin/python'), '-c', code],
                            env=env, text=True, capture_output=True)
    records.append({'index': index, 'set_threads_before_init': threads,
                    'command': code, 'seconds': time.perf_counter()-started,
                    'exit_code': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr, 'gpu_before': gpu_before})
    print(json.dumps(records[-1]), flush=True)
report = {'host': platform.node(), 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'physical_gpu': 1, 'precision': 'float32', 'allocator_cache': False,
          'source_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68', 'records': records}
(directory / 'cuda_init_probe_e036f57.json').write_text(json.dumps(report,indent=2)+'\n')
