"""核对 CUDA 数字编号与 NVML UUID 映射，只运行小张量诊断。"""
import json
import os
import platform
from pathlib import Path
import subprocess
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
directory = root / 'volume_parity_20260930'
inventory = subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,uuid,pci.bus_id,memory.free', '--format=csv,noheader,nounits'], text=True)
uuid = [line.split(',')[1].strip() for line in inventory.splitlines()
        if int(line.split(',')[0]) == 1][0]
code = ('import json,torch; from fnit.recon_all.native_free import run_recon_all_python; '
        'torch.set_num_threads(4); torch.cuda.init(); '
        'print("actual_uuid",torch.cuda.get_device_properties(0).uuid,flush=True); '
        'print("free_bytes",torch.cuda.mem_get_info(0),flush=True); '
        'x=torch.ones(1,dtype=torch.float32,device="cuda:0"); '
        'torch.cuda.synchronize(0); print("succeeded",x.item(),flush=True)')
records = []
for mask in ['1', uuid, '1', uuid, '1', uuid]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=mask,
               PYTORCH_NO_CUDA_MEMORY_CACHING='1',
               PYTHONPATH=str(directory / 'performance_e036f57_code/src'))
    started = time.perf_counter()
    result = subprocess.run([str(root / 'fnit_main_env/bin/python'), '-c', code],
                            env=env, text=True, capture_output=True)
    records.append({'mask':mask, 'seconds':time.perf_counter()-started,
                    'exit_code':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr})
    print(json.dumps(records[-1]),flush=True)
report = {'host': platform.node(), 'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'target_nvml_uuid':uuid, 'inventory_before':inventory, 'command':code,
          'threads':4, 'precision':'float32', 'allocator_cache':False,
          'source_commit':'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68','records':records}
(directory/'cuda_device_probe_e036f57.json').write_text(json.dumps(report,indent=2)+'\n')
