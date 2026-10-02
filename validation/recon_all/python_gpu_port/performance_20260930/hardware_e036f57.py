"""在实际计算主机记录版本、硬件及当前候选父进程线程；不初始化 CUDA。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

import nibabel
import numba
import numpy
import scipy
import torch


def main() -> None:
    """读取固定 Conda 运行环境及 /proc，写出新的逐主机 JSON；失败时抛异常。"""
    host = platform.node()
    root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
    directory = root / 'volume_parity_20260930'
    subject = 'full_sub01_e036f57_uuid' if host == 'gpucw1' else 'full_sub02_e036f57'
    marker = ('run_initialized_cuda_api_e036f57_uuid.py' if host == 'gpucw1'
              else '-m fnit.recon_all.native_free')
    cpu = next(line.split(':', 1)[1].strip() for line in
               Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'host': host, 'cpu': cpu,
              'code_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68',
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'numpy': numpy.__version__,
              'scipy': scipy.__version__, 'nibabel': nibabel.__version__,
              'numba': numba.__version__, 'numba_default_threads_for_probe': numba.get_num_threads(),
              'actual_candidate_parent_processes': []}
    keys = ['OMP_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_NO_CUDA_MEMORY_CACHING']
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or process.stat().st_uid != os.getuid():
            continue
        try:
            cmd = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if subject not in cmd or marker not in cmd or not cmd.startswith(str(root / 'fnit_main_env/bin/python')):
                continue
            status = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
            env = dict(item.split('=', 1) for item in (process / 'environ').read_text().split('\0') if '=' in item)
            report['actual_candidate_parent_processes'].append(
                {'pid': int(process.name), 'cmd': cmd, 'threads': status['Threads'].strip(),
                 'thread_environment': {key: env.get(key) for key in keys}})
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(report['actual_candidate_parent_processes']) != 1:
        raise ValueError('expected one active candidate parent; collect during the run')
    if host == 'gpucw1':
        report['gpu_nvidia_smi'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).splitlines()
    with (directory / f'hardware_{host}_e036f57.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
