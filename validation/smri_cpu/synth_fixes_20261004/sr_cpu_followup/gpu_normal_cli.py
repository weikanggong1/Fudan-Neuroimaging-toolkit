"""Normal FNIT CLI with an explicit 20 GB allocator cap and tiny bootstrap.

GNU wall includes this bootstrap and the actual normal CLI, with one NIfTI
save. There are no CNN traces, private array copies, NPZ saves or comparisons.
"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import torch


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True)
    return result.stdout.strip().splitlines()


def main():
    output = Path(sys.argv[1]); args = sys.argv[2:]
    before = gpu_state()
    torch.cuda.device_count(); torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(20_000_000_000/properties.total_memory, 0)
    torch.arange(12, device='cuda', dtype=torch.float32).add_(1)
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    sys.argv = ['fnit', *args]
    runpy.run_module('fnit.cli', run_name='__main__')
    torch.cuda.synchronize(0)
    report = {'allocator_budget_bytes': 20_000_000_000,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
              'external_gpu_before': before, 'external_gpu_after': gpu_state(),
              'free_total_after_init_bytes': [free,total], 'torch_version': torch.__version__,
              'cuda_policy': {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32},
              'cpu_math_imported': 'fnit.synthsr._cpu_math' in sys.modules,
              'cpu_dispatch_imported': 'fnit.synthsr._cpu_inference' in sys.modules,
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'timing_scope': 'cold bootstrap plus actual normal public CLI, one NIfTI save, no trace/NPZ/hash/comparison',
              'status':'complete'}
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
