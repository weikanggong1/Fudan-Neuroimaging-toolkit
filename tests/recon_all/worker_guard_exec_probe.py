"""真实 exec 合同探针；替换第一个数值入口，不计算影像或初始化 CUDA。"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

import torch


def main():
    module, executable, marker, receipt = sys.argv[1:5]
    arguments = sys.argv[5:]
    before = dict(os.environ)
    state = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)

    def first_stage(*args, **kwargs):
        subprocess.run([executable, marker], check=True)
        raise LookupError('contract probe stopped before numerical computation')

    target = ('torch.set_num_threads' if module.endswith('talairach_synthmorph')
              else 'torch.cuda.is_initialized')
    sys.argv = [module, *arguments]
    error = None
    try:
        with patch(target, side_effect=first_stage):
            runpy.run_module(module, run_name='__main__')
    except (RuntimeError, LookupError) as failure:
        error = type(failure).__name__
    Path(receipt).write_text(json.dumps({
        'error': error, 'probe_executed': Path(marker).exists(),
        'environment_restored': dict(os.environ) == before,
        'thread_count_restored': torch.get_num_threads() == state[0],
        # Talairach CLI has its own pre-existing precision setup before stage
        # entry; the decorator does not replace that protocol.
        'entry_precision': list(state[1:]),
        'final_precision': [torch.backends.cuda.matmul.allow_tf32,
                            torch.backends.cudnn.allow_tf32],
        'inherited_flag': before.get('FNIT_RECON_FORBID_ORIGINAL_STAGES'),
    }))
    if error:
        raise SystemExit(19)
    raise AssertionError('probe must stop before numerical computation')


if __name__ == '__main__':
    main()
