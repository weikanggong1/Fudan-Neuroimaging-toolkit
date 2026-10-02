"""独立 exec 半球 worker；环境在导入 Torch/Numba 前由父进程设置。"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    request_path, report_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'started_monotonic': started}
    try:
        import torch
        from .profiling import StageProfiler, configure_cuda_allocator
        from .thread_budget import thread_budget
        # 精度按调度方当前状态传递，进程内互不影响；不启用 autocast。
        torch.backends.cuda.matmul.allow_tf32 = request['precision']['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = request['precision']['cudnn_tf32']
        torch.set_num_interop_threads(1)
        allocator = configure_cuda_allocator(request['device'], 'auto')
        bootstrap_tick = time.monotonic()
        if torch.device(request['device']).type == 'cuda':
            # 先显式FP32分配再同步；与既有SynthMorph首CUDA分配规则一致。
            bootstrap = torch.empty(1, dtype=torch.float32, device=request['device'])
            torch.cuda.synchronize(torch.device(request['device']))
            properties = torch.cuda.get_device_properties(torch.device(request['device']))
            report['actual_cuda_device'] = {'name': properties.name,
                'uuid': str(getattr(properties, 'uuid', 'unavailable'))}
            del bootstrap
        report['cuda_bootstrap_seconds'] = time.monotonic() - bootstrap_tick
        profiler = StageProfiler(device=request['device'],
                                 synchronize=request['profile_stages'], allocator=allocator)
        module, name = request['callable'].rsplit(':', 1)
        function = getattr(importlib.import_module(module), name)
        with thread_budget(threads=request['threads']) as budget:
            try:
                report['value'] = profiler.run(request['operation'], function,
                                               **request['kwargs'])
                if torch.device(request['device']).type == 'cuda' and torch.cuda.is_initialized():
                    # 完成屏障是进程生命周期必需，不能仅计异步 kernel 提交时间。
                    torch.cuda.synchronize(torch.device(request['device']))
            finally:
                report['stage'] = profiler.last_row
                report['thread_budget'] = budget
        report['status'] = 'complete'
        report['precision'] = request['precision']
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    report['finished_monotonic'] = time.monotonic()
    report['total_seconds'] = report['finished_monotonic'] - started
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
