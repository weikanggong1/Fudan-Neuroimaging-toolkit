"""独立 exec 半球 worker；环境在导入 Torch/Numba 前由父进程设置。"""
from __future__ import annotations
import importlib
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback


def _is_cuda_oom(error):
    """Match CUDA's explicit error prefix, including the real C10 runtime message.

    torch.cuda.OutOfMemoryError aliases a general Torch exception: its type alone
    cannot distinguish host allocation failure. Parent separately gates phase.
    """
    return (isinstance(error, RuntimeError) and re.match(
        r'^CUDA (?:error:\s*)?out of memory(?:[.!:]|\s|$)', str(error).lstrip()) is not None)


def _wait_for_go(request):
    """Bound only pre-callable readiness; never imposes an algorithm deadline."""
    while True:
        if os.getppid() != request['parent_pid']:
            raise RuntimeError('worker parent exited before GO')
        if time.monotonic() >= request['startup_deadline_monotonic']:
            raise TimeoutError('worker READY deadline exceeded before GO')
        if Path(request['go_path']).is_file():
            return
        time.sleep(.02)


def main():
    request_path, report_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'started_monotonic': started,
              'logical_device': request.get('device'), 'precision': request.get('precision'),
              'operation_entered': False}
    phase = 'imports'
    torch = None
    try:
        import torch
        from .profiling import StageProfiler, configure_cuda_allocator
        from .thread_budget import thread_budget
        phase = 'policy'
        # 精度按调度方当前状态传递，进程内互不影响；不启用 autocast。
        torch.backends.cuda.matmul.allow_tf32 = request['precision']['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = request['precision']['cudnn_tf32']
        torch.set_num_interop_threads(1)
        allocator = configure_cuda_allocator(request['device'], request.get('allocator_policy', 'auto'))
        report['cuda_allocator'] = allocator
        bootstrap_tick = time.monotonic()
        if torch.device(request['device']).type == 'cuda':
            # 先显式FP32分配再同步；与既有SynthMorph首CUDA分配规则一致。
            phase = 'first_allocation'
            bootstrap = torch.empty(1, dtype=torch.float32, device=request['device'])
            phase = 'sync'
            torch.cuda.synchronize(torch.device(request['device']))
            phase = 'device_properties'
            properties = torch.cuda.get_device_properties(torch.device(request['device']))
            report['actual_cuda_device'] = {'name': properties.name,
                'uuid': str(getattr(properties, 'uuid', 'unavailable'))}
            del bootstrap
        report['cuda_bootstrap_seconds'] = time.monotonic() - bootstrap_tick
        # The actual process retains its CUDA context while the parent admits both sides.
        if 'ready_path' in request:
            phase = 'startup_barrier'
            report['ready_monotonic'] = time.monotonic()
            ready_path = Path(request['ready_path'])
            temporary = ready_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'pid': os.getpid(),
                'operation_entered': False, 'ready_monotonic': report['ready_monotonic']}))
            temporary.replace(ready_path)
            _wait_for_go(request)
            report['go_monotonic'] = time.monotonic()
        phase = 'import_callable'
        profiler = StageProfiler(device=request['device'],
                                 synchronize=request['profile_stages'], allocator=allocator)
        module, name = request['callable'].rsplit(':', 1)
        function = getattr(importlib.import_module(module), name)
        with thread_budget(threads=request['threads']) as budget:
            try:
                phase = 'operation'
                report['operation_entered'] = True
                report['operation_started_monotonic'] = time.monotonic()
                report['value'] = profiler.run(request['operation'], function,
                                               **request['kwargs'])
                report['operation_finished_monotonic'] = time.monotonic()
                if torch.device(request['device']).type == 'cuda' and torch.cuda.is_initialized():
                    # 完成屏障是进程生命周期必需，不能仅计异步 kernel 提交时间。
                    phase = 'sync'
                    torch.cuda.synchronize(torch.device(request['device']))
            finally:
                report['stage'] = profiler.last_row
                report['thread_budget'] = budget
        report['status'] = 'complete'
        report['precision'] = request['precision']
    except BaseException as error:
        report.update(status='failed', failure_phase=phase,
                      cuda_oom=_is_cuda_oom(error),
                      error=repr(error),
                      traceback=traceback.format_exc())
        # is_initialized reads Python state only; no CUDA query after failure.
        try:
            report['cuda_initialized_after_failure'] = (
                torch.cuda.is_initialized() if torch is not None else None)
        except Exception as diagnostic_error:
            report['cuda_initialized_after_failure'] = None
            report['failure_state_diagnostic_error'] = repr(diagnostic_error)
        print(report['traceback'], file=sys.stderr, flush=True)
    report['finished_monotonic'] = time.monotonic()
    report['total_seconds'] = report['finished_monotonic'] - started
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
