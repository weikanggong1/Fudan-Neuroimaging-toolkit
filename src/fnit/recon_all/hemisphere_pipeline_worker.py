"""持久半球执行器：每阶段完成后暂停，等父进程审计/发布后继续。"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def execute(subject, hemi, device, threads, operation, *, operations,
            control_directory, parent_pid, precision, profile_stages=False):
    """fresh exec 内串行调用既有半球函数；输入/返回结构见持久调度说明。"""
    import torch
    from .profiling import StageProfiler, autocast_state, configure_cuda_allocator
    control = Path(control_directory)
    allocator = configure_cuda_allocator(device)
    rows = []
    for sequence, specification in enumerate(operations):
        name = specification['operation']
        receipt_path = control / f'{hemi}.{sequence:03d}.stage.json'
        acknowledgement = control / f'{hemi}.{sequence:03d}.ack.json'
        row = {'operation': name, 'sequence': sequence, 'hemi': hemi,
               'pid': os.getpid(), 'status': 'running',
               'started_monotonic': time.monotonic()}
        try:
            # 旧模式每阶段 fresh exec 重建精度；持久模式在当前独立进程复位。
            torch.backends.cuda.matmul.allow_tf32 = precision['matmul_tf32']
            torch.backends.cudnn.allow_tf32 = precision['cudnn_tf32']
            profiler = StageProfiler(device=device, synchronize=profile_stages, allocator=allocator,
                                     reset_peak_memory=False)
            module, attribute = specification['callable_path'].rsplit(':', 1)
            function = getattr(importlib.import_module(module), attribute)
            arguments = dict(specification['kwargs'], subject=str(subject), hemi=hemi,
                             device=device, threads=threads, operation=name)
            row['value'] = profiler.run(name, function, **arguments)
            if torch.device(device).type == 'cuda':
                # 发布生命周期屏障：文件/张量完成后才向父进程发送阶段回执。
                torch.cuda.synchronize(torch.device(device))
            row['status'] = 'complete'
            row['stage'] = profiler.last_row
            row['precision_actual_after_operation'] = {
                'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32),
                'autocast': {kind: autocast_state(kind) for kind in ('cpu', 'cuda')}}
        except BaseException as error:
            row.update(status='failed', error=repr(error))
            quality = getattr(error, 'diagnostics', None)
            if isinstance(quality, dict):
                row['quality_failure'] = quality
            raise
        finally:
            row['finished_monotonic'] = time.monotonic()
            row['seconds_including_io'] = row['finished_monotonic'] - row['started_monotonic']
            atomic_json(receipt_path, row)
        rows.append(row)
        while not acknowledgement.is_file():
            if os.getppid() != parent_pid:
                raise RuntimeError('pipeline parent exited before stage acknowledgement')
            time.sleep(.02)
        ack = json.loads(acknowledgement.read_text())
        if ack.get('sequence') != sequence or ack.get('pid') != os.getpid():
            raise RuntimeError('invalid pipeline stage acknowledgement')
    return {'operations': rows, 'private_subject': str(subject)}
