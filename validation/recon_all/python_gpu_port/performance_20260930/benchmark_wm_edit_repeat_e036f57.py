"""同主机重放未修改的 WM 编辑，区分单独 C++ 耗时与整例异常等待。

固定输入为 full_sub01_e036f57_uuid 的四张自产 conform MRI（256³），
不读取官方结果。输出为 sub01/wm_edit_repeat_e036f57_retry 中两次新 WM、日志及
report.json，含输入/源码/程序哈希、墙钟、子进程 CPU 时间和最大 RSS。
必要文件缺失、输出不对应或体素/几何改变时失败；不修改生产输入或续接整例。
原命令和全部固定参数由 command 数组原样记录。CPU 原生程序不初始化 CUDA。
"""
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import nibabel as nib
import numpy as np

from fingerprint_e036f57 import fingerprint


def main() -> None:
    """用两个新目录运行相同命令；仅此固定实验，没有未声明的输入选项。"""
    root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
    directory = root / 'volume_parity_20260930'
    source = directory / 'full_sub01_e036f57_uuid/mri'
    output = directory / 'sub01/wm_edit_repeat_e036f57_retry'
    output.mkdir(exist_ok=False)
    binary = root / 'fnit_main_env/bin/mri_edit_wm_with_aseg'
    names = ['entowm.mgz', 'aseg.presurf.mgz', 'wm.seg.mgz', 'brain.mgz']
    command = [str(binary), '-keep-in', '-fix-ento-wm', 'entowm.mgz', '3', '255', '255',
               '-fix-acj', 'aseg.presurf.mgz', '255', '255', '-fill-seg-wm', '-fix-scm-ha',
               '1', 'wm.seg.mgz', 'brain.mgz', 'aseg.presurf.mgz', 'wm.asegedit.mgz']
    report = {'code_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68',
              'host': platform.node(), 'device': 'CPU; no CUDA context', 'command': command,
              'timing_scope': 'subprocess launch, native input/output and log writes; no parent CUDA synchronization',
              'binary': fingerprint(binary), 'script': fingerprint(Path(__file__)),
              'inputs': {name: fingerprint(source / name) for name in names}, 'repeats': []}
    env = dict(os.environ, FREESURFER_HOME=str(root / 'assets'))
    reference = nib.load(str(source / 'wm.asegedit.mgz'))
    data = np.asanyarray(reference.dataobj)
    for index in range(2):
        work = output / str(index)
        work.mkdir()
        for name in names:
            (work / name).symlink_to(source / name)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        with (work / 'command.log').open('w') as log:
            subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        seconds = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        image = nib.load(str(work / 'wm.asegedit.mgz'))
        result = np.asanyarray(image.dataobj)
        difference = np.abs(result.astype(np.float64) - data.astype(np.float64))
        row = {'seconds': seconds, 'child_user_seconds': after.ru_utime - before.ru_utime,
               'child_system_seconds': after.ru_stime - before.ru_stime,
               'children_max_rss_bytes': after.ru_maxrss * 1024, 'shape': [int(n) for n in image.shape],
               'dtype': str(result.dtype), 'geometry_equal': bool(np.array_equal(image.affine, reference.affine)),
               'different_voxels': int(np.count_nonzero(difference)),
               'maximum_error': float(difference.max()), 'p99_error': float(np.percentile(difference, 99))}
        report['repeats'].append(row)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        assert row['geometry_equal'] and row['different_voxels'] == 0
        assert result.shape == data.shape and result.dtype == data.dtype


if __name__ == '__main__':
    main()
