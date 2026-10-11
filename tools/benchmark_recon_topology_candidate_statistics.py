"""完整真实拓扑候选统计 before→after 回归；不运行原版或完整 GA。"""
from __future__ import annotations

import time
_START = time.perf_counter()
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from benchmark_recon_topology_candidate_displacement import difference
from fnit.recon_all import topology_candidate_statistics as implementation
from fnit.recon_all.thread_budget import thread_budget
_IMPORT = time.perf_counter() - _START


def load_statistics_capture(*, capture_path: Path) -> dict:
    """FNITRST1 小端固定字段 → 完整状态副本，严格核对 magic/长度。

    文件包含8字节magic、u32 N/K/B/E、F32调用入口fitness，随后完整
    N marks前/后、N F32 curvbak、K×2映射/status、B border、E×3边/状态、
    K RP使用次数前/后、K F32运行均值前/后。原版执行只在隔离参考环境。
    返回字段区分before/after；after只用于比较。缺字段/未知尾字节报错。
    """
    raw = Path(capture_path).read_bytes()
    if len(raw) < 28 or raw[:8] != b'FNITRST1':
        raise ValueError('complete FNITRST1 statistics capture required')
    n, k, b, e = map(int, np.frombuffer(raw, '<u4', 4, 8))
    if n < 1:
        raise ValueError('positive complete vertex count required')
    offset = 24
    def take(dtype, count, shape=None):
        nonlocal offset
        size = np.dtype(dtype).itemsize * count
        if offset + size > len(raw):
            raise ValueError('truncated complete statistics fields')
        out = np.frombuffer(raw, dtype, count, offset).copy()
        offset += size
        return out.reshape(shape) if shape is not None else out
    fitness = take('<f4', 1)[0]
    fields = dict(candidate_fitness=fitness, marks_before=take('<i4',n),
        marks_after=take('<i4',n), vertex_curvbak=take('<f4',n),
        mapped_status=take('<i4',2*k,(k,2)), border=take('<i4',b),
        edges=take('<i4',3*e,(e,3)), counts_before=take('<i4',k),
        counts_after=take('<i4',k), fitness_before=take('<f4',k),
        fitness_after=take('<f4',k))
    if offset != len(raw):
        raise ValueError('unknown trailing statistics data')
    return fields


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, capture, native_repeat, observer_receipt, output, threads) -> dict:
    """冻结真实FNITRST1前状态 → 自有更新，完整参考后状态只作比较。

    capture/native_repeat是两次固定参考，observer_receipt绑定源版本/程序；
    output为不存在的隔离目录，threads为正整数（默认CLI 4）。只使用
    before/mapped/status/边/curvbak求解，不读after统计作输入，不启动原
    程序。输出JSON、私有NPZ、PNG，包含完整marks/RP比较和真实时间，
    不称完整候选自产或GA，不代表整例提速。失败保留报告，CLI退出非零。
    """
    if output.exists():
        raise FileExistsError('new isolated output required')
    if isinstance(threads,bool) or not isinstance(threads,int) or threads < 1:
        raise ValueError('positive thread budget required')
    started = time.perf_counter()
    fields = load_statistics_capture(capture_path=capture)
    repeat = load_statistics_capture(capture_path=native_repeat)
    args = dict(vertex_marks=fields['marks_before'], vertex_curvbak=fields['vertex_curvbak'],
        candidate_edges=fields['edges'][:,:2], edge_used=fields['edges'][:,2],
        translated_defect_vertices=fields['mapped_status'][:,0],
        discarded_defect_vertices=fields['mapped_status'][:,1] == 1,
        translated_border_vertices=fields['border'], use_counts=fields['counts_before'],
        vertex_fitness=fields['fitness_before'], candidate_fitness=fields['candidate_fitness'])
    with thread_budget(threads=threads):
        tick = time.perf_counter(); candidate = implementation.update_candidate_statistics(**args)
        first = time.perf_counter() - tick
        tick = time.perf_counter(); again = implementation.update_candidate_statistics(**args)
        warm = time.perf_counter() - tick
    comparisons = {key:difference(candidate[key],fields[reference]) for key,reference in [
        ('vertex_marks','marks_after'),('use_counts','counts_after'),('vertex_fitness','fitness_after')]}
    self_repeat = {key:difference(value,again[key]) for key,value in candidate.items()}
    native = {key:difference(value,repeat[key]) for key,value in fields.items()}
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz',**candidate)
    figure = output/'statistics_error.png'
    _plot(fields,candidate,figure)
    report = dict(status='complete_frozen_candidate_statistics_regression',
        strict_statistics_passed=all(row['bitwise_equal'] for row in comparisons.values()),
        candidate_repeat_exact=all(row['bitwise_equal'] for row in self_repeat.values()),
        native_captured_fields_repeat_exact=all(row['bitwise_equal'] for row in native.values()),
        comparisons=comparisons,candidate_repeat=self_repeat,native_repeat=native,
        full_vertices=len(fields['vertex_curvbak']),complete_defect_entries=len(fields['counts_before']),
        discarded_entries=int(np.count_nonzero(fields['mapped_status'][:,1]==1)),
        updated_entries=len(candidate['updated_defect_entries']),complete_edges=len(fields['edges']),
        seconds=dict(top_level_imports=_IMPORT,first_complete_statistics_api=first,
            repeat_complete_statistics_api=warm,api_with_load_npz_and_png_io_before_report_write=time.perf_counter()-started,
            script_wall_before_report_write=time.perf_counter()-_START),
        module_sha256=sha(implementation.__file__),benchmark_sha256=sha(__file__),
        source_inputs_sha256=dict(capture=sha(capture),native_repeat=sha(native_repeat),observer_receipt=sha(observer_receipt)),
        observer_receipt=json.loads(observer_receipt.read_text()),candidate_npz_sha256=sha(output/'candidate.private.npz'),
        figure_sha256=sha(figure),cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),
        host=platform.node(),python=platform.python_version(),numpy=np.__version__,gpu_used=False,
        process_gpu_memory_bytes=None,native_component_seconds=None,original_component_speedup=None,
        scope='complete before-to-after statistics with explicit frozen candidate inputs; no raw candidate or GA production',
        overall_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _plot(fields, candidate, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    x=np.arange(len(fields['fitness_after']))
    axes[0].plot(x,candidate['vertex_fitness'],'.',label='FNIT');axes[0].plot(x,fields['fitness_after'],'+',label='source')
    axes[0].set(xlabel='original defect entry',ylabel='running feedback mean');axes[0].legend()
    axes[1].plot(x,candidate['vertex_fitness'].astype(np.float64)-fields['fitness_after'].astype(np.float64),'.')
    axes[1].set(xlabel='original defect entry',ylabel='FNIT minus source')
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','observer-receipt','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    report=run(**vars(parser.parse_args()))
    if not all(report[key] for key in ['strict_statistics_passed','candidate_repeat_exact','native_captured_fields_repeat_exact']):
        raise SystemExit(1)


if __name__=='__main__': main()
