"""完整首次 unMRI 体积聚合/反馈冻结回归；参考只在此隔离诊断读取。"""
from __future__ import annotations
import time
_ENTRY_TIME = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
from benchmark_recon_topology_unmri_fields import load_volume_capture
from fnit.recon_all import topology_defect_volume_likelihood as implementation
from fnit.recon_all.thread_budget import thread_budget
_IMPORT_SECONDS = time.perf_counter() - _ENTRY_TIME


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape:
        return dict(shape_equal=False, candidate_shape=list(a.shape), reference_shape=list(b.shape))
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(shape_equal=True, different_elements=int(np.count_nonzero(a != b)),
                max_abs=float(error.max(initial=0)), p99_abs=float(np.percentile(error, 99)) if error.size else 0.0)


def arguments(fields):
    intensity = fields['intensity']
    if not np.isfinite(intensity).all() or np.any(intensity < 0) or np.any(intensity > 255) or np.any(intensity != np.floor(intensity)):
        raise ValueError('capture intensity must preserve actual uint8 MRI values')
    means = fields['means_and_sentinel']
    geometry = fields['intensity_geometry']
    vertex, defect = fields['vertex_fields'], fields['defect_vertices']
    return dict(defect_intensity_volume=intensity.astype(np.uint8),
        signed_distance_mm=fields['signed_distance'],
        white_log_probability=fields['white_log_pdf'], gray_log_probability=fields['gray_log_pdf'],
        initial_white_mean_log_probability=means[0], initial_gray_mean_log_probability=means[1],
        initial_white_mean_intensity=means[2], initial_gray_mean_intensity=means[3],
        volume_origin_mm=geometry[:3], volume_index_scale_per_mm=geometry[3:6],
        vertex_sampling_positions_mm=vertex[:, 6:9], initial_vertex_curvbak=vertex[:, 9],
        translated_defect_vertices=defect[:, 1], discarded_defect_vertices=defect[:, 2] == 1,
        distance_unset_sentinel_mm=means[4])


def evaluate(arguments):
    start = time.perf_counter()
    result = implementation.defect_volume_likelihood(**arguments)
    return result, time.perf_counter() - start


def plot(fields, candidate, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vertex = fields['vertex_fields']
    active = fields['defect_vertices'][fields['defect_vertices'][:, 2] != 1, 1]
    xyz = vertex[active, :3]
    before, after = vertex[active, 9], candidate['vertex_curvbak'][active]
    error = after.astype(np.float64) - fields['reference_curvbak'][active].astype(np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    volume = fields['signed_distance'].copy()
    volume[volume == fields['means_and_sentinel'][4]] = np.nan
    image = axes[0].imshow(volume[:, :, volume.shape[2] // 2].T, origin='lower', cmap='coolwarm', vmin=-2, vmax=2)
    axes[0].set_title('Frozen signed-distance input (mm)')
    fig.colorbar(image, ax=axes[0], shrink=.75)
    for ax, values, title in [(axes[1], after - before, 'FNIT curvbak update'),
                               (axes[2], error, 'FNIT minus source curvbak')]:
        image = ax.scatter(xyz[:, 1], xyz[:, 2], c=values, cmap='coolwarm', s=24)
        ax.set_aspect('equal'); ax.set_xlabel('surface RAS Y (mm)'); ax.set_ylabel('Z (mm)')
        ax.set_title(title); fig.colorbar(image, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run(*, capture, native_repeat, observer_receipt, output, threads):
    """完整 FNITVOL1 和重复参考→独立聚合、全 curvbak 回归、SHA、PNG。

    capture/native_repeat 为固定源码首次unMRI、距离计算之后的只读完整
    快照；仅诊断输入。observer_receipt绑定源码/程序/构建；output必须
    不存在，threads正整数默认4并恢复父线程设置。返回机器JSON；写
    candidate.npz私有状态、report.private.json与授权公开T1衍生PNG。
    不重算距离、不改生产参考/影像，不把首次聚合等同完整GA。
    来源损坏、类型/状态非法或参考重复性失败均保留结果或抛异常；
    精度容差预先为源float32字段/全curvbak逐位相同，阈值不随结果变。
    """
    start = time.perf_counter()
    if output.exists():
        raise FileExistsError('fresh isolated diagnostic output required')
    if threads < 1:
        raise ValueError('positive thread budget required')
    fields = load_volume_capture(capture_path=capture)
    repeated_fields = load_volume_capture(capture_path=native_repeat)
    args = arguments(fields)
    output.mkdir(parents=True)
    with thread_budget(threads=threads):
        first, first_seconds = evaluate(args)
        second, second_seconds = evaluate(args)
    names = ['white_ll', 'gray_ll', 'white_mean_intensity', 'gray_mean_intensity',
             'max_distance_mm', 'unmri_ll', 'nwhite', 'ngray']
    values = np.array([first[name] for name in names], np.float32)
    comparisons = {name: difference(first[name], fields['reference_results'][i]) for i, name in enumerate(names)}
    comparisons['vertex_curvbak'] = difference(first['vertex_curvbak'], fields['reference_curvbak'])
    repeat = {name: difference(first[name], second[name]) for name in names}
    repeat['vertex_curvbak'] = difference(first['vertex_curvbak'], second['vertex_curvbak'])
    reference_repeat = {name: difference(value, repeated_fields[name]) for name, value in fields.items()}
    np.savez_compressed(output / 'candidate.npz', results=values, vertex_curvbak=first['vertex_curvbak'])
    from benchmark_recon_topology_unmri_fields import __file__ as parser_path
    source_receipt_raw = observer_receipt.read_bytes()
    report = dict(status='complete_frozen_unmri_volume_likelihood_regression',
        strict_source_fields_and_feedback_passed=all(row.get('different_elements') == 0 for row in comparisons.values()),
        candidate_repeat_exact=all(row.get('different_elements') == 0 for row in repeat.values()),
        native_full_capture_repeat_exact=all(row.get('different_elements') == 0 for row in reference_repeat.values()),
        scope='complete first unMRI volume aggregation/curvbak feedback only; distance field, dynamic mutation fitness and full GA not evaluated',
        native_source_receipt=json.loads(source_receipt_raw), native_source_receipt_sha256=hashlib.sha256(source_receipt_raw).hexdigest(),
        capture_sha256=sha(capture), native_repeat_capture_sha256=sha(native_repeat),
        module_sha256=dict(implementation=sha(implementation.__file__), capture_parser=sha(parser_path), benchmark=sha(__file__)),
        input_fields={name:dict(shape=list(value.shape),dtype=str(value.dtype),
            data_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()) for name,value in fields.items()},
        candidate_scores={name:first[name] for name in names}, raw_max_distance_mm=first['raw_max_distance_mm'],
        reference_scores={name:float(fields['reference_results'][i]) for i,name in enumerate(names)},
        scores_and_feedback=comparisons, candidate_repeat=repeat, native_repeat=reference_repeat,
        candidate_npz_sha256=sha(output/'candidate.npz'), vertex_count=len(fields['vertex_fields']),
        defect_vertices=len(fields['defect_vertices']), volume_shape=list(fields['signed_distance'].shape),
        discarded_vertices=int(np.count_nonzero(fields['defect_vertices'][:,2]==1)),
        curvbak_changed_vertices=int(np.count_nonzero(first['vertex_curvbak'] != args['initial_vertex_curvbak'])),
        coordinate_space='surface RAS mm / source local volume reciprocal spacing mm^-1; volume axes XYZ',
        seconds=dict(first_complete_api_including_available_jit=first_seconds, repeat_complete_api=second_seconds,
            top_level_imports=_IMPORT_SECONDS, api_with_load_validation_npz_and_hash=time.perf_counter()-start),
        native_component_seconds=None, native_component_timing_limitation='observer includes upstream distance/preparation; no isolated aggregate timer or speedup ratio',
        cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)), host=platform.node(),
        python=platform.python_version(),numpy=np.__version__,gpu_used=False,gpu_process_memory_bytes=None,
        overall_equivalence='not_assessed')
    tick=time.perf_counter(); plot(fields,first,output/'aggregate.png')
    report['seconds']['plot_with_import_and_png_write']=time.perf_counter()-tick
    report['figure_sha256']=sha(output/'aggregate.png')
    report['seconds']['tool_entry_wall_excluding_preceding_interpreter_bootstrap_and_final_report_write']=time.perf_counter()-_ENTRY_TIME
    (output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','observer-receipt','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    run(**vars(parser.parse_args()))
