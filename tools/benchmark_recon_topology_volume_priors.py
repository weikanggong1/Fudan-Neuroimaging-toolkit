"""原 MRI→完整局部缺陷强度/logP 的冻结回归；参考仅在诊断比较中读取。"""
from __future__ import annotations

import time
_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
import nibabel as nib
from benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
from fnit.recon_all import topology_defect_volume_priors as implementation
from fnit.recon_all import topology_edge_mri_score as sampler
from fnit.recon_all import place_surface_border as interpolation
from fnit.recon_all import mri_em_register as matrix_inverse
from fnit.recon_all.thread_budget import thread_budget
_IMPORT = time.perf_counter() - _ENTRY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def difference(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    result = dict(candidate_shape=list(a.shape), reference_shape=list(b.shape),
                  candidate_dtype=str(a.dtype), reference_dtype=str(b.dtype),
                  shape_equal=bool(a.shape == b.shape))
    if a.shape != b.shape:
        result.update(bitwise_equal=False, different_elements=None, max_abs=None, p99_abs=None)
        return result
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result.update(bitwise_equal=bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
                  different_elements=int(np.count_nonzero(a != b)),
                  max_abs=float(error.max(initial=0)),
                  p99_abs=float(np.percentile(error, 99)) if error.size else 0.)
    return result


def _evaluate(arguments):
    """隔离诊断计时；不改变公式、输入或生产调用，finally 恢复符号。"""
    names = ['_statistics', '_geometry', '_edge_mask', '_prior_fields']
    originals = {name: getattr(implementation, name) for name in names}
    timings = {}
    def timed(name, function):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.) + time.perf_counter() - start
        return call
    for name, function in originals.items():
        setattr(implementation, name, timed(name, function))
    start = time.perf_counter()
    try:
        result = implementation.defect_volume_priors(**arguments)
    finally:
        for name, function in originals.items():
            setattr(implementation, name, function)
    wall = time.perf_counter() - start
    timings['validation_copy_and_return'] = max(0., wall - sum(timings.values()))
    return result, wall, timings


def _plot(candidate, reference, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layer = candidate['defect_intensity_volume'].shape[2] // 2
    fields = [('defect_intensity_volume', 'intensity', 'Local MRI intensity'),
              ('white_log_probability', 'white_log_pdf', 'White log probability'),
              ('gray_log_probability', 'gray_log_pdf', 'Gray log probability')]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, (own_key, ref_key, label) in enumerate(fields):
        own, ref = candidate[own_key][:, :, layer], reference[ref_key][:, :, layer]
        low, high = float(min(own.min(), ref.min())), float(max(own.max(), ref.max()))
        for col, (data, title) in enumerate([(own, 'FNIT'), (ref, 'Source'),
                                             (own.astype(np.float64)-ref.astype(np.float64), 'FNIT minus source')]):
            image = axes[row, col].imshow(data.T, origin='lower', cmap='gray' if row == 0 and col < 2 else 'coolwarm',
                                          vmin=low if col < 2 else None, vmax=high if col < 2 else None)
            axes[row, col].set_title(label + ': ' + title)
            axes[row, col].set_xlabel('local X voxel'); axes[row, col].set_ylabel('local Y voxel')
            fig.colorbar(image, ax=axes[row, col], shrink=.8)
    fig.suptitle('Public ds000114 derived local fields; surface RAS sampling preserved')
    fig.tight_layout(); fig.savefig(output, dpi=145); plt.close(fig)


def run(*, creation_capture, likelihood_capture, creation_repeat, likelihood_repeat,
        brain, observer_build_receipt, observer_run_receipt, producer_benchmark, output, threads):
    """两个完整原程序创建前后快照+原 MRI→自产局部场、严格比较及重复。

    creation/likelihood_capture 与各 repeat 是独立 source reference 的
    phase0 mask / phase1 全强度与 logP；只读取原坐标、完整候选边表、
    校准 histogram 和参数作为阶段输入，参考局部数组不进入算法。
    brain 为相同自产归一化 uint8 brain MRI；用 nibabel 头独立生成矩阵。
    build/run receipts 及 producer_benchmark 绑定程序、代码、原始影像。
    output 必须不存在，threads 正数且恢复调用前设置。返回及落盘 JSON、
    私有 candidate NPZ、公开数据衍生 PNG；无生产文件写入、无原程序调用。
    全部已声明字段逐位门在测试前固定；不是生成校准 histogram 或完整 GA。
    格式/阶段/输入版本错误明确抛异常；数值失败保留完成报告，不改变阈值。
    """
    if output.exists():
        raise FileExistsError('fresh isolated output directory required')
    if threads < 1:
        raise ValueError('positive total thread budget required')
    start = time.perf_counter()
    captures = [creation_capture, likelihood_capture, creation_repeat, likelihood_repeat]
    before, after, before_repeat, after_repeat = [load_volume_preparation_capture(capture_path=p) for p in captures]
    if [int(value['header'][7]) for value in [before, after, before_repeat, after_repeat]] != [0, 1, 0, 1]:
        raise ValueError('explicit phase0/phase1 creation/likelihood pairs required')
    shared = ['algorithm_volume_geometry', 'parameters', 'ras_to_voxel', 'original_vertices', 'potential_edges',
              'white_bins', 'white_pdf', 'gray_bins', 'gray_pdf']
    stage_input_consistency = {key: difference(before[key], after[key]) for key in shared}
    if not all(row['bitwise_equal'] for row in stage_input_consistency.values()):
        raise ValueError('pre/post reference input changed: cannot combine different local volume versions')
    image = nib.load(brain)
    brain_before_sha = sha(brain)
    data = np.asarray(image.dataobj)
    if data.dtype != np.uint8 or data.ndim != 3:
        raise ValueError('raw MRI must preserve original uint8 three-dimensional storage')
    matrix = sampler.source_topology_ras_to_voxel(volume_header=image.header)
    matrix_comparison = difference(matrix.astype(np.float64), before['ras_to_voxel'])
    parameters = before['parameters']
    arguments = dict(original_vertices=before['original_vertices'], candidate_edges=before['potential_edges'],
        volume=data, ras_to_voxel=matrix, white_histogram_bins=before['white_bins'],
        white_histogram_counts=before['white_pdf'], white_histogram_bin_size=parameters[1],
        gray_histogram_bins=before['gray_bins'], gray_histogram_counts=before['gray_pdf'],
        gray_histogram_bin_size=parameters[2], volume_resolution=parameters[0])
    input_hashes_before = {key: array_sha(value) for key, value in arguments.items() if isinstance(value, np.ndarray)}
    with thread_budget(threads=threads):
        first, first_seconds, first_components = _evaluate(arguments)
        second, repeat_seconds, repeat_components = _evaluate(arguments)
    input_unchanged = {key: array_sha(value) == input_hashes_before[key] for key, value in arguments.items() if isinstance(value, np.ndarray)}
    comparisons = dict(initial_edge_mask=difference(first['initial_edge_mask'], before['intensity']),
        intensity=difference(first['defect_intensity_volume'], after['intensity']),
        white_log_pdf=difference(first['white_log_probability'], after['white_log_pdf']),
        gray_log_pdf=difference(first['gray_log_probability'], after['gray_log_pdf']),
        algorithm_volume_geometry=difference(first['algorithm_volume_geometry'], before['algorithm_volume_geometry']),
        volume_shape=difference(np.asarray(first['volume_shape'], np.int32), before['header'][2:5]))
    source_statistics = np.asarray([first['white_mean'], first['gray_mean'], first['initial_white_mean_log_probability'],
                                    first['initial_gray_mean_log_probability']], np.float32)
    comparisons['four_calibrated_statistics'] = difference(source_statistics, parameters[3:7])
    candidate_repeat = {key: difference(value, second[key]) for key, value in first.items() if isinstance(value, np.ndarray)}
    for key in ['white_mean', 'gray_mean', 'white_variance', 'gray_variance',
                'initial_white_mean_log_probability', 'initial_gray_mean_log_probability']:
        candidate_repeat[key] = difference(np.asarray(first[key], np.float32), np.asarray(second[key], np.float32))
    native_repeat = {'creation': {key: difference(value, before_repeat[key]) for key, value in before.items()},
                     'likelihood': {key: difference(value, after_repeat[key]) for key, value in after.items()}}
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz', **{key: value for key, value in first.items() if isinstance(value, np.ndarray)},
                        calibrated_statistics=source_statistics)
    from benchmark_recon_topology_volume_prepare_fields import __file__ as capture_parser
    report = dict(status='complete_frozen_topology_volume_priors_regression',
        strict_full_local_fields_passed=bool(matrix_comparison['bitwise_equal'] and all(row['bitwise_equal'] for row in comparisons.values())),
        candidate_repeat_exact=all(row['bitwise_equal'] for row in candidate_repeat.values()),
        native_creation_repeat_exact=all(row['bitwise_equal'] for row in native_repeat['creation'].values()),
        native_likelihood_repeat_exact=all(row['bitwise_equal'] for row in native_repeat['likelihood'].values()),
        brain_sha256_before=brain_before_sha, brain_sha256_after=sha(brain),
        brain_voxels_sha256=array_sha(data), brain_shape=list(data.shape), brain_dtype=str(data.dtype),
        brain_affine=np.asarray(image.affine).tolist(), brain_vox2ras_tkr=np.asarray(image.header.get_vox2ras_tkr()).tolist(),
        raw_mri_reference_local_fields_not_used=True, calibration_histograms_are_frozen_stage_inputs=True,
        source_stage_input_consistency=stage_input_consistency, source_matrix_comparison=matrix_comparison,
        candidate_inputs_not_modified=input_unchanged, field_comparisons=comparisons, candidate_repeat=candidate_repeat,
        native_repeat=native_repeat, candidate_calibrated_statistics=dict(zip(
            ['white_mean','gray_mean','white_mean_log_probability','gray_mean_log_probability'], [float(x) for x in source_statistics])),
        source_calibrated_statistics=parameters[3:7].tolist(),
        candidate_variance_diagnostics_not_captured_by_source=dict(white=first['white_variance'],gray=first['gray_variance']),
        reference_capture_sha256=dict(zip(['creation','likelihood','creation_repeat','likelihood_repeat'], [sha(p) for p in captures])),
        native_build_receipt=json.loads(observer_build_receipt.read_bytes()), native_build_receipt_sha256=sha(observer_build_receipt),
        native_run_receipt=json.loads(observer_run_receipt.read_bytes()), native_run_receipt_sha256=sha(observer_run_receipt),
        producer_benchmark_sha256=sha(producer_benchmark), candidate_npz_sha256=sha(output/'candidate.private.npz'),
        module_sha256=dict(implementation=sha(implementation.__file__), sampler=sha(sampler.__file__),
            interpolation=sha(interpolation.__file__), matrix_inverse=sha(matrix_inverse.__file__),
            parser=sha(capture_parser), benchmark=sha(__file__)),
        input_fields={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_sha(value)) for key,value in before.items()},
        vertex_count=len(before['original_vertices']),potential_edge_count=len(before['potential_edges']),
        volume_shape=list(first['volume_shape']), edge_mask_nonzero=int(np.count_nonzero(first['initial_edge_mask'])),
        volume_resolution=float(parameters[0]), coordinate_space='surface RAS mm; volume index scale mm^-1; MRI and local arrays XYZ',
        cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),python=platform.python_version(),numpy=np.__version__,
        seconds=dict(imports=_IMPORT,first_complete_api_available_jit_cache=first_seconds,repeat_complete_api=repeat_seconds,
            first_components=first_components,repeat_components=repeat_components,
            read_validation_both_apis_npz_and_hash=time.perf_counter()-start),
        native_preparation_component_seconds=None,
        native_component_timing_limitation='observer wall includes earlier topology setup; no isolated preparation timer or whole-GA speedup ratio',
        gpu_used=False,gpu_process_memory_bytes=None,full_ga='not executed',whole_reconstruction='not executed',overall_equivalence='not_assessed')
    tick=time.perf_counter();_plot(first,after,output/'local_priors.png')
    report['seconds']['plot_with_import_and_png_write']=time.perf_counter()-tick
    report['figure_sha256']=sha(output/'local_priors.png')
    report['seconds']['tool_wall_including_imports_plot_excluding_interpreter_bootstrap_final_report_write']=time.perf_counter()-_ENTRY
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['creation-capture','likelihood-capture','creation-repeat','likelihood-repeat','brain',
                 'observer-build-receipt','observer-run-receipt','producer-benchmark','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    run(**vars(p.parse_args()))
