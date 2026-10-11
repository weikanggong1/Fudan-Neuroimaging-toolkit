"""自产 nofix/MRI→白灰校准→完整局部 prior 的同输入真实诊断。"""
from __future__ import annotations
import time
_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
from benchmark_recon_topology_volume_priors import difference, _plot
from fnit.recon_all.topology_defect_calibration import calibrated_defect_volume_priors
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel, source_mri_edge_scores
from fnit.recon_all.topology_preflight_python import (center_sphere, project_and_smooth_sphere,
    defect_component_labels, defect_regions, defect_retention_status, genetic_base_translation,
    genetic_candidate_edge_table)
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.thread_budget import thread_budget
_IMPORT = time.perf_counter() - _ENTRY


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run(*, input_surface, input_qsphere, brain, creation_capture, likelihood_capture,
        creation_repeat, likelihood_repeat, producer_benchmark, observer_build_receipt,
        observer_run_receipt, source_manifest, output, threads):
    """读取自产全脑MRI与nofix/qsphere，独立生成校准PDF后再读取参考局部场。

    全部路径为具名参数；参考捕获只在保存候选后打开，不作为采样、直方图
    或局部体积输入。源shape/数组/dtype/零符号严格比较另列；候选/原程序
    重复和源码/原始影像绑定完整保留。output 必须不存在，threads≥1。
    返回及落盘报告、私有NPZ与公开ds000114衍生PNG；不写生产目录。
    仅覆盖首缺陷校准和priors；不是完整GA、原始T1整例或OS隔离验收。
    """
    if output.exists(): raise FileExistsError('fresh isolated output required')
    if threads < 1: raise ValueError('positive thread budget required')
    started = time.perf_counter()
    raw, faces = fsio.read_geometry(str(input_surface))
    sphere, sphere_faces = fsio.read_geometry(str(input_qsphere))
    raw, faces = np.asarray(raw, np.float32), np.asarray(faces, np.int32)
    if raw.shape != sphere.shape or not np.array_equal(faces, sphere_faces):
        raise ValueError('same ordered nofix and qsphere meshes required')
    image = nib.load(brain); volume = np.asarray(image.dataobj)
    if volume.dtype != np.uint8 or volume.ndim != 3: raise ValueError('unchanged XYZ uint8 MRI required')
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    with thread_budget(threads=threads):
        canonical, centering = center_sphere(project_and_smooth_sphere(sphere, faces))
        labels = defect_component_labels(canonical, faces)
        regions = defect_regions(labels, faces)
        if not regions: raise ValueError('actual first defect required')
        status = defect_retention_status(canonical, raw, faces, labels)
        translation, _ = genetic_base_translation(labels, faces)
        table, unpruned = genetic_candidate_edge_table(canonical, faces, labels, status, defect_number=0)
        inside, border = regions[0]
        neighbors = ordered_neighbors(faces, len(raw))
        # Same SMALL_CONVEX_HULL rule already used by the source preflight.
        occupied = set(inside) | set(border); hull = list(border)
        for vertex in border:
            for neighbor in neighbors[vertex]:
                if neighbor not in occupied: occupied.add(neighbor); hull.append(neighbor)
        preparation_inputs_seconds = time.perf_counter() - started
        arguments = dict(vertices=raw, faces=faces, defect_vertices=labels != 0,
            defect_hull=np.asarray(hull, np.int32), candidate_edges=table[:, :2],
            vertex_translation=translation, volume=volume, ras_to_voxel=matrix)
        first = calibrated_defect_volume_priors(**arguments)
        second = calibrated_defect_volume_priors(**arguments)
        # The complete EDGE table is sorted solely from independent score data.
        prep = first['preparation']; reverse = np.argsort(translation)
        scores = source_mri_edge_scores(original_vertices=prep['original_vertices'],
            original_normals=prep['original_normals'], edge_vertices=reverse[table[:, :2]],
            white_values=prep['white_values'], gray_values=prep['gray_values'],
            original_edge_present=table[:, 2] != 0, volume=volume, ras_to_voxel=matrix)
        order = np.lexsort((np.arange(len(table)), table[:, 0], scores))
        sorted_edges = table[order, :2]
    calibration = first['calibration']; priors = first['priors']
    output.mkdir(parents=True)
    artifacts = dict(**{key: value for key, value in priors.items() if isinstance(value, np.ndarray)},
        **{'calibration_'+key: value for key, value in calibration.items() if isinstance(value, np.ndarray)},
        white_values=prep['white_values'], gray_values=prep['gray_values'],
        smoothed_original=prep['original_vertices'], corrected_original=prep['original_vertices'][reverse],
        translation=translation, hull=np.asarray(hull, np.int32), marked=labels != 0,
        sorted_edges=sorted_edges, canonical=canonical, labels=labels)
    np.savez_compressed(output/'candidate.private.npz', **artifacts)
    candidate_wall_before_reference = time.perf_counter() - started
    # Independent reference local arrays and calibrated histograms are only
    # now read for diagnostics; candidate inputs did not consume any of them.
    captures = [creation_capture, likelihood_capture, creation_repeat, likelihood_repeat]
    before, after, before2, after2 = [load_volume_preparation_capture(capture_path=p) for p in captures]
    if [int(x['header'][7]) for x in [before, after, before2, after2]] != [0, 1, 0, 1]:
        raise ValueError('actual phase0/phase1 pairs required')
    comparisons = dict(
        white_calibrated_pdf=difference(calibration['white_smooth'], before['white_pdf']),
        gray_calibrated_pdf=difference(calibration['gray_smooth'], before['gray_pdf']),
        white_bins=difference(calibration['bins'], before['white_bins']),
        gray_bins=difference(calibration['bins'], before['gray_bins']),
        two_bin_size=difference(np.zeros(2, np.float32), before['parameters'][1:3]),
        corrected_original=difference(artifacts['corrected_original'], before['original_vertices']),
        full_ordered_candidate_edges=difference(sorted_edges, before['potential_edges']),
        source_matrix=difference(matrix.astype(np.float64), before['ras_to_voxel']),
        initial_edge_mask=difference(priors['initial_edge_mask'], before['intensity']),
        intensity=difference(priors['defect_intensity_volume'], after['intensity']),
        white_log_pdf=difference(priors['white_log_probability'], after['white_log_pdf']),
        gray_log_pdf=difference(priors['gray_log_probability'], after['gray_log_pdf']),
        algorithm_volume_geometry=difference(priors['algorithm_volume_geometry'], before['algorithm_volume_geometry']),
        volume_shape=difference(np.asarray(priors['volume_shape'], np.int32), before['header'][2:5]),
        four_calibrated_statistics=difference(np.asarray([priors['white_mean'], priors['gray_mean'],
            priors['initial_white_mean_log_probability'],priors['initial_gray_mean_log_probability']], np.float32), before['parameters'][3:7]))
    repeat = {group: {key: difference(value, second[group][key]) for key,value in first[group].items()
                     if isinstance(value, np.ndarray)} for group in ['calibration','priors']}
    for key in ['white_values','gray_values','original_vertices','original_normals']:
        repeat.setdefault('preparation',{})[key] = difference(prep[key], second['preparation'][key])
    native_repeat = dict(creation={key:difference(value,before2[key]) for key,value in before.items()},
                         likelihood={key:difference(value,after2[key]) for key,value in after.items()})
    _plot(priors, after, output/'local_fields.png')
    _plot_histogram(calibration, before, output/'calibrated_histograms.png')
    report = dict(status='complete_frozen_nofix_mri_calibration_priors_regression',
        raw_calibration_histograms_consumed=False, reference_local_fields_consumed=False,
        independently_generated_defect_labels_hull_translation_edges=True,
        strict_full_internal_fields_passed=all(x['bitwise_equal'] for x in comparisons.values()),
        computed_histograms_and_local_fields_passed=all(x['bitwise_equal'] for key,x in comparisons.items() if key!='source_matrix'),
        comparisons=comparisons, candidate_repeat=repeat,
        candidate_repeat_exact=all(x['bitwise_equal'] for group in repeat.values() for x in group.values()),
        native_repeat=native_repeat, native_repeat_exact=all(x['bitwise_equal'] for group in native_repeat.values() for x in group.values()),
        source_phase_inputs_equal={key:difference(before[key],after[key]) for key in ['original_vertices','potential_edges','white_pdf','gray_pdf','parameters']},
        selected_vertices=len(calibration['selected']), defect_hull_vertices=len(hull),
        first_defect_vertices=len(inside), first_defect_border=len(border), all_defects=len(regions),
        nvertices=len(raw), nfaces=len(faces), full_candidate_edges=len(table), before_pruning_edges=unpruned,
        candidate_input_sha256={key:sha(path) for key,path in dict(orig_nofix=input_surface,qsphere_nofix=input_qsphere,brain=brain).items()},
        reference_capture_sha256={key:sha(path) for key,path in dict(creation=creation_capture,likelihood=likelihood_capture,
            creation_repeat=creation_repeat,likelihood_repeat=likelihood_repeat).items()},
        producer_benchmark_sha256=sha(producer_benchmark), observer_build_receipt_sha256=sha(observer_build_receipt),
        observer_run_receipt_sha256=sha(observer_run_receipt), source_manifest_sha256=sha(source_manifest),
        source_manifest=json.loads(source_manifest.read_text()), script_sha256=sha(__file__),
        actual_imported_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in [
            'topology_defect_calibration','topology_intensity_histogram','topology_edge_mri_score',
            'topology_defect_volume_priors','topology_preflight_python','smooth_surface_python',
            'place_surface_normals','place_surface_repulsion','place_surface_border','mri_em_register','thread_budget']},
        candidate_checkpoint_sha256=sha(output/'candidate.private.npz'),
        candidate_array_sha256={key:array_sha(value) for key,value in artifacts.items()},
        calibration_means={key:float(calibration[key]) for key in ['white_mean','gray_mean']},
        final_calibrated_statistics={key:priors[key] for key in ['white_mean','gray_mean','white_variance','gray_variance','initial_white_mean_log_probability','initial_gray_mean_log_probability']},
        timing=dict(import_seconds=_IMPORT, upstream_nofix_state_seconds=preparation_inputs_seconds,
            first_api=first['stage_seconds'], repeat_api=second['stage_seconds'],
            candidate_wall_before_reference_seconds=candidate_wall_before_reference,
            complete_tool_seconds_excluding_initial_import_final_json=time.perf_counter()-started),
        host=platform.node(), affinity=sorted(os.sched_getaffinity(0)), threads=threads,
        gpu_used=False, gpu_memory_bytes=None, gpu_memory_measured=False, shared_host=True,
        thread_environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
        production_changed=False, complete_ga=False, full_pipeline_run=False,
        clean_environment_isolation='not_validated', overall_metric_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def _plot_histogram(candidate, reference, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11,6))
    for row, tissue in enumerate(['white','gray']):
        own, native = candidate[tissue+'_smooth'], reference[tissue+'_pdf']
        axes[row,0].plot(candidate['bins'],own,label='FNIT independently calibrated')
        axes[row,0].plot(reference[tissue+'_bins'],native,'--',label='Pinned source')
        axes[row,0].set_title(tissue+' calibrated PDF');axes[row,0].legend()
        axes[row,1].plot(candidate['bins'],own.astype(np.float64)-native.astype(np.float64))
        axes[row,1].set_title('FNIT minus source');axes[row,1].set_ylabel('probability difference')
        for col in range(2):axes[row,col].set_xlabel('normalized MRI intensity bin')
    fig.suptitle('Public ds000114 sub-07 LH: MRI sampling and calibration to local priors')
    fig.tight_layout();fig.savefig(output,dpi=145);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','brain','creation-capture','likelihood-capture',
                 'creation-repeat','likelihood-repeat','producer-benchmark','observer-build-receipt',
                 'observer-run-receipt','source-manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    result=run(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],first_api=result['timing']['first_api'],
        complete_fields=result['computed_histograms_and_local_fields_passed']),allow_nan=False))


if __name__=='__main__':main()
