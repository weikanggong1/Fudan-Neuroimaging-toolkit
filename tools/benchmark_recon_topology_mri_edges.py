"""真实冻结首缺陷 MRI 边评分；参考表仅在候选计算和保存完成后读入。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import time
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.topology_edge_mri_score import (prepare_topology_edge_inputs,
    source_mri_edge_scores, source_topology_ras_to_voxel)
from fnit.recon_all.topology_preflight_python import (center_sphere, project_and_smooth_sphere,
    defect_component_labels, defect_retention_status, genetic_candidate_edge_table,
    genetic_base_translation, defect_regions)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def errors(candidate, reference):
    difference = np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64))
    return {'different_elements': int(np.count_nonzero(difference)),
        'maximum_absolute_error': float(difference.max(initial=0)),
        'p99_absolute_error': float(np.percentile(difference, 99)) if difference.size else 0.,
        'mean_absolute_error': float(difference.mean()) if difference.size else 0.}


def run(*, input_surface, input_qsphere, intensity_volume, output_dir,
        native_before, native_after, source_root):
    started = time.perf_counter()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    xyz, faces, metadata = fsio.read_geometry(str(input_surface), read_metadata=True)
    sphere, sphere_faces = fsio.read_geometry(str(input_qsphere))
    if not np.array_equal(faces, sphere_faces) or xyz.shape != sphere.shape or not metadata:
        raise ValueError('same ordered mesh and complete MRI geometry required')
    image = nib.load(str(intensity_volume)); volume = np.ascontiguousarray(np.asarray(image.dataobj))
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    canonical, iterations = center_sphere(project_and_smooth_sphere(sphere, faces))
    labels = defect_component_labels(canonical, faces)
    status = defect_retention_status(canonical, np.asarray(xyz, np.float32), faces, labels)
    translation, _ = genetic_base_translation(labels, faces)
    reverse = np.argsort(translation)
    table, all_count = genetic_candidate_edge_table(canonical, faces, labels, status, defect_number=0)
    prepared = prepare_topology_edge_inputs(vertices=xyz, faces=faces,
        defect_vertices=labels != 0, volume=volume, ras_to_voxel=matrix)
    source_edges = reverse[table[:, :2]]
    prepared_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    scores = source_mri_edge_scores(original_vertices=prepared['original_vertices'],
        original_normals=prepared['original_normals'], edge_vertices=source_edges,
        white_values=prepared['white_values'], gray_values=prepared['gray_values'],
        original_edge_present=table[:, 2] != 0, volume=volume, ras_to_voxel=matrix)
    score_cold_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    repeated = source_mri_edge_scores(original_vertices=prepared['original_vertices'],
        original_normals=prepared['original_normals'], edge_vertices=source_edges,
        white_values=prepared['white_values'], gray_values=prepared['gray_values'],
        original_edge_present=table[:, 2] != 0, volume=volume, ras_to_voxel=matrix)
    score_warm_seconds = time.perf_counter() - tick
    independent_seconds = time.perf_counter() - started
    artifacts = {'candidate_edges': table, 'candidate_scores': scores,
        'source_edges': source_edges, 'translation': translation, 'canonical': canonical,
        'faces': faces, 'labels': labels, 'status': status, 'matrix': matrix,
        'smoothed_original': prepared['original_vertices'],
        'original_normals': prepared['original_normals'],
        'white_values': prepared['white_values'], 'gray_values': prepared['gray_values']}
    # 已计算并保存全部候选状态后才打开独立参考，杜绝借参考修补候选。
    np.savez_compressed(out / 'candidate_checkpoint.npz', **artifacts)
    reference_dtype = np.dtype([('a','<i4'),('b','<i4'),('score','<f4'),('used','<i2'),('padding','<i2')])
    reference = np.fromfile(native_before, dtype=reference_dtype)
    after = np.fromfile(native_after, dtype=reference_dtype)
    native_table = np.column_stack((reference['a'], reference['b'], reference['used']))
    table_equal = np.array_equal(table, native_table)
    comparison = errors(scores, reference['score']) if table_equal else None
    predicted_order = np.lexsort((np.arange(len(table)), table[:, 0], scores))
    reference_order = np.column_stack((after['a'], after['b']))
    order_equal = np.array_equal(table[predicted_order, :2], reference_order)
    first = np.flatnonzero(scores != reference['score']) if table_equal else np.empty(0, np.int64)
    report = {'scope': 'complete first-defect MRI EDGE.len sampling, stops before full fitness and GA',
        'complete_ga_solver': False, 'full_pipeline_run': False, 'production_changed': False,
        'overall_metric_equivalence': 'not_assessed', 'native_program_called_by_candidate': False,
        'input_sha256': {'orig_nofix': sha(input_surface), 'qsphere_nofix': sha(input_qsphere),
                        'brain': sha(intensity_volume)},
        'reference_capture_sha256': {'before': sha(native_before), 'after': sha(native_after)},
        'source_sha256': {str(p.relative_to(source_root)): sha(p) for p in [
            Path(source_root)/'src/fnit/recon_all/topology_edge_mri_score.py',
            Path(source_root)/'src/fnit/recon_all/place_surface_normals.py',
            Path(source_root)/'src/fnit/recon_all/place_surface_repulsion.py',
            Path(source_root)/'src/fnit/recon_all/smooth_surface_python.py',
            Path(source_root)/'src/fnit/recon_all/topology_defect_mri_match.py',
            Path(source_root)/'src/fnit/recon_all/place_surface_border.py',
            Path(source_root)/'src/fnit/recon_all/mri_em_register.py',
            Path(source_root)/'src/fnit/recon_all/topology_preflight_python.py']},
        'script_sha256': sha(__file__), 'candidate_checkpoint_sha256': sha(out/'candidate_checkpoint.npz'),
        'array_sha256': {name: array_sha(value) for name, value in artifacts.items()},
        'defect_components': len(defect_regions(labels, faces)),
        'first_patch_inside': len(defect_regions(labels, faces)[0][0]),
        'first_patch_border': len(defect_regions(labels, faces)[0][1]),
        'candidate_edges_before_prune': all_count, 'candidate_edges': len(table),
        'reference_edges': len(reference), 'complete_table_with_flags_equal': bool(table_equal),
        'score_errors_same_table': comparison, 'repeat_scores': errors(scores, repeated),
        'sort_scope': 'stable score/vno1/source-index order; original qsort tie implementation remains separately diagnosed',
        'predicted_sorted_endpoint_order_equal': bool(order_equal),
        'first_score_difference_index': int(first[0]) if len(first) else None,
        'first_score_difference': {'source_edge': source_edges[first[0]].tolist(),
            'candidate': float(scores[first[0]]), 'reference': float(reference['score'][first[0]])} if len(first) else None,
        'preparation_IO_JIT_seconds': prepared_seconds, 'score_first_JIT_seconds': score_cold_seconds,
        'score_warm_seconds': score_warm_seconds, 'candidate_wall_before_reference_seconds': independent_seconds,
        'complete_harness_seconds_with_reference_comparison': time.perf_counter() - started,
        'host': platform.node(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'thread_environment': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS','NUMBA_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
        'gpu_used': False, 'gpu_memory_bytes': 0,
        'remaining_GA_components': ['full patch MRI/unMRI/normal/curvature fitness',
            'USE_SOME classification/retry', 'mutation', 'crossover', 'population controller',
            'all-defect sequencing and global geometry acceptance']}
    (out/'report.private.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','intensity-volume','output-dir','native-before','native-after','source-root']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
