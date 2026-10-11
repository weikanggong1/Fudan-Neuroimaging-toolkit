"""冻结真实网格的MRI原语默认兼容与独立碰撞候选回归，不是整例。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_normals import initial_vertex_normals
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_defect_mri_match import defect_mri_match
from fnit.recon_all.topology_tissue_patch_python import _sample_sides, _score, run_tissue_patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, '<f4').tobytes()).hexdigest()


def compare(first, second):
    displacement = np.linalg.norm(first.astype(np.float64) - second, axis=1)
    return {'different_coordinates': int(np.count_nonzero(first != second)),
            'different_vertices': int(np.count_nonzero(displacement)),
            'mean_mm': float(displacement.mean()), 'p99_mm': float(np.percentile(displacement, 99)),
            'maximum_mm': float(displacement.max(initial=0)),
            'first_sha256': array_sha(first), 'second_sha256': array_sha(second)}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-surface', type=Path, required=True)
    parser.add_argument('--intensity-volume', type=Path, required=True)
    parser.add_argument('--reference-python', type=Path, required=True,
                        help='冻结旧自有 topology_defect_mri_match.py；不使用原版阶段程序')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--log-shape-prior', type=float, default=0.0)
    parser.add_argument('--normal-offset-mm', type=float, default=0.5)
    parser.add_argument('--step-motion-limit-mm', type=float, default=0.3)
    parser.add_argument('--compatibility-only', action='store_true',
                        help='仅完成默认40轮真实兼容；不启动独立慢候选')
    parser.add_argument('--collision-acceptance',choices=('ordered','ordered-active'),default='ordered')
    parser.add_argument('--probe-query-cache',action='store_true',help='先做完整query/active-query同一步ABBA')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('fresh output directory required')
    args.output_dir.mkdir(parents=True)
    specification = importlib.util.spec_from_file_location('fnit_frozen_mri_match_reference', args.reference_python)
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)
    points, faces, metadata = fsio.read_geometry(str(args.input_surface), read_metadata=True)
    points, faces = np.ascontiguousarray(points, np.float32), np.ascontiguousarray(faces, np.int32)
    image = nib.load(str(args.intensity_volume))
    volume = np.asarray(image.dataobj)
    matrix = np.asarray(surface_ras_to_voxel(image.header, metadata), np.float64)
    neighbors = ordered_neighbors(faces, len(points))
    normals = initial_vertex_normals(vertices=points, triangles=faces)
    white, gray = _sample_sides(points, normals, volume, matrix, np.float32(args.normal_offset_mm))
    score, histograms = _score(white, gray, neighbors, args.log_shape_prior)
    inside = np.flatnonzero(score > 0).astype(np.int32)
    parameters = dict(xyz=points, faces=faces, inside_vertices=inside,
                      runtime_neighbors=[neighbors[int(vertex)] for vertex in inside],
                      volume=volume, ras_to_voxel=matrix,
                      white_mean=float(histograms['white_mean']), gray_mean=float(histograms['gray_mean']))
    runs = []
    arrays = []
    identity_trace = []
    for label, operation in (
            ('frozen_original_40', lambda: reference.defect_mri_match(**parameters)),
            ('current_default_40', lambda: defect_mri_match(**parameters)),
            ('current_identity_acceptor_40', lambda: defect_mri_match(
                **parameters, step_acceptor=lambda current, proposed, iteration:
                (identity_trace.append({'iteration': iteration, 'sha256': array_sha(proposed)}) or proposed))),
            ('frozen_original_repeat_40', lambda: reference.defect_mri_match(**parameters))):
        tick = time.perf_counter()
        values = operation()
        arrays.append(values)
        runs.append({'label': label, 'seconds_including_first_JIT': time.perf_counter() - tick,
                     'sha256': array_sha(values)})
    default_comparisons = {'old_vs_current_default': compare(arrays[0], arrays[1]),
                           'default_vs_identity_acceptor': compare(arrays[1], arrays[2]),
                           'old_repeat': compare(arrays[0], arrays[3])}
    report = {'scope': 'frozen real input local primitive compatibility and independent candidate',
              'input_sha256': {'surface': sha(args.input_surface), 'MRI': sha(args.intensity_volume)},
              'benchmark_source_sha256': sha(__file__), 'reference_python_sha256': sha(args.reference_python),
              'current_mri_match_sha256':sha(defect_mri_match.__globals__['__file__']),
              'current_tissue_patch_sha256':sha(run_tissue_patch.__globals__['__file__']),
              'reference_kind': 'frozen FNIT own Python primitive, not complete official GA',
              'selected_vertices': len(inside), 'identity_step_trace': identity_trace,
              'default_runs': runs, 'default_comparisons': default_comparisons,
              'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')},
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'native_program_called': False, 'complete_dynamic_topology_solver': False,
              'overall_metric_equivalence': 'not_assessed'}
    if any(comparison['different_coordinates'] for comparison in default_comparisons.values()):
        report.update(execution_status='failed_default_compatibility', total_seconds=time.perf_counter()-started)
        (args.output_dir/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')
        raise RuntimeError('default MRI primitive changed: candidate not run')
    # 长时间候选前保留已完成的兼容收据；候选未完成不能改称全任务完成。
    (args.output_dir/'default_compatibility.private.json').write_text(
        json.dumps({**report, 'execution_status':'complete_default_compatibility_only'}, indent=2)+'\n')
    if args.compatibility_only:
        report.update(execution_status='complete_default_compatibility_only',
                      total_seconds_including_IO_JIT_quality=time.perf_counter()-started,
                      candidate_started=False)
        (args.output_dir/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')
        return
    if args.probe_query_cache:
        from fnit.recon_all.topology_collision_cache_python import FixedPatchCollisionCache
        from fnit.recon_all.place_surface_collision import asynchronous_first_step
        mask=np.zeros(len(points),bool);mask[inside]=True
        cache=FixedPatchCollisionCache(faces=faces,selected=mask)
        proposal=defect_mri_match(**parameters,iterations=1)
        delta=proposal.astype(np.float64)-points
        distances=np.linalg.norm(delta,axis=1);clip=distances>args.step_motion_limit_mm
        delta[clip]*=(args.step_motion_limit_mm/distances[clip])[:,None]
        proposal=np.float32(points+delta)
        probe_runs=[];probe_values=[];probe_orders=[]
        for label in ('full','active','active','full'):
            tick=time.perf_counter();diagnostics={}
            if label=='full':
                values,order=asynchronous_first_step(vertices=points,faces=faces,proposed=proposal,
                    ripped=~mask,candidate_backend='snapshot',candidate_diagnostics=diagnostics)
            else:
                values,order,diagnostics=cache.step(current=points,proposal=proposal,faces=faces)
            probe_runs.append({'label':label,'seconds_including_first_JIT':time.perf_counter()-tick,
                               'diagnostics':diagnostics,'coordinates_sha256':array_sha(values)})
            probe_values.append(values);probe_orders.append(order)
        cache_probes={'runs':probe_runs,'pair_comparisons':[compare(probe_values[0],values) for values in probe_values[1:]],
                      'all_order_equal':all(np.array_equal(probe_orders[0],order) for order in probe_orders[1:]),
                      'scope':'one frozen real MRI proposal complete ordered step; not 40-step or E2E',
                      'source_sha256':sha(FixedPatchCollisionCache.step.__globals__['__file__'])}
        report['query_cache_probes']=cache_probes
        (args.output_dir/'query_cache_probes.private.json').write_text(json.dumps(cache_probes,indent=2)+'\n')
        if not cache_probes['all_order_equal'] or any(c['different_coordinates'] for c in cache_probes['pair_comparisons']):
            report.update(execution_status='failed_query_cache_step')
            (args.output_dir/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
            raise RuntimeError('active query cache changed full ordered step')
    try:
        candidate = run_tissue_patch(input_surface=args.input_surface, intensity_volume=args.intensity_volume,
            output_surface=args.output_dir/'tissue-patch.surface', log_shape_prior=args.log_shape_prior,
            normal_offset_mm=args.normal_offset_mm, require_quality=True,
            report_path=args.output_dir/'candidate.private.json', collision_acceptance=args.collision_acceptance,
            step_motion_limit_mm=args.step_motion_limit_mm)
        values, candidate_faces, candidate_metadata = fsio.read_geometry(str(args.output_dir/'tissue-patch.surface'),read_metadata=True)
        report.update(execution_status='complete_primitive_pair', candidate=candidate,
                      candidate_vs_input=compare(points, np.float32(values)),
                      ordered_faces_equal=bool(np.array_equal(faces, candidate_faces)),
                      geometry_header_equal=all(np.array_equal(metadata[key], candidate_metadata[key]) for key in metadata))
    except Exception as error:
        report.update(execution_status='failed_candidate', error_type=type(error).__name__, error=str(error))
        candidate_path = args.output_dir/'candidate.private.json'
        if candidate_path.exists(): report['candidate'] = json.loads(candidate_path.read_text())
        raise
    finally:
        report['total_seconds_including_IO_JIT_quality'] = time.perf_counter()-started
        (args.output_dir/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
