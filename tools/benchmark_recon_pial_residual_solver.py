"""冻结自产 pial 的独立局部残余求解，保存失败并保留全量质量门。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def run(*, frozen_source: Path, overlay_source: Path, diagnostic_case: Path,
        input_hashes: Path, protocol: Path, output_directory: Path,
        distance_module: Path, reference_run: Path | None = None, threads: int = 4):
    """同七输入阶段；RAS/mm、有序网格；完整墙钟含校验/读写/所有候选。

    frozen_source是已冻结自产依赖包src；overlay_source仅本轮实验模块。
    diagnostic_case包含subject七前置和cleanup-result.private.npy；
    input_hashes为七个相对路径/SHA JSON。protocol为执行前固定的三
    算法参数及验收约束JSON。distance_module是现有点到三角面距离工具。
    可选reference_run只能在本轮候选保存后读取两次既有参考，禁止
    根据参考选择候选。新output_directory保存完整report、candidate
    NPY/表面及完成SHA收据。threads正整数默认4。返回complete_diagnosis
    与明确的独立质量状态；残余失败不删除，不能当作生产成功。
    """
    started = time.perf_counter()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("positive integer thread budget required")
    if output_directory.exists(): raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    expected = json.loads(input_hashes.read_text())
    settings = json.loads(protocol.read_text())
    algorithms = {"independent_local_residual_solver_v1", "independent_projected_pair_solver_v1", "independent_free_plane_pair_solver_v1",
                  "independent_free_plane_pair_solver_v2_crossing_guard"}
    if len(expected) != 7 or settings.get("algorithm") not in algorithms:
        raise ValueError("seven input hashes and fixed solver protocol required")
    if settings["algorithm"] == "independent_local_residual_solver_v1":
        required = {"max_displacement_mm": 1., "minimum_area_ratio": .2, "max_passes": 4}
    else:
        required = {"max_displacement_mm": 1., "minimum_area_ratio": .2, "clearance_mm": .002,
                    "max_sweeps": 4, "optimizer_iterations": 100,
                    "diagnostic_pair_limit": settings.get("parameters", {}).get("diagnostic_pair_limit")}
        if settings['algorithm'] in ('independent_free_plane_pair_solver_v1', 'independent_free_plane_pair_solver_v2_crossing_guard'):
            required['projection_mode'] = 'free_separating_plane'
        if settings['algorithm'] == 'independent_free_plane_pair_solver_v2_crossing_guard':
            required['crossing_guard'] = True
        if required['diagnostic_pair_limit'] not in (None, 1) or isinstance(required['diagnostic_pair_limit'], bool):
            raise ValueError('projected protocol permits full processing or exactly one diagnostic pair')
    if settings.get("parameters") != required:
        raise ValueError("this benchmark is bound to its predeclared solver protocol")
    for name, digest in expected.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(diagnostic_case/"subject"/name) != digest:
            raise ValueError("stage input differs or has unsafe relative path")
    sys.path.insert(0, str(frozen_source))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(overlay_source))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    import scipy
    numba.set_num_threads(threads)
    if settings["algorithm"] == "independent_local_residual_solver_v1":
        from fnit.recon_all.place_surface_residual_cleanup import resolve_residual_intersections as resolver
    else:
        from fnit.recon_all.place_surface_residual_projection import resolve_projected_intersections as resolver
    from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    from fnit.recon_all.mris_remove_intersection_python import mark_intersections, _write_preserving_surface
    from fnit.recon_all.compare_subject import _topology
    from fnit.recon_all.topology_volume_python import _nonmanifold_vertex_links
    white_path = diagnostic_case/"subject/surf/lh.white"
    white, faces = fs.read_geometry(str(white_path)); white = white.astype(np.float32); faces = faces.astype(np.int32)
    terminal_path = diagnostic_case/"cleanup-result.private.npy"
    terminal = np.load(terminal_path)
    if terminal.shape != white.shape or terminal.dtype != np.float32 or not np.isfinite(terminal).all():
        raise ValueError("finite float32 terminal ordered on the own white mesh required")
    ripped = np.ones(len(terminal), bool)
    ripped[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex+hipamyg.label")).astype(int)] = False
    cortex = np.zeros(len(terminal), bool)
    cortex[fs.read_label(str(diagnostic_case/"subject/label/lh.cortex.label")).astype(int)] = True
    fixed = ripped | ~cortex
    report = {"schema_version": 1, "execution_status": "running", "scope": "frozen same-input local residual cleanup; not raw-T1 workflow",
              "input_sha256": expected, "terminal_sha256": sha(terminal_path), "protocol_sha256": sha(protocol),
              "protocol": settings, "script_sha256": sha(__file__), "threads": threads,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_memory_bytes": None, "gpu_computation": False, "whole_pipeline_speedup": None,
              "strict_reproduction": "independent_algorithm_not_source_reproduction", "overall_metric_equivalence": "not_assessed",
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__, "nibabel": sys.modules['nibabel'].__version__, "scipy": scipy.__version__},
              "coordinate_space": "surface_RAS", "coordinate_unit": "mm", "original_stage_program_called": False,
              "source_sha256": {name: sha(module.__file__) for name, module in sorted(sys.modules.items())
                                if name.startswith('fnit.recon_all.') and getattr(module, '__file__', None) and str(module.__file__).endswith('.py')},
              "mesh": {"vertices": len(terminal), "faces": len(faces), "fixed_vertices": int(fixed.sum()), "ripped_vertices": int(ripped.sum()),
                       "outside_cortex_elements_different_from_white_at_entry": int(np.count_nonzero(terminal[~cortex] != white[~cortex])),
                       "ripped_elements_different_from_white_at_entry": int(np.count_nonzero(terminal[ripped] != white[ripped]))}}
    write(output_directory/'report.json', report)
    def check(name, xyz):
        tick = time.perf_counter()
        source_marked, source_count = mark_source_intersections(xyz, faces, predicate_backend='numba')
        source_seconds = time.perf_counter()-tick; tick = time.perf_counter()
        independent_marked, independent_count = mark_intersections(xyz, faces)
        area2 = np.linalg.norm(np.cross(xyz.astype(float)[faces[:, 1]]-xyz.astype(float)[faces[:, 0]],
                                       xyz.astype(float)[faces[:, 2]]-xyz.astype(float)[faces[:, 0]]), axis=1)
        row = {"state": name, "source_faces": int(source_count), "source_vertices": int(source_marked.sum()),
               "float64_faces": int(independent_count), "float64_vertices": int(independent_marked.sum()),
               "source_seconds": source_seconds, "float64_and_area_seconds": time.perf_counter()-tick,
               "finite": bool(np.isfinite(xyz).all()), "zero_area_faces": int(np.count_nonzero(area2 == 0)),
               "minimum_area_mm2": float(area2.min()/2), "marked_vertex_sets_equal": bool(np.array_equal(source_marked, independent_marked))}
        return row, source_marked
    initial_qc, marked = check('initial_terminal', terminal)
    report['qc'] = [initial_qc]
    # 固定点原因仅观察源的真实float32加法顺序；不改变任何标签。
    neighbors, valid, _ = _ordered_neighbors(faces, len(terminal))
    residual = []
    for v in np.flatnonzero(marked & ~ripped):
        moving_neighbors = neighbors[v, valid[v] & ~ripped[neighbors[v]]]
        total = terminal[v].copy(); den = np.float32(1)
        for other in moving_neighbors: total = (total+terminal[other]).astype(np.float32); den = np.float32(den+1)
        residual.append(int(np.count_nonzero(total/den != terminal[v])))
    report['fixed_point_diagnosis'] = {"marked_nonrip_vertices": len(residual), "vertices_with_nonzero_source_soap_update": int(np.count_nonzero(residual)),
                                        "source_soap_skips_ripped_neighbors": True}
    report['topology'] = _topology(terminal.astype(float), faces)
    report['nonmanifold_vertex_links'] = int(_nonmanifold_vertex_links(faces, len(terminal)))
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, groups = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    report['edge_orientation_conflicts'] = int(np.count_nonzero(np.bincount(groups, weights=np.where(directed[:, 0]<directed[:, 1], 1, -1))))
    write(output_directory/'report.json', report)
    candidate, diagnostics = resolver(vertices=terminal, faces=faces, fixed_vertices=fixed,
                                      white_vertices=white, **required)
    report['resolver'] = diagnostics
    # 显式crossing_guard延迟加载的实际helper也必须绑定源码；初始导入
    # 清单不能当作整次阶段所用源码的完整清单。
    report['source_sha256'] = {name: sha(module.__file__) for name, module in sorted(sys.modules.items())
                              if name.startswith('fnit.recon_all.') and getattr(module, '__file__', None)
                              and str(module.__file__).endswith('.py')}
    np.save(output_directory/'candidate.private.npy', candidate)
    surface_path = output_directory/'candidate.pial'
    _write_preserving_surface(white_path, surface_path, candidate)
    decoded, decoded_faces = fs.read_geometry(str(surface_path))
    if not np.array_equal(decoded.astype(np.float32), candidate) or not np.array_equal(decoded_faces, faces):
        raise ValueError("written candidate geometry/order differs")
    if not np.array_equal(candidate[fixed], terminal[fixed]): raise ValueError("fixed coordinates changed")
    report['output_sha256'] = {'candidate.private.npy': sha(output_directory/'candidate.private.npy'), 'candidate.pial': sha(surface_path)}
    final_qc, _ = check('candidate', candidate); report['qc'].append(final_qc)
    if final_qc['source_faces'] != diagnostics['source_intersecting_faces_after'] or final_qc['float64_faces'] != diagnostics['float64_intersecting_faces_after']:
        raise ValueError("existing full independent detectors disagree with solver global pair sets")
    if initial_qc['source_faces'] == 0:
        raise ValueError("this residual benchmark requires a nonzero initial case")
    triangle = terminal.astype(float)[faces]
    normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    after = candidate.astype(float)[faces]
    after_normal = np.cross(after[:, 1]-after[:, 0], after[:, 2]-after[:, 0])
    ratio = np.linalg.norm(after_normal, axis=1)/np.linalg.norm(normal, axis=1)
    report['geometry_guards'] = {"fixed_different_elements": int(np.count_nonzero(candidate[fixed] != terminal[fixed])),
                                 "ordered_faces_equal": True, "nonpositive_initial_normal_dot_faces": int(np.count_nonzero(np.sum(normal*after_normal, axis=1)<=0)),
                                 "minimum_area_ratio": float(ratio.min()),
                                 "white_pial_crossing": diagnostics.get('white_pial_guard', 'not_assessed')}
    displacement = np.linalg.norm(candidate.astype(float)-terminal, axis=1)
    changed = displacement[displacement != 0]
    report['ordered_displacement_mm'] = {"different_elements": int(np.count_nonzero(candidate != terminal)), "changed_vertices": len(changed),
                                         "mean": float(displacement.mean()), "p99": float(np.quantile(displacement, .99)), "max": float(displacement.max()),
                                         "changed_vertex_p99": float(np.quantile(changed, .99)) if len(changed) else 0.}
    spec = importlib.util.spec_from_file_location('pial_residual_triangle_distances', distance_module)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tick = time.perf_counter()
    distances = {}
    for name, query, target in (('candidate_to_entry', candidate, terminal), ('entry_to_candidate', terminal, candidate)):
        value = module._point_to_mesh(query, target, faces)
        distances[name] = {'mean_mm': float(value.mean()), 'p99_mm': float(np.quantile(value, .99)), 'max_mm': float(value.max())}
    report['bidirectional_vertex_to_triangle'] = distances
    report['surface_distance_seconds'] = time.perf_counter()-tick; report['surface_distance_module_sha256'] = sha(distance_module)
    # 原参考坐标严格延迟到独立候选、完整QC和全部候选收据写完后。
    write(output_directory/'report.json', report)
    if reference_run is not None:
        report['reused_reference_diagnostics'] = []
        for repeat in (1, 2):
            subject = reference_run/f'reference-{repeat}/subject'
            if any(sha(subject/name) != digest for name, digest in expected.items()): raise ValueError('reference seven input hashes differ')
            path = subject/'surf/lh.pial.T1'; reference, f = fs.read_geometry(str(path)); reference = reference.astype(np.float32)
            if not np.array_equal(f, faces) or reference.shape != candidate.shape: raise ValueError('reference ordered correspondence differs')
            d = np.linalg.norm(candidate.astype(float)-reference, axis=1)
            report['reused_reference_diagnostics'].append({'repeat': repeat, 'file_sha256': sha(path),
                'entry_elements_differ_from_reference': int(np.count_nonzero(terminal != reference)),
                'candidate_elements_differ_from_reference': int(np.count_nonzero(candidate != reference)),
                'candidate_mean_mm': float(d.mean()), 'candidate_p99_mm': float(np.quantile(d, .99)), 'candidate_max_mm': float(d.max())})
    report['execution_status'] = 'complete_diagnosis'
    report['residual_self_intersection_quality'] = 'passed' if final_qc['source_faces'] == final_qc['float64_faces'] == 0 else 'failed'
    report['quality_scope'] = ('self-intersection and unchanged mesh; no new white/pial crossings/contact; initial and final crossing counts separate; whole not assessed'
                               if diagnostics.get('white_pial_guard') else 'self-intersection and unchanged mesh structure only; whole/white-pial crossing not assessed')
    report['peak_process_cpu_memory_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    report['complete_wall_seconds_before_report'] = time.perf_counter()-started
    write(output_directory/'report.json', report)
    write(output_directory/'COMPLETE.json', {'execution_status': report['execution_status'], 'quality': report['residual_self_intersection_quality'],
        'report_sha256': sha(output_directory/'report.json'), 'output_sha256': report['output_sha256'], 'complete_wall_seconds': time.perf_counter()-started})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source', 'overlay-source', 'diagnostic-case', 'input-hashes', 'protocol', 'output-directory', 'distance-module'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reference-run', type=Path); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    report = run(**{key: value for key, value in vars(a).items()})
    print(json.dumps({'execution_status': report['execution_status'], 'quality': report['residual_self_intersection_quality']}))


if __name__ == '__main__': main()
