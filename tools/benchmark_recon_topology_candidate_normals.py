"""真实冻结候选的独立面法向回归及完整有向距离传播；不执行GA。"""
from __future__ import annotations
import time
_START = time.perf_counter()
import argparse, hashlib, json, os, platform
from pathlib import Path
import numpy as np
from benchmark_recon_topology_distance_fields import load_distance_capture
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from fnit.recon_all import topology_candidate_normals as implementation
from fnit.recon_all.topology_defect_signed_distance import defect_signed_distance
from fnit.recon_all.thread_budget import thread_budget
_IMPORT = time.perf_counter() - _START


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def difference(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape: return dict(shape_equal=False, bitwise_equal=False)
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(shape_equal=True, different_elements=int(np.count_nonzero(a != b)),
        different_bytes=int(np.count_nonzero(np.frombuffer(a.tobytes(), np.uint8) != np.frombuffer(b.tobytes(), np.uint8))) if a.dtype == b.dtype else None,
        bitwise_equal=bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
        max_abs=float(error.max(initial=0)), p99_abs=float(np.percentile(error, 99)) if error.size else 0.)


def run(*, capture, native_repeat, patch_capture, observer_receipt, output, threads):
    """完整同输入FNITDST2/FNITFIT1→自产触及面法向、完整距离及收据。

    capture/native_repeat 为独立参考进程的当前候选几何/CSR/法向/距离；
    patch_capture 是同候选99个patch顶点，所有坐标和有序面须先匹配。
    observer_receipt 绑定参考程序/输入；output 必须新目录，threads正数。
    按patch原incident顺序选择非ripped面，每面一次；只将这些自产法向
    替换到冻结的其余输入缓存，完整调用已有自产有向距离。参考距离
    不进入求解。未触及缓存仍是冻结阶段输入，不宣称整份缓存自产。
    缓存orig_area当前参考未捕获，只记录输出SHA，不宣称真实area通过；
    顶点法向未在评分caller重算，完全不调用candidate_vertex_normals。
    写出私有candidate.npz、标准JSON和候选原RAS误差PNG。无生产修改，
    不调用原版程序、不改全局精度。格式/版本/预算错误抛异常。
    """
    started = time.perf_counter()
    if output.exists(): raise FileExistsError('new isolated diagnostic output required')
    if not isinstance(threads, int) or threads < 1: raise ValueError('positive thread budget required')
    f = load_distance_capture(capture_path=capture)
    repeat = load_distance_capture(capture_path=native_repeat)
    patch = load_statistics_capture(capture_path=patch_capture)
    if 'algorithm_volume_geometry' not in f: raise ValueError('actual FNITDST2 algorithm geometry required')
    if not np.array_equal(f['faces'], patch['faces']) or not np.array_equal(f['vertex_fields'][:, :3], patch['vertex_fields'][:, :3]):
        raise ValueError('source patch and distance captures must share complete candidate geometry')
    selected, seen = [], set()
    for vertex in patch['selected_vertices']:
        for slot in range(f['incident_offsets'][vertex], f['incident_offsets'][vertex + 1]):
            face = int(f['incident_faces'][slot])
            if not f['face_rip'][face] and face not in seen:
                selected.append(face); seen.add(face)
    selected = np.asarray(selected, np.int64)
    args = dict(original_vertices=f['vertex_fields'][:, :3], faces=f['faces'], selected_faces=selected)
    with thread_budget(threads=threads):
        tick = time.perf_counter(); candidate = implementation.candidate_face_normals(**args); first = time.perf_counter() - tick
        tick = time.perf_counter(); second = implementation.candidate_face_normals(**args); warm = time.perf_counter() - tick
        normals = f['face_normals'].copy(); normals[selected] = candidate['face_normals']
        geometry = f['algorithm_volume_geometry']
        tick = time.perf_counter()
        distance = defect_signed_distance(original_vertices=f['vertex_fields'][:, :3], faces=f['faces'],
            face_normals=normals, incident_offsets=f['incident_offsets'], incident_faces=f['incident_faces'],
            incident_corners=f['incident_corners'], volume_shape=tuple(map(int, f['algorithm_volume_dimensions'][:3])),
            volume_origin_mm=geometry[:3], volume_index_scale_per_mm=geometry[3:6])
        distance_seconds = time.perf_counter() - tick
    output.mkdir(parents=True)
    np.savez_compressed(output / 'candidate.private.npz', selected_faces=selected, **candidate, distance_mm=distance['distance_mm'])
    comparisons = dict(updated_face_normals=difference(candidate['face_normals'], f['face_normals'][selected]),
        propagated_complete_signed_distance=difference(distance['distance_mm'], f['distance_after']))
    native = dict(face_normals=difference(f['face_normals'], repeat['face_normals']),
        complete_signed_distance=difference(f['distance_after'], repeat['distance_after']))
    repeated = {key:difference(value, second[key]) for key,value in candidate.items()}
    report = dict(status='complete_frozen_candidate_face_normal_regression',
        strict_face_normal_passed=comparisons['updated_face_normals']['bitwise_equal'],
        strict_downstream_distance_passed=comparisons['propagated_complete_signed_distance']['bitwise_equal'],
        candidate_repeat_exact=all(row['bitwise_equal'] for row in repeated.values()),
        native_selected_fields_repeat_exact=all(row['bitwise_equal'] for row in native.values()),
        comparisons=comparisons, candidate_repeat=repeated, native_repeat=native,
        native_original_area_reference='not captured; real-data accuracy not assessed',
        candidate_original_area_sha256=array_sha(candidate['cached_original_area_mm2']),
        fallback_counts={str(value):int(np.count_nonzero(candidate['fallback'] == value)) for value in range(3)},
        module_sha256=sha(implementation.__file__), benchmark_sha256=sha(__file__),
        source_inputs_sha256=dict(capture=sha(capture), native_repeat=sha(native_repeat), patch=sha(patch_capture), observer_receipt=sha(observer_receipt)),
        observer_receipt=json.loads(observer_receipt.read_text()),
        candidate_npz_sha256=sha(output / 'candidate.private.npz'),
        vertices=len(f['vertex_fields']), faces=len(f['faces']), patch_vertices=len(patch['selected_vertices']), updated_faces=len(selected),
        coordinate_space='orig tkregister/surface RAS mm; normals dimensionless; original_area cache mm^2',
        seconds=dict(top_level_imports=_IMPORT, first_complete_face_api=first, repeat_complete_face_api=warm,
                     downstream_signed_distance_api=distance_seconds),
        cpu_threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)), host=platform.node(),
        python=platform.python_version(), numpy=np.__version__, gpu_used=False, process_gpu_memory_bytes=None,
        native_component_seconds=None, original_component_speedup=None,
        scope='same-input touched face cache and frozen-input distance propagation; not vertex normals, raw MRI preparation or complete GA',
        overall_equivalence='not_assessed')
    _plot(f, selected, candidate['face_normals'], output / 'face_normal_error.png')
    report['figure_sha256'] = sha(output / 'face_normal_error.png')
    report['seconds']['api_including_load_npz_and_png_io_before_report_write'] = time.perf_counter() - started
    report['seconds']['script_wall_before_report_write'] = time.perf_counter() - _START
    (output / 'report.private.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def _plot(fields, selected, candidate, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz = fields['vertex_fields'][:, :3]
    centers = xyz[fields['faces'][selected]].mean(axis=1)
    error = np.linalg.norm(candidate.astype(np.float64) - fields['face_normals'][selected].astype(np.float64), axis=1)
    fig, ax = plt.subplots(figsize=(7,5))
    bg = xyz[::max(1,len(xyz)//5000)]
    ax.scatter(bg[:,1], bg[:,2], s=.2, c='lightgray')
    p = ax.scatter(centers[:,1], centers[:,2], c=error, s=5, cmap='magma', vmin=0, vmax=max(1e-7,float(error.max(initial=0))))
    ax.set(xlabel='surface RAS y (mm)', ylabel='surface RAS z (mm)', title=f'Touched candidate faces; max normal error={error.max(initial=0):.3g}')
    ax.set_aspect('equal'); fig.colorbar(p, ax=ax, label='normal vector error'); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','patch-capture','observer-receipt','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    report = run(**vars(parser.parse_args()))
    if not (report['strict_face_normal_passed'] and report['strict_downstream_distance_passed']
            and report['candidate_repeat_exact'] and report['native_selected_fields_repeat_exact']): raise SystemExit(1)


if __name__ == '__main__': main()
