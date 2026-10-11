"""只读量化pial阳性面对的几何类型，不改变碰撞质量门。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slice_parameters(triangle, signed, direction, epsilon=1e-9):
    points = [p for p, d in zip(triangle, signed) if abs(d) <= epsilon]
    for a, b, da, db in zip(triangle, np.roll(triangle, -1, axis=0), signed, np.roll(signed, -1)):
        if da*db < 0: points.append(a+(b-a)*da/(da-db))
    return None if not points else np.asarray(points)@direction


def _coplanar_area(first, second, normal):
    """旧诊断专用：normal须为单位法向，返回mm²；不用于质量分类。"""
    drop = int(np.argmax(np.abs(normal))); axes = [i for i in range(3) if i != drop]
    polygon = [p.copy() for p in first[:, axes]]; clip = second[:, axes]
    orient = np.cross(clip[1]-clip[0], clip[2]-clip[0])
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        previous = polygon; polygon = []
        if not previous: break
        def distance(point): return float(np.cross(b-a, point-a))*np.sign(orient)
        for p, q in zip(previous, previous[1:]+previous[:1]):
            dp, dq = distance(p), distance(q)
            if dp >= 0: polygon.append(p)
            if dp*dq < 0: polygon.append(p+(q-p)*dp/(dp-dq))
    if len(polygon) < 3: return 0.
    points = np.asarray(polygon); after = np.roll(points, -1, axis=0)
    return float(abs(np.sum(points[:, 0]*after[:, 1]-points[:, 1]*after[:, 0]))/2/abs(normal[drop]))


def pair_geometry(first: np.ndarray, second: np.ndarray) -> dict:
    """两有限F64三角面；返回面积、穿透、交线/共面面积和重复点计数。

    surface RAS/mm，正面积；不会调用FNIT质量检测或改变其阈值。
    1e-9mm用于F64交线端点稳定性，1e-5mm仅给诊断近接触标签；
    所有原始距离保留，分类不能作为碰撞豁免。
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != (3, 3) or second.shape != (3, 3) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('two finite (3,3) triangles required')
    normal_a = np.cross(first[1]-first[0], first[2]-first[0]); normal_b = np.cross(second[1]-second[0], second[2]-second[0])
    length_a, length_b = np.linalg.norm(normal_a), np.linalg.norm(normal_b)
    if length_a == 0 or length_b == 0: raise ValueError('positive triangle area required')
    na, nb = normal_a/length_a, normal_b/length_b
    a_to_b, b_to_a = (first-second[0])@nb, (second-first[0])@na
    direction = np.cross(na, nb); sine = np.linalg.norm(direction)
    edge_a = np.linalg.norm(np.roll(first, -1, axis=0)-first, axis=1)
    edge_b = np.linalg.norm(np.roll(second, -1, axis=0)-second, axis=1)
    distance = np.linalg.norm(first[:, None]-second[None], axis=2)
    result = {'first_area_mm2': float(length_a/2), 'second_area_mm2': float(length_b/2),
              'first_collinearity_ratio': float(length_a/edge_a.max()**2),
              'second_collinearity_ratio': float(length_b/edge_b.max()**2),
              'normal_cross_magnitude': float(sine),
              'first_to_second_plane_min_mm': float(a_to_b.min()), 'first_to_second_plane_max_mm': float(a_to_b.max()),
              'second_to_first_plane_min_mm': float(b_to_a.min()), 'second_to_first_plane_max_mm': float(b_to_a.max()),
              'first_plane_straddle_depth_mm': float(max(0., min(a_to_b.max(), -a_to_b.min()))),
              'second_plane_straddle_depth_mm': float(max(0., min(b_to_a.max(), -b_to_a.min()))),
              'exact_duplicate_corner_pairs': int(np.count_nonzero(distance == 0)),
              'minimum_cross_corner_distance_mm': float(distance.min()),
              'intersection_segment_overlap_mm': None, 'coplanar_overlap_area_mm2': None}
    if sine < 1e-10:
        if max(np.max(np.abs(a_to_b)), np.max(np.abs(b_to_a))) <= 1e-9:
            area = _coplanar_area(first, second, na); result['coplanar_overlap_area_mm2'] = area
            result['classification'] = 'coplanar_area_overlap' if area > 1e-10 else 'coplanar_edge_or_point_contact'
        else: result['classification'] = 'parallel_near_contact_no_exact_coplanarity'
    else:
        direction /= sine
        a = _slice_parameters(first, a_to_b, direction); b = _slice_parameters(second, b_to_a, direction)
        if a is None or b is None:
            result['classification'] = 'no_exact_plane_crossing_at_1e-9mm'
        else:
            overlap = float(min(a.max(), b.max())-max(a.min(), b.min()))
            result['intersection_segment_overlap_mm'] = overlap
            if overlap > 1e-5: result['classification'] = 'noncoplanar_segment_overlap'
            elif overlap >= -1e-9: result['classification'] = 'noncoplanar_point_or_short_segment_contact'
            else: result['classification'] = 'near_contact_disjoint_exact_slices'
    return result


def run(*, diagnostic_case: Path, input_hashes: Path, terminal_sha256: str,
        pair_artifact: Path, pair_artifact_sha256: str, output: Path) -> dict:
    """读取同七输入/末态/完整阳性面对工件；输出无坐标的聚合JSON。

    输入路径绑定显式SHA，output须不存在；七前置同此前阶段协议。
    terminal为float32(N,3)，white包含有序整数(F,3)，artifact为
    pairs(P,4:两面编号和两有向标志)/face_flags(F,) NPZ。只读、不
    调原版程序、不读参考、不改变固定域或质量门。返回所有逐面对
    数值、重复点/共线计数、诊断分类、版本/源码SHA、墙钟/RSS。
    非法哈希/几何/面对/输出已有直接抛异常；不把近接触分类作为通过。
    """
    started = time.perf_counter()
    import nibabel as nib
    import nibabel.freesurfer as fs
    if output.exists(): raise FileExistsError(output)
    expected = json.loads(input_hashes.read_text())
    if len(expected) != 7: raise ValueError('seven stage input hashes required')
    for name, digest in expected.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or _sha(diagnostic_case/'subject'/name) != digest:
            raise ValueError('seven stage input hash or relative path mismatch')
    terminal_path = diagnostic_case/'cleanup-result.private.npy'
    if _sha(terminal_path) != terminal_sha256 or _sha(pair_artifact) != pair_artifact_sha256:
        raise ValueError('terminal/pair artifact hash mismatch')
    xyz = np.load(terminal_path); white, faces = fs.read_geometry(diagnostic_case/'subject/surf/lh.white')
    if xyz.dtype != np.float32 or xyz.shape != white.shape or not np.isfinite(xyz).all(): raise ValueError('finite float32 terminal on the ordered white mesh required')
    with np.load(pair_artifact) as artifact:
        pairs = artifact['pairs']; flags = artifact['face_flags']
    if pairs.ndim != 2 or pairs.shape[1] != 4 or not np.issubdtype(pairs.dtype,np.integer) or flags.dtype != np.bool_ or flags.shape != (len(faces),) or np.any(pairs[:, :2] < 0) or np.any(pairs[:, :2] >= len(faces)):
        raise ValueError('complete canonical positive pair artifact required')
    if np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(pairs[:, :2],axis=0)) != len(pairs) or np.any((pairs[:, 2:] != 0)&(pairs[:, 2:] != 1)):
        raise ValueError('unique ordered pairs and boolean source directions required')
    inferred = np.zeros(len(faces),bool)
    inferred[pairs[pairs[:, 2].astype(bool),0]] = True
    inferred[pairs[pairs[:, 3].astype(bool),1]] = True
    if not np.array_equal(inferred,flags): raise ValueError('full face flags differ from source positive directed pairs')
    marked = np.unique(faces[flags]); _, multiplicities = np.unique(xyz[marked], axis=0, return_counts=True)
    triangle = xyz.astype(float)[faces]; normals = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]); area2 = np.linalg.norm(normals, axis=1)
    rows = [{'first_face': int(a), 'second_face': int(b), 'source_forward': bool(forward), 'source_reverse': bool(reverse)} | pair_geometry(triangle[a], triangle[b]) for a, b, forward, reverse in pairs]
    classes = {name: sum(row['classification'] == name for row in rows) for name in sorted({row['classification'] for row in rows})}
    result = {'schema_version': 1, 'execution_status': 'complete_readonly_diagnosis', 'scope': 'frozen pial pair geometry; not replacement quality detector or raw T1',
              'coordinate_space': 'surface_RAS', 'coordinate_unit': 'mm', 'vertices': len(xyz), 'faces': len(faces),
              'input_sha256': expected, 'terminal_sha256': terminal_sha256, 'pair_artifact_sha256': pair_artifact_sha256,
              'script_sha256': _sha(__file__), 'quality_gate_changed': False, 'original_stage_program_called': False,
              'diagnostic_thresholds': {'plane_slice_epsilon_mm': 1e-9, 'near_contact_length_mm': 1e-5, 'normal_cross_epsilon': 1e-10, 'coplanar_area_epsilon_mm2': 1e-10},
              'marked_faces': int(flags.sum()), 'marked_vertices': len(marked), 'positive_pairs': len(pairs),
              'exact_duplicate_marked_vertex_groups': int(np.count_nonzero(multiplicities > 1)),
              'maximum_exact_marked_vertex_multiplicity': int(multiplicities.max(initial=0)),
              'zero_area_faces_global': int(np.count_nonzero(area2 == 0)), 'minimum_global_area_mm2': float(area2.min()/2),
              'classes': classes, 'pairs': rows, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'nibabel': nib.__version__},
              'overall_metric_equivalence': 'not_assessed', 'gpu_memory_bytes': None,
              'peak_process_cpu_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
              'wall_seconds_before_report': time.perf_counter()-started}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-case', 'input-hashes', 'pair-artifact', 'output'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('terminal-sha256', 'pair-artifact-sha256'): p.add_argument('--'+name, required=True)
    a = p.parse_args(); j = run(**vars(a));print(json.dumps({'status':j['execution_status'],'classes':j['classes']}))


if __name__ == '__main__': main()
