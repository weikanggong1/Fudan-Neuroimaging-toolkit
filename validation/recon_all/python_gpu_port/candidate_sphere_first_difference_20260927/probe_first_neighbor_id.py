"""在已保存的候选 smoothwm 上追踪首次邻居 ID 分叉；不运行球面优化。"""

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import sphere_standard_metric as metric
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.sphere_standard_python import original_metric_distance_rings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoothwm', type=Path, required=True, help='候选 smoothwm 表面')
    parser.add_argument('--capture-dir', type=Path, required=True, help='保存的原生矩阵目录')
    parser.add_argument('--report-json', type=Path, required=True, help='输出审计 JSON')
    args = parser.parse_args()
    start = time.perf_counter()
    xyz, faces = fsio.read_geometry(str(args.smoothwm))
    native_offsets = np.fromfile(args.capture_dir / 'native_offsets.bin', dtype='<u8')
    native_ids = np.fromfile(args.capture_dir / 'native_neighbor_ids.bin', dtype='<i4')
    vertex, position = 42548, 73

    # 临时编译一个有条件日志的函数副本；FNIT 源文件和算法均不改。
    source = inspect.getsource(metric._sample_rows.py_func)
    anchor = '                    tries += 1\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '''                    if vertex == 42548 and depth == 7 and count == 73:
                        print("trial", index, other, accepted, tries,
                              float(min_angle))
''' + anchor)
    namespace = metric.__dict__.copy()
    exec(source, namespace)
    metric._sample_rows = namespace['_sample_rows']

    offsets, ids, raw, sampling = metric.sample_standard_metric_matrix(
        xyz, faces, limit=vertex + 1)
    assert np.array_equal(offsets, native_offsets[:vertex + 2])
    first = int(np.flatnonzero(ids != native_ids[:len(ids)])[0])
    assert first == int(offsets[vertex]) + position
    native_row = native_ids[int(offsets[vertex]):int(offsets[vertex + 1])]
    python_row = ids[int(offsets[vertex]):int(offsets[vertex + 1])]
    neighbors = ordered_neighbors(faces, len(xyz))
    rings, _ = original_metric_distance_rings(
        xyz, faces, vertex=vertex, neighbors=neighbors)
    ring_ids = rings[6]
    begin = sum(map(len, rings[:3])) + 4 * 8
    previous = [int(v) for v in python_row[begin:position]]
    ids_at_first_difference = [int(native_row[position]), int(python_row[position])]
    report = {
        'first_flat_index': first, 'vertex': vertex, 'row_position': position,
        'row_count': len(python_row), 'ring_sizes': [len(r) for r in rings],
        'ring7_begin_row_position': begin,
        'ring7_previous_ids': previous,
        'native_row_around_first_difference': native_row[position-4:position+4].tolist(),
        'python_row_around_first_difference': python_row[position-4:position+4].tolist(),
        'candidate_ids': ids_at_first_difference,
        'candidate_ring7_positions': [ring_ids.index(v) for v in ids_at_first_difference],
        'candidate_angles_to_previous': [{
            'id': candidate,
            'angles_radians': [float(metric._angle(np.asarray(xyz, np.float32),
                                                    vertex, candidate, prev))
                               for prev in previous],
        } for candidate in ids_at_first_difference],
        'initial_min_angle_radians': float(np.float32(0.9 * 2 * math.pi / 8)),
        'python_metric_sampling': sampling,
        'wall_seconds_including_jit': time.perf_counter() - start,
    }
    args.report_json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
