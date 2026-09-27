"""按有序顶点/面比较同候选输入的 Python、原生和归档官方 sphere。"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_parts(path):
    with path.open('rb') as stream:
        if stream.read(3) != b'\xff\xff\xfe':
            raise ValueError('not a FreeSurfer triangle surface')
        stamp = stream.readline().decode('utf-8', errors='replace').strip()
        stream.readline()
        vertices, faces = struct.unpack('>ii', stream.read(8))
        geometry = stream.read(12 * (vertices + faces))
        footer = stream.read()
    return {'stamp': stamp, 'geometry_sha256': hashlib.sha256(geometry).hexdigest(),
            'footer_sha256': hashlib.sha256(footer).hexdigest(),
            'footer_length': len(footer)}, footer


def comparison(candidate, reference):
    candidate_xyz, candidate_faces = fsio.read_geometry(str(candidate))
    reference_xyz, reference_faces = fsio.read_geometry(str(reference))
    if candidate_xyz.shape != reference_xyz.shape or candidate_faces.shape != reference_faces.shape:
        return {'same_shape': False, 'candidate_vertices': len(candidate_xyz),
                'reference_vertices': len(reference_xyz),
                'candidate_faces': len(candidate_faces), 'reference_faces': len(reference_faces)}
    candidate_xyz = np.asarray(candidate_xyz, np.float32)
    reference_xyz = np.asarray(reference_xyz, np.float32)
    displacement = np.linalg.norm(candidate_xyz.astype(np.float64)
                                  - reference_xyz.astype(np.float64), axis=1)
    return {'same_shape': True,
            'vertices': len(candidate_xyz), 'faces': len(candidate_faces),
            'ordered_faces_equal': bool(np.array_equal(candidate_faces, reference_faces)),
            'exact_coordinate_components': int(np.count_nonzero(candidate_xyz == reference_xyz)),
            'coordinate_components': int(candidate_xyz.size),
            'exact_vertices': int(np.count_nonzero(np.all(candidate_xyz == reference_xyz, axis=1))),
            'mean_euclidean_mm': float(displacement.mean()),
            'p99_euclidean_mm': float(np.quantile(displacement, 0.99)),
            'max_euclidean_mm': float(displacement.max()),
            'vertices_over_0p1_mm': int(np.count_nonzero(displacement > 0.1)),
            'candidate_sha256': sha256(candidate), 'reference_sha256': sha256(reference)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inflated', required=True, type=Path, help='本轮候选 inflated 输入')
    parser.add_argument('--smoothwm', required=True, type=Path, help='本轮候选 smoothwm 输入')
    parser.add_argument('--python-sphere', required=True, type=Path, help='修复后的 Python sphere')
    parser.add_argument('--same-input-official', required=True, type=Path,
                        help='同候选输入的 FreeSurfer 官方 sphere')
    parser.add_argument('--archived-official', required=True, type=Path,
                        help='原始归档 FreeSurfer sphere')
    parser.add_argument('--python-report', required=True, type=Path,
                        help='Python run_standard_sphere 的 JSON 报告')
    parser.add_argument('--native-time', required=True, type=Path,
                        help='同输入官方运行的 wall time 文件')
    parser.add_argument('--report-json', required=True, type=Path, help='输出配对审计 JSON')
    args = parser.parse_args()
    run = json.loads(args.python_report.read_text())
    python_parts, python_footer = file_parts(args.python_sphere)
    same_parts, same_footer = file_parts(args.same_input_official)
    archived_parts, _ = file_parts(args.archived_official)
    result = {
        'audit_script_sha256': sha256(Path(__file__)),
        'inputs': {'inflated_sha256': sha256(args.inflated),
                   'smoothwm_sha256': sha256(args.smoothwm)},
        'python': {'sphere_sha256': sha256(args.python_sphere),
                   'report_sha256': sha256(args.python_report),
                   'update_count': len(run['updates']),
                   'finish_negative_counts': run['negative_counts'],
                   'total_seconds_including_io': run['total_seconds_including_io'],
                   'metric_seconds_including_jit': run['metric_seconds_including_jit'],
                   'finish_seconds': run['finish_seconds']},
        'same_input_official_time': args.native_time.read_text().strip(),
        'surface_file_parts': {
            'python': python_parts, 'same_input_official': same_parts,
            'archived_official': archived_parts,
            'python_footer_prefix_of_same_input_official': same_footer.startswith(python_footer)},
        'python_vs_same_input_official': comparison(args.python_sphere, args.same_input_official),
        'python_vs_archived_official': comparison(args.python_sphere, args.archived_official),
        'same_input_vs_archived_official': comparison(args.same_input_official,
                                                       args.archived_official),
    }
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
