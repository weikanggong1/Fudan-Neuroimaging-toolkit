"""只读比较同一 LH white.preaparc 命令的三张 MRI 输入和阈值文件。"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_full_sphere import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-subject', required=True, type=Path,
                        help='FNIT 候选 subject 目录')
    parser.add_argument('--official-subject', required=True, type=Path,
                        help='归档官方 subject 目录')
    parser.add_argument('--report-json', required=True, type=Path,
                        help='输出输入值和空间几何核对 JSON')
    args = parser.parse_args()
    volumes = {}
    for name in ('wm.mgz', 'aseg.presurf.mgz', 'brain.finalsurfs.mgz'):
        candidate_path = args.candidate_subject / 'mri' / name
        official_path = args.official_subject / 'mri' / name
        candidate = nib.load(str(candidate_path))
        official = nib.load(str(official_path))
        a = np.asanyarray(candidate.dataobj)
        b = np.asanyarray(official.dataobj)
        volumes[name] = {
            'candidate_sha256': sha256(candidate_path),
            'official_sha256': sha256(official_path),
            'shape': list(a.shape),
            'shape_equal': bool(a.shape == b.shape),
            'mismatched_voxels': int(np.count_nonzero(a != b)) if a.shape == b.shape else None,
            'affine_max_abs_mm': float(np.max(np.abs(candidate.affine - official.affine))),
        }
    stats_name = 'autodet.gw.stats.lh.dat'
    candidate_stats = args.candidate_subject / 'surf' / stats_name
    official_stats = args.official_subject / 'surf' / stats_name
    result = {'volumes': volumes,
              'threshold_stats': {'candidate_sha256': sha256(candidate_stats),
                                  'official_sha256': sha256(official_stats),
                                  'bytes_equal': candidate_stats.read_bytes() == official_stats.read_bytes()}}
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
