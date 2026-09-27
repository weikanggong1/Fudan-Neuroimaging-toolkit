"""比较保存的候选与归档官方 LH 表面，定位同索引首差。"""

import argparse
import json
from pathlib import Path

from audit_full_sphere import comparison, sha256


STAGES = (
    'orig.premesh', 'orig.nofix', 'smoothwm.nofix', 'inflated.nofix',
    'qsphere.nofix', 'orig', 'white.preaparc', 'smoothwm', 'inflated', 'sphere',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-surf-dir', required=True, type=Path,
                        help='FNIT 候选 subject/surf 目录')
    parser.add_argument('--official-surf-dir', required=True, type=Path,
                        help='归档官方 subject/surf 目录')
    parser.add_argument('--repaired-sphere', required=True, type=Path,
                        help='修复后完整 Python LH sphere 输出')
    parser.add_argument('--report-json', required=True, type=Path,
                        help='输出有序阶段审计 JSON')
    args = parser.parse_args()
    stages = {}
    first = None
    for name in STAGES:
        candidate = (args.repaired_sphere if name == 'sphere'
                     else args.candidate_surf_dir / ('lh.' + name))
        official = args.official_surf_dir / ('lh.' + name)
        if not candidate.is_file() or not official.is_file():
            stages[name] = {'present': False}
            continue
        entry = comparison(candidate, official)
        entry['candidate_sha256'] = sha256(candidate)
        entry['official_sha256'] = sha256(official)
        stages[name] = entry
        if first is None and (not entry['same_shape'] or not entry['ordered_faces_equal']
                              or entry['exact_coordinate_components'] != entry['coordinate_components']):
            first = name
    result = {'stage_order': STAGES, 'first_nonmatching_stage': first, 'stages': stages}
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'first_nonmatching_stage': first,
                      'stages': {name: {'same_shape': entry.get('same_shape'),
                                        'faces_equal': entry.get('ordered_faces_equal'),
                                        'exact_xyz': entry.get('exact_coordinate_components'),
                                        'mean_mm': entry.get('mean_euclidean_mm'),
                                        'max_mm': entry.get('max_euclidean_mm')}
                                 for name, entry in stages.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
