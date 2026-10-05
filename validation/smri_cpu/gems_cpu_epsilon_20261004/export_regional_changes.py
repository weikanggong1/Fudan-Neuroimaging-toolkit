"""Export every recorded regional change, including failed and empty regions."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    spec = importlib.util.spec_from_file_location(
        'encoding', Path(__file__).parent.parent / 'gems_fixes_20261004/report_encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    columns = ['report', 'report_sha256', 'group', 'label', 'name',
               'nonempty_before', 'nonempty_after',
               'dice_before', 'dice_after', 'dice_delta',
               'hard_volume_official_mm3', 'hard_volume_before_mm3', 'hard_volume_after_mm3',
               'hard_volume_error_before', 'hard_volume_error_after', 'hard_volume_error_delta',
               'soft_volume_official_mm3', 'soft_volume_before_mm3', 'soft_volume_after_mm3',
               'soft_volume_absolute_error_before_mm3', 'soft_volume_absolute_error_after_mm3',
               'soft_volume_absolute_error_delta_mm3',
               'soft_volume_error_before', 'soft_volume_error_after', 'soft_volume_error_delta',
               'passed_before', 'passed_after']
    with args.output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        count = 0
        for path in args.reports:
            report = encoding.load_report(path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            for group in report['regional_changes']:
                old = next(item for item in report['baseline']['groups'] if item['id'] == group['id'])
                new = next(item for item in report['candidate']['groups'] if item['id'] == group['id'])
                assert old['grid']['voxel_volume_mm3'] == new['grid']['voxel_volume_mm3']
                voxel_volume = old['grid']['voxel_volume_mm3']
                old_rows = {row['label']: row for pair in old['pairs'] if pair['kind'] == 'cross_method'
                            for row in pair['regions']}
                new_rows = {row['label']: row for pair in new['pairs'] if pair['kind'] == 'cross_method'
                            for row in pair['regions']}
                for row in group['regions']:
                    first, second = old_rows[row['label']], new_rows[row['label']]
                    assert first['first_voxels'] == second['first_voxels']
                    volumes = {'hard_volume_official_mm3': first['first_voxels'] * voxel_volume,
                               'hard_volume_before_mm3': first['second_voxels'] * voxel_volume,
                               'hard_volume_after_mm3': second['second_voxels'] * voxel_volume}
                    writer.writerow({'report': path.name, 'report_sha256': checksum,
                                     'group': group['id'],
                                     **{key: row.get(key) for key in columns[3:]}, **volumes})
                    count += 1
    print(json.dumps({'rows': count, 'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'empty_values': 'missing or not assessed; zero is preserved as a numerical value'}))


if __name__ == '__main__':
    main()
