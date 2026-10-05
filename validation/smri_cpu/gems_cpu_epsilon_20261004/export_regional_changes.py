"""Export every recorded regional change, including failed and empty regions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    columns = ['report', 'report_sha256', 'group', 'label', 'name',
               'nonempty_before', 'nonempty_after',
               'dice_before', 'dice_after', 'dice_delta',
               'hard_volume_error_before', 'hard_volume_error_after', 'hard_volume_error_delta',
               'soft_volume_official_mm3', 'soft_volume_before_mm3', 'soft_volume_after_mm3',
               'soft_volume_absolute_error_before_mm3', 'soft_volume_absolute_error_after_mm3',
               'soft_volume_absolute_error_delta_mm3',
               'soft_volume_error_before', 'soft_volume_error_after', 'soft_volume_error_delta',
               'passed_before', 'passed_after']
    with args.output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        count = 0
        for path in args.reports:
            report = json.loads(path.read_text())
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            for group in report['regional_changes']:
                for row in group['regions']:
                    writer.writerow({'report': path.name, 'report_sha256': checksum,
                                     'group': group['id'],
                                     **{key: row.get(key) for key in columns[3:]}})
                    count += 1
    print(json.dumps({'rows': count, 'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'empty_values': 'missing or not assessed; zero is preserved as a numerical value'}))


if __name__ == '__main__':
    main()
