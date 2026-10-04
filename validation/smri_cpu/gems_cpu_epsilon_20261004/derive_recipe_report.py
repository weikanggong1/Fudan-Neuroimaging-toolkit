"""Derive every regional soft-volume delta without changing the frozen scorer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def derive(report):
    for before, after, changes in zip(report['baseline']['groups'],
                                      report['candidate']['groups'],
                                      report['regional_changes']):
        assert before['id'] == after['id'] == changes['id']
        old_rows = next(pair['regions'] for pair in before['pairs']
                        if pair['kind'] == 'cross_method')
        new_rows = next(pair['regions'] for pair in after['pairs']
                        if pair['kind'] == 'cross_method')
        for old, new, row in zip(old_rows, new_rows, changes['regions']):
            assert old['label'] == new['label'] == row['label']
            reference = old['first_soft_volume_mm3']
            assert reference == new['first_soft_volume_mm3']
            first, second = old['second_soft_volume_mm3'], new['second_soft_volume_mm3']
            row.update(soft_volume_official_mm3=reference,
                       soft_volume_before_mm3=first, soft_volume_after_mm3=second)
            valid = reference is not None and first is not None and second is not None
            first_absolute = abs(first - reference) if valid else None
            second_absolute = abs(second - reference) if valid else None
            row.update(soft_volume_absolute_error_before_mm3=first_absolute,
                       soft_volume_absolute_error_after_mm3=second_absolute,
                       soft_volume_absolute_error_delta_mm3=(second_absolute - first_absolute
                                                           if valid else None))
            relative_valid = valid and reference > 0
            first_relative = first_absolute / reference if relative_valid else None
            second_relative = second_absolute / reference if relative_valid else None
            row.update(soft_volume_error_before=first_relative,
                       soft_volume_error_after=second_relative,
                       soft_volume_error_delta=(second_relative - first_relative
                                                if relative_valid else None))
        for key in ('soft_volume_absolute_error_delta_mm3', 'soft_volume_error_delta'):
            valid_rows = [row for row in changes['regions'] if row[key] is not None]
            changes[key + '_extremes'] = {
                'minimum': min(valid_rows, key=lambda row: row[key]) if valid_rows else None,
                'maximum': max(valid_rows, key=lambda row: row[key]) if valid_rows else None,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    raw = args.input.read_bytes()
    original = json.loads(raw)
    result = derive(json.loads(raw))
    result['derived_soft_volume_deltas_from_original_report_sha256'] = hashlib.sha256(raw).hexdigest()
    result['derivation_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        'gems_report_encoding', Path(__file__).parent.parent / 'gems_fixes_20261004/report_encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    args.output.write_text(encoding.compact(result) + '\n')
    assert json.loads(args.output.read_text()) == result
    for old, new in zip(original['regional_changes'], result['regional_changes']):
        assert old['old_gate'] == new['old_gate'] and old['new_gate'] == new['new_gate']
        for first, second in zip(old['regions'], new['regions']):
            assert all(second[key] == value for key, value in first.items())
    print(json.dumps({'source_sha256': hashlib.sha256(raw).hexdigest(),
                      'decoded_canonical_sha256': encoding.digest(result),
                      'bytes': args.output.stat().st_size,
                      'all_original_regional_metrics_preserved': True}))


if __name__ == '__main__':
    main()
