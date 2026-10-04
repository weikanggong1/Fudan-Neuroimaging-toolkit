"""Posthoc whole-recipe scoring with the established grid and regional gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def extremes(rows, field):
    valid = [row for row in rows if row[field] is not None]
    return {'minimum': min(valid, key=lambda row: row[field]) if valid else None,
            'maximum': max(valid, key=lambda row: row[field]) if valid else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'baseline', 'official-root', 'like', 'audit-helper', 'score-helper', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    scorer = module(args.score_helper, 'gems_existing_score')
    audit = module(args.audit_helper, 'gems_existing_fixed_grid')
    before = scorer.score(args.baseline, args.official_root, args.like, audit)
    after = scorer.score(args.candidate, args.official_root, args.like, audit)
    changes = []
    for old_group, new_group in zip(before['groups'], after['groups']):
        assert old_group['id'] == new_group['id']
        old = next(p for p in old_group['pairs'] if p['kind'] == 'cross_method')['regions']
        new = next(p for p in new_group['pairs'] if p['kind'] == 'cross_method')['regions']
        rows = []
        for initial, updated in zip(old, new):
            assert initial['label'] == updated['label']
            old_dice, new_dice = initial['dice'], updated['dice']
            old_volume = initial['hard_volume_relative_to_official']
            new_volume = updated['hard_volume_relative_to_official']
            rows.append({'label': initial['label'], 'name': initial['name'],
                         'nonempty_before': initial['nonempty'], 'nonempty_after': updated['nonempty'],
                         'dice_before': old_dice, 'dice_after': new_dice,
                         'dice_delta': new_dice - old_dice if old_dice is not None and new_dice is not None else None,
                         'hard_volume_error_before': old_volume, 'hard_volume_error_after': new_volume,
                         'hard_volume_error_delta': new_volume - old_volume if old_volume is not None and new_volume is not None else None,
                         'passed_before': initial['preexisting_gate_passed'],
                         'passed_after': updated['preexisting_gate_passed']})
        summary = {'id': old_group['id'], 'regions': rows,
                   'old_gate': old_group['gate_summary'], 'new_gate': new_group['gate_summary'],
                   'dice_delta_extremes': extremes(rows, 'dice_delta'),
                   'volume_error_delta_extremes': extremes(rows, 'hard_volume_error_delta'),
                   'lost_old_passed_labels': [row['label'] for row in rows if row['passed_before'] and not row['passed_after']]}
        changes.append(summary)
    report = {'schema': 'fnit.gems.cpu-epsilon.recipe-score.v1', 'scope': 'fixed saved real norm/aseg/wmparc recipe; official fine labels loaded only after fitting',
              'threshold': {'dice_minimum': .95, 'hard_volume_relative_error_maximum': .05},
              'both_empty_dice': None, 'baseline': before, 'candidate': after, 'regional_changes': changes,
              'scoring_sources_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (args.score_helper, args.audit_helper, Path(__file__))}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for group in changes:
        print(json.dumps({key: value for key, value in group.items() if key != 'regions'}), flush=True)


if __name__ == '__main__':
    main()
