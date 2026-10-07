"""Score completed GEMS stages with the established fixed-grid audit."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


FILES = {
    'brainstem': ('brainstem', 'brainstemSsLabels', ['brainstemSsLabels.volumes.txt']),
    'thalamus': ('thalamus', 'ThalamicNuclei', ['ThalamicNuclei.volumes.txt']),
    'hippo-amygdala-left': ('hippo-amygdala', 'lh.hippoAmygLabels',
                           ['lh.hippoSfVolumes.txt', 'lh.amygNucVolumes.txt']),
    'hippo-amygdala-right': ('hippo-amygdala', 'rh.hippoAmygLabels',
                            ['rh.hippoSfVolumes.txt', 'rh.amygNucVolumes.txt']),
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def score(directory, official_root, like, audit):
    report = json.loads((directory / 'report.json').read_text())
    groups, sources = [], {}
    for family in report['structures']:
        metadata = {int(label): row for label, row in report['labels'].items()
                    if row['source'] == family}
        if not metadata:
            raise ValueError('No declared labels for ' + family)
        folder, stem, volume_names = FILES[family]
        official_dir = official_root / folder
        for space in ('native', 'hr'):
            official = official_dir / (stem + ('.FSvoxelSpace.mgz' if space == 'native' else '.mgz'))
            candidate = (directory / 'subregions_native.nii.gz' if space == 'native'
                         else directory / 'highres' / (family + '.nii.gz'))
            sources[str(official.relative_to(official_root))] = digest(official)
            definition = {
                'id': family + '_' + space, 'family': family, 'space': space,
                'label_ids': sorted(metadata),
                'label_names': {str(k): v['name'] for k, v in metadata.items()},
                'grid': {'like': str(like)} if space == 'native' else {'union_like': str(official)},
                'official': [{'id': 'official_frozen', 'labels': str(official),
                              'label_offset': 10000 if family.endswith('-right') else 0,
                              'soft_volumes_files': [str(official_dir / name) for name in volume_names]}],
                'fnit': [{'id': 'candidate', 'labels': str(candidate),
                          'soft_volumes_file': str(directory / 'volumes.tsv')}]}
            result = audit.audit_group(definition, Path('/'))
            pair = next(row for row in result['pairs'] if row['kind'] == 'cross_method')
            for region in pair['regions']:
                region['name'] = metadata[region['label']]['name']
                reference, actual = region['first_voxels'], region['second_voxels']
                relative = abs(reference - actual) / reference if reference else None
                region['hard_volume_relative_to_official'] = relative
                region['preexisting_gate_passed'] = (
                    region['dice'] >= .95 and relative <= .05
                    if region['dice'] is not None and relative is not None else False)
                region['nonempty'] = bool(reference or actual)
            rows = pair['regions']
            result['gate_summary'] = {
                'pass': sum(row['preexisting_gate_passed'] for row in rows),
                'nonempty': sum(row['nonempty'] for row in rows),
                'both_empty': sum(not row['nonempty'] for row in rows),
                'all_nonempty_passed': all(row['preexisting_gate_passed'] for row in rows if row['nonempty']),
            }
            groups.append(result)
    return {'groups': groups, 'official_label_sha256': sources,
            'candidate_report_sha256': digest(directory / 'report.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--like', type=Path, required=True)
    parser.add_argument('--audit-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output path')
    spec = importlib.util.spec_from_file_location('fixed_grid_audit', args.audit_helper)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    report = {'schema': 'fnit.gems.fixed_grid_repair.v1', 'status': 'scored_after_fitting',
              'scope': 'same_frozen_stage_inputs_no_reference_labels_used_for_fitting',
              'both_empty_dice': None,
              'threshold': {'per_label_dice_minimum': .95,
                            'hard_volume_relative_to_official_maximum': .05},
              'audit_helper_sha256': digest(args.audit_helper), 'like_sha256': digest(args.like),
              'candidate': score(args.candidate, args.official_root, args.like, audit)}
    if args.baseline:
        report['baseline'] = score(args.baseline, args.official_root, args.like, audit)
        changes = []
        for first, second in zip(report['baseline']['groups'], report['candidate']['groups']):
            assert first['id'] == second['id']
            old = next(p for p in first['pairs'] if p['kind'] == 'cross_method')['regions']
            new = next(p for p in second['pairs'] if p['kind'] == 'cross_method')['regions']
            delta = []
            for before, after in zip(old, new):
                assert before['label'] == after['label']
                dice_before, dice_after = before['dice'], after['dice']
                delta.append({'label': before['label'], 'name': before['name'],
                              'dice_before': dice_before, 'dice_after': dice_after,
                              'dice_delta': dice_after - dice_before if dice_before is not None and dice_after is not None else None,
                              'hard_volume_error_before': before['hard_volume_relative_to_official'],
                              'hard_volume_error_after': after['hard_volume_relative_to_official'],
                              'passed_before': before['preexisting_gate_passed'],
                              'passed_after': after['preexisting_gate_passed']})
            changes.append({'id': first['id'], 'regions': delta})
        report['regional_changes'] = changes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for group in report['candidate']['groups']:
        print(group['id'], group['gate_summary'], flush=True)


if __name__ == '__main__':
    main()
