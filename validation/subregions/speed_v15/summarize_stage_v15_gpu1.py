import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
label = 'full_stage_fast_v15_gpu1_20261001'
comparison = json.loads((root / (label + '_vs_v12.json')).read_text())
report = json.loads((root / label / 'report.json').read_text())
api_report = json.loads((root / label / 'api_report.json').read_text())
rows = comparison['old_new']['native']['per_label']


def stats(values):
    values = [value for value in values if value is not None]
    return {'min': min(values), 'median': float(np.median(values)), 'max': max(values)} if values else None


def official_family(row):
    regions = row['regions'] + row.get('empty_hard_regions', [])
    evaluated = [value for value in regions if value['reference_voxels'] or value['fnit_voxels']]
    return {'foreground_dice': row['foreground_dice'], 'evaluated_labels': len(evaluated),
            'both_empty_labels': [value['label'] for value in regions if not value['reference_voxels'] and not value['fnit_voxels']],
            'accepted': sum(value.get('accepted') is True for value in regions),
            'per_label_dice': stats([value.get('dice') for value in evaluated]),
            'hard_relative_difference': stats([value.get('hard_volume_difference') for value in evaluated]),
            'soft_relative_difference': stats([value.get('soft_volume_difference') for value in regions])}

families = {name: {'old_new': values,
                  'official_old': official_family(comparison['official']['old']['families'][name]),
                  'official_new': official_family(comparison['official']['new']['families'][name])}
            for name, values in comparison['old_new']['families'].items()}
soft_rows = [row for row in rows if row['soft_volume_mm3']['signed_difference'] is not None]
worst_absolute = max(soft_rows, key=lambda row: abs(row['soft_volume_mm3']['signed_difference']))
relative_rows = [row for row in soft_rows if row['soft_volume_mm3']['relative_abs_difference'] is not None]
worst_relative = max(relative_rows, key=lambda row: row['soft_volume_mm3']['relative_abs_difference'])
joint = comparison['official']['joint_acceptance']
final_jac = report['fit_min_jacobians']
summary = {'scope': 'completed same-input real official_stage_inputs T1; old-new and official metrics are distinct',
           'reports': comparison['reports'], 'inputs': comparison['inputs'],
           'source_provenance': comparison['source_provenance'], 'checks': comparison['checks'],
           'checks_passed': comparison['checks_passed'],
           'final_min_jacobians': final_jac,
           'all_four_final_jacobians_positive': len(final_jac) == 4 and all(value is not None and value > 0 for value in final_jac.values()),
           'native': {'geometry': comparison['old_new']['native']['geometry'],
                      'different_voxels': comparison['old_new']['native']['different_voxels'],
                      'foreground_roi': comparison['old_new']['foreground'],
                      'per_label_dice': comparison['old_new']['native']['old_new_label_dice'],
                      'zero_dice_labels': [row['label'] for row in rows if row['old_new_dice'] == 0]},
           'families': families,
           'all_110_soft': {'rows': len(soft_rows), 'per_label': [{'label': row['label'], 'name': row['name'],
                           **row['soft_volume_mm3']} for row in soft_rows],
                           'worst_absolute_change': worst_absolute, 'worst_relative_change': worst_relative},
           'official_joint_acceptance': joint,
           'official_strict_all': {'dice_min': .95, 'hard_relative_difference_max': .05,
                                  'old_all_evaluated_pass': joint['old_accepted'] == joint['old_evaluated_labels'],
                                  'new_all_evaluated_pass': joint['new_accepted'] == joint['new_evaluated_labels'],
                                  'both_empty_excluded_from_hard_acceptance': True},
           'highres': {name: values['geometry'] for name, values in comparison['old_new']['highres'].items()},
           'performance': {**comparison['performance'], 'new_api_total_including_save_seconds': report['api_total_seconds'],
                           'new_output_save_seconds': report['output_save_seconds'],
                           'new_detailed_api_timings': api_report['timings']},
           'environment': comparison['environment'],
           'new_initialization_timings': {name: value.get('timing_seconds', value.get('seconds')) for name, value in report['initialization'].items()}}
soft_gates = {}
for tag in ('old', 'new'):
    official_rows = [row['official'][tag] for row in rows]
    differences = [row.get('soft_volume_difference') for row in official_rows]
    present = [value for value in differences if value is not None]
    soft_gates[tag] = {'reference_soft_comparisons_present': len(present), 'labels': len(rows),
                       'relative_difference_max': max(present) if present else None,
                       'labels_within_five_percent': sum(value <= .05 for value in present),
                       'all_110_within_five_percent': len(present) == 110 and all(value <= .05 for value in present),
                       'interpretation': 'soft volume check is separate from the saved Dice plus hard volume acceptance'}
summary['official_soft_volume_check'] = soft_gates
output = root / (label + '_summary.json')
with output.open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'output': str(output), 'jacobians': final_jac, 'official': joint,
                  'wall': comparison['performance']['api_wall_seconds'],
                  'foreground_roi': comparison['old_new']['foreground']}))
