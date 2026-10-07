"""Analyze existing scalar trajectories and label counts; never run fitting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_summary(stage):
    history = stage['objective_history']
    steps = stage['mesh_steps']
    budget = stage['outer_iteration_limit'] * stage['mesh_iteration_limit']
    accepted = history[1:]
    block_size = stage['mesh_iteration_limit']
    # Only full-budget stages have an unambiguous outer-iteration partition.
    blocks = ([accepted[i:i + block_size] for i in range(0, len(accepted), block_size)]
              if steps == budget or stage['outer_iteration_limit'] == 1 else [])
    return {'stage_index': stage['stage_index'], 'synthetic': stage['synthetic'],
            'outer_iteration_limit': stage['outer_iteration_limit'],
            'mesh_iteration_limit': stage['mesh_iteration_limit'],
            'mesh_step_budget': budget, 'mesh_steps': steps, 'budget_reached': steps == budget,
            'history_count': len(history), 'history_zero_is_em_nll_not_mesh_objective': True,
            'history_zero': history[0], 'first_recorded_mesh_objective': history[1],
            'last_recorded_mesh_objective': history[-1],
            'within_fixed_likelihood_increases': sum(b > a for block in blocks for a, b in zip(block, block[1:])),
            'within_fixed_likelihood_equal_values': sum(b == a for block in blocks for a, b in zip(block, block[1:])),
            'last_three_relative_cost_changes': [abs(a - b) / max(abs(b), 1)
                                                  for a, b in zip(history[-4:-1], history[-3:])],
            'mesh_evaluations': stage['mesh_evaluations'],
            'line_search_restarts': stage['line_search_restarts'],
            'line_search_recovered': stage['line_search_recovered'],
            'mesh_interpolation_precision': stage.get('mesh_interpolation_precision', 'original FP32'),
            'mesh_data_epsilon': stage.get('mesh_data_epsilon', 0),
            'gems_fit_seconds': stage['gems_fit_seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'encoding', Path(__file__).parent.parent / 'gems_fixes_20261004/report_encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    capture_path = args.directory.parent / 'gems_first_state_20261004/capture_fnit.py'
    capture_source = capture_path.read_text()
    assert 'fixed = options["fixed_gaussians"]' in capture_source
    assert 'optim.CachedArmijoLBFGS.step = first_step' in capture_source
    result = {'scope': 'Local saved-report analysis only; no TTY, objective reevaluation or fitting',
              'first_capture': {'phase': 'synthetic stage1, initial closure before optimizer update',
                                'covered': 'actual same-state cost and complete gradient',
                                'not_covered': 'first trial/accepted point, search direction, displacement and Armijo decision',
                                'capture_code_sha256': checksum(capture_path)},
              'objective_interpretation': 'EM NLL history[0] differs in definition from later full data+prior mesh costs. Old/new mesh formulas and states also differ; lower scalar values are not an official accuracy/convergence ranking.',
              'structures': {}, 'label_examples': {}, 'source_read_sha256': {}, 'report_sha256': {}}
    for name in ('core.py', 'optim.py', 'recipes/base.py', 'recipes/thalamus.py', 'recipes/hippo_amygdala.py'):
        result['source_read_sha256'][name] = checksum(args.source / 'fnit/gems' / name)
    result['source_read_note'] = 'Local candidate core includes the documented inactive dense guard; running v1 source hashes remain those in recovery_queue/source/binding. Optimizer and recipe definitions are unchanged.'
    for family in ('thalamus', 'hippo-amygdala'):
        state_path = args.directory / f'state-{family}.public.json'
        score_path = args.directory / f'score-{family}.public.json'
        state = encoding.load_report(state_path)
        score = encoding.load_report(score_path)
        result['report_sha256'].update({path.name: checksum(path) for path in (state_path, score_path)})
        for name in state['trajectories']['baseline']['fits']:
            old = state['trajectories']['baseline']['initialization'][name]
            new = state['trajectories']['candidate']['initialization'][name]
            item = {'alignment_matrix_exact': old['atlas_to_native_voxel'] == new['atlas_to_native_voxel'],
                    'alignment_dice_exact': old['alignment_dice'] == new['alignment_dice'],
                    'phases': {}, 'jacobian_and_displacement': {}, 'postprocess': {}}
            for arm, init in (('baseline', old), ('candidate', new)):
                sg = init['segmentation_fit']
                item['jacobian_and_displacement'][arm] = {key: sg[key] for key in (
                    'min_jacobian', 'mean_displacement_voxels', 'p95_displacement_voxels', 'seconds')}
                item['jacobian_and_displacement'][arm]['intensity_final_min_jacobian'] = (
                    state['trajectories'][arm]['fits'][name]['min_jacobian'])
            for phase in ('synthetic', 'intensity'):
                old_stages = (old['segmentation_fit']['mesh_solver']['stages'] if phase == 'synthetic' else
                              state['trajectories']['baseline']['fits'][name]['optimization_stats']['stages'])
                new_stages = (new['segmentation_fit']['mesh_solver']['stages'] if phase == 'synthetic' else
                              state['trajectories']['candidate']['fits'][name]['optimization_stats']['stages'])
                assert len(old_stages) == len(new_stages)
                paired = []
                for before, after in zip(old_stages, new_stages):
                    first, second = before['objective_history'], after['objective_history']
                    assert len(first) == before['mesh_steps'] + 1
                    assert len(second) == after['mesh_steps'] + 1
                    common = min(len(first), len(second))
                    difference = next((i for i in range(common) if first[i] != second[i]), None)
                    paired.append({'baseline': stage_summary(before), 'candidate': stage_summary(after),
                                   'common_history_length': common,
                                   'first_recorded_scalar_difference_index': difference,
                                   'first_difference_baseline': first[difference] if difference is not None else None,
                                   'first_difference_candidate': second[difference] if difference is not None else None,
                                   'candidate_minus_baseline_at_first_difference': second[difference] - first[difference]
                                       if difference is not None else None})
                item['phases'][phase] = paired
            if 'component_selection' in old:
                for arm, init in (('baseline', old), ('candidate', new)):
                    component = init['component_selection']
                    removed = {label: count - component['retained_label_counts'][label]
                               for label, count in component['input_label_counts'].items()}
                    item['postprocess'][arm] = {
                        'rule': component['rule'], 'selection_components': component['selection_component_count'],
                        'raw_foreground_voxels': component['input_foreground_voxels'],
                        'retained_foreground_voxels': component['retained_original_foreground_voxels'],
                        'removed_voxels': sum(removed.values()),
                        'removed_fraction': sum(removed.values()) / component['input_foreground_voxels'],
                        'removed_counts_per_label': removed,
                        'new_labeled_voxels': component['new_labeled_voxels'],
                        'changed_label_ids': component['changed_label_ids']}
                item['postprocess']['interpretation'] = 'Saved counts prove class-to-background deletion, not inter-class relabeling. They do not identify spatial old/new argmax flips or posterior ties.'
            result['structures'][name] = item
        for old_group, new_group, changes in zip(score['baseline']['groups'], score['candidate']['groups'], score['regional_changes']):
            old_rows = {row['label']: row for pair in old_group['pairs'] if pair['kind'] == 'cross_method' for row in pair['regions']}
            new_rows = {row['label']: row for pair in new_group['pairs'] if pair['kind'] == 'cross_method' for row in pair['regions']}
            eligible = [row for row in changes['regions'] if row['dice_delta'] is not None]
            labels = set(changes['lost_old_passed_labels']) | {min(eligible, key=lambda row: row['dice_delta'])['label']}
            examples = []
            volume = old_group['grid']['voxel_volume_mm3']
            for label in sorted(labels):
                first, second = old_rows[label], new_rows[label]
                assert first['first_voxels'] == second['first_voxels']
                fields = ('second_voxels', 'intersection_voxels', 'different_voxels', 'dice',
                          'hard_volume_relative_to_official', 'second_soft_volume_mm3')
                examples.append({'label': label, 'name': first['name'], 'official_voxels': first['first_voxels'],
                                 'voxel_volume_mm3': volume, 'one_voxel_relative_volume_fraction':
                                     1 / first['first_voxels'] if first['first_voxels'] else None,
                                 'baseline': {key: first[key] for key in fields},
                                 'candidate': {key: second[key] for key in fields},
                                 'hard_volume_change_mm3': (second['second_voxels'] - first['second_voxels']) * volume})
            result['label_examples'][changes['id']] = examples
    right = result['structures']['hippo-amygdala-right']['phases']['synthetic'][0]['candidate']
    result['right_ha_early_stop'] = {'mesh_steps': right['mesh_steps'], 'limit': right['mesh_step_budget'],
                                    'last_three_relative_cost_changes': right['last_three_relative_cost_changes'],
                                    'relative_cost_patience_rule_excluded_at_stop': all(x >= 1e-6 for x in right['last_three_relative_cost_changes']),
                                    'source_inference': 'With one synthetic outer iteration, the observed 36<300 and these last three changes exclude the fast 1e-6 relative-cost patience stop; the source movement-stop branch remains. Saved stats do not identify zero displacement versus three <=0.005 voxel displacements.',
                                    'not_a_confirmed_bug': 'No native same-state trial/update/termination record is available.'}
    result['program_sha256'] = checksum(Path(__file__))
    args.output.write_text(encoding.compact(result) + '\n')
    print(json.dumps({'structures': list(result['structures']),
                      'right_ha_early_stop': result['right_ha_early_stop']}))


if __name__ == '__main__':
    main()
