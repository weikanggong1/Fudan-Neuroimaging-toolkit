"""Summarize all regions, stored states and resource observations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'encoding', Path(__file__).parent.parent / 'gems_fixes_20261004/report_encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    result = {'scope': 'Fixed public sub-02 checkpoint, CPU old/new recipes; no raw T1 pipeline equivalence',
              'decision': 'Do not adopt mixed candidate as default: lost old passing thalamic regions and many remaining failures',
              'gpu': 'CUDA formula unchanged, epsilon still missing; no new whole GPU run for rejected CPU candidate',
              'timing_interpretation': 'Cold process wall and API observations under varying shared CPU load; no causal speed ratio',
              'source': 'Frozen v1 with accepted Gaussian fix; actual core/rasterize bindings in binding/source/recovery reports',
              'groups': {}, 'families': {}, 'inputs_sha256': {}}
    for family in ('brainstem', 'thalamus', 'hippo-amygdala'):
        score_path = args.directory / f'score-{family}.public.json'
        state_path = args.directory / f'state-{family}.public.json'
        score = encoding.load_report(score_path)
        state = encoding.load_report(state_path)
        result['inputs_sha256'][score_path.name] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        result['inputs_sha256'][state_path.name] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        for group in score['regional_changes']:
            rows = group['regions']
            summary = {'old_gate': group['old_gate'], 'new_gate': group['new_gate'],
                       'lost_old_passed_regions': [row for row in rows if row['passed_before'] and not row['passed_after']],
                       'newly_passed_regions': [row for row in rows if not row['passed_before'] and row['passed_after']],
                       'extremes_and_counts': {}}
            for key in ('dice_delta', 'hard_volume_error_delta',
                        'soft_volume_absolute_error_delta_mm3', 'soft_volume_error_delta'):
                eligible = [row for row in rows if row.get(key) is not None]
                summary['extremes_and_counts'][key] = {
                    'minimum': min(eligible, key=lambda row: row[key]) if eligible else None,
                    'maximum': max(eligible, key=lambda row: row[key]) if eligible else None,
                    'negative_count': sum(row[key] < 0 for row in eligible),
                    'zero_count': sum(row[key] == 0 for row in eligible),
                    'positive_count': sum(row[key] > 0 for row in eligible)}
            result['groups'][group['id']] = summary
        summary = {'geometry_dtype_finite_gate_passed': state['geometry_dtype_finite_gate_passed'],
                   'parameter_storage_finite_gate_passed': state['parameter_storage_finite_gate_passed'],
                   'images': state['images'], 'parameters': state['parameters'], 'arms': {}}
        for arm, trajectory in state['trajectories'].items():
            record = trajectory['process_record']
            fields = ('status', 'hostname', 'started_utc', 'finished_utc', 'returncode', 'wall_seconds',
                      'cpu_affinity', 'load_before', 'load_after', 'maximum_sampled_tree_rss_bytes',
                      'maximum_sampled_tree_threads')
            arm_summary = {field: record.get(field) for field in fields}
            arm_summary['worker'] = trajectory['worker']
            arm_summary['fits'] = {}
            for fit_name, fit in trajectory['fits'].items():
                optimization = fit['optimization_stats']
                stages = []
                for stage in optimization.get('stages', [optimization]):
                    objectives = stage['objective_history']
                    excluded = {'objective_history'}
                    stages.append({**{key: value for key, value in stage.items() if key not in excluded},
                                   'objective_count': len(objectives), 'objective_first': objectives[0],
                                   'objective_last': objectives[-1]})
                arm_summary['fits'][fit_name] = {
                    'min_jacobian': fit['min_jacobian'],
                    'totals': {key: value for key, value in optimization.items()
                               if key not in {'stages', 'objective_history'}}, 'stages': stages}
            summary['arms'][arm] = arm_summary
        result['families'][family] = summary
    result['program_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(encoding.compact(result) + '\n')
    print(json.dumps({'group_count': len(result['groups']),
                      'all_geometry_storage_finite_checks': all(
                          value['geometry_dtype_finite_gate_passed'] and value['parameter_storage_finite_gate_passed']
                          for value in result['families'].values()),
                      'decision': result['decision']}))


if __name__ == '__main__':
    main()
