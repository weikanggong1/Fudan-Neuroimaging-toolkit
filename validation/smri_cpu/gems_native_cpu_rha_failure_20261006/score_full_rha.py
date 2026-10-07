"""Posthoc only: fixed-grid RHA Dice/volumes after the full FNIT recipe."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public(value):
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items() if key not in {'host', 'hostname', 'device_UUID'}}
    if isinstance(value, list):
        return [public(item) for item in value]
    return Path(value).name if isinstance(value, str) and value.startswith('/') else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-run', 'official-root', 'like', 'bindings', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve old scores; choose a fresh output directory')
    args.output.mkdir(mode=0o700)
    binding = json.loads(args.bindings.read_text())
    root = Path(__file__).parent
    for name, expected in binding['score_helpers_sha256'].items():
        if sha(root / name) != expected:
            raise RuntimeError('fixed-grid score helper changed: ' + name)
    for name, item in binding['official_final_reference_for_posthoc_scoring'].items():
        path = args.official_root / 'hippo-amygdala' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('saved official reference changed: ' + name)
    run = json.loads((args.candidate_run / 'summary.private.json').read_text())
    if run['source_sha256'] != binding['GEMS_sources_sha256'] or run['plan_sha256'] != binding['full_plan_sha256']:
        raise RuntimeError('completed full recipe source/input plan changed')
    if run['point_dtype'] != 'torch.float64' or run['Gaussian_dtype'] != 'torch.float64':
        raise RuntimeError('full recipe CPU dtype report mismatch')
    artifact = args.candidate_run / 'artifacts'
    for name, item in run['output_files'].items():
        path = args.candidate_run / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('full recipe output changed: ' + name)
    api = json.loads((artifact / 'report.json').read_text())
    if api['structures'] != ['hippo-amygdala-right']:
        raise RuntimeError('this acceptance is exactly one right HA recipe')
    audit = load_module(root / 'analyze_repeatability.py', 'fixed_RHA_grid_audit')
    scorer = load_module(root / 'score_stage.py', 'fixed_RHA_score')
    started = perf_counter()
    score = scorer.score(artifact, args.official_root, args.like, audit)
    wall = perf_counter() - started
    rows = []
    for group in score['groups']:
        voxel_volume = group['grid']['voxel_volume_mm3']
        for pair in group['pairs']:
            if pair['kind'] != 'cross_method':
                continue
            for region in pair['regions']:
                official_soft = region['first_soft_volume_mm3']
                actual_soft = region['second_soft_volume_mm3']
                rows.append({
                    'space': group['space'], 'label': region['label'], 'name': region['name'],
                    'official_voxels': region['first_voxels'], 'FNIT_voxels': region['second_voxels'],
                    'dice': region['dice'],
                    'official_hard_volume_mm3': region['first_voxels'] * voxel_volume,
                    'FNIT_hard_volume_mm3': region['second_voxels'] * voxel_volume,
                    'hard_volume_relative_to_official': region['hard_volume_relative_to_official'],
                    'official_soft_volume_mm3': official_soft, 'FNIT_soft_volume_mm3': actual_soft,
                    'soft_volume_relative_to_official': (abs(actual_soft - official_soft) / official_soft
                        if official_soft is not None and official_soft > 0 and actual_soft is not None else None),
                    'passed': region['preexisting_gate_passed'], 'nonempty': region['nonempty'],
                })
    result = {
        'status': 'full_recipe_scored_after_fitting',
        'scope': run['scope'],
        'threshold': {'per_region_Dice_minimum': .95, 'hard_volume_relative_to_official_maximum': .05},
        'both_empty_Dice': None,
        'all_nonempty_regions_passed': all(row['passed'] for row in rows if row['nonempty']),
        'regional_groups': {group['id']: group['gate_summary'] for group in score['groups']},
        'regions': rows, 'fixed_grid_audit': public(score), 'score_seconds': wall,
        'full_run_scalar_receipt': public(run),
        'fit_min_jacobians': api.get('fit_min_jacobians'),
        'full_recipe_configuration_and_stages': public(api['initialization']),
        'like_sha256': sha(args.like), 'bindings_sha256': sha(args.bindings),
        'scoring_program_sha256': sha(__file__),
        'official_outputs_read_only_after_fitting': True,
        'same_node_official_wall_time_denominator': 'unavailable; this test reuses saved official outputs',
    }
    (args.output / 'FULL_RHA_RESULTS.public.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'REGIONS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in ['status', 'regional_groups', 'all_nonempty_regions_passed', 'score_seconds']}))


if __name__ == '__main__':
    main()
