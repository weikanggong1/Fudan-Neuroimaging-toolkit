"""同一原始病例的独立官方/FNIT 全链矩阵 envelope；不重算矩阵、不放宽固定输入工具。

跨软件严格核对 canonical nodes 语义与 raw case 来源；两条独立链的 atlas 内容可以不同。
每条链自己的重复仍须使用相同 atlas。只比较已有四矩阵，不要求正式 CLI 保存 TCK。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .connectome_repeat_common import (METRIC_POLICY, NAMES, check_metadata, envelope,
        fnit_repeat_envelope, load_profiles, overall_status, pairwise, profile_metrics, seed_labels, sha256)
except ImportError:
    from connectome_repeat_common import (METRIC_POLICY, NAMES, check_metadata, envelope,
        fnit_repeat_envelope, load_profiles, overall_status, pairwise, profile_metrics, seed_labels, sha256)

FIELDS = ('count_relative_l1', 'sift2_fbc_relative_l1', 'count_support_dice', 'count_pearson',
          'mean_length_common_normalized_mae', 'mean_fa_common_normalized_mae')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_map(rows, *, verified=False):
    result = {}
    for item in rows:
        path = str(Path(item['path']).resolve())
        require(path not in result, 'duplicate original raw file provenance')
        if verified:
            require(item['actual_sha256'] == item['sha256'], 'FNIT raw file was not verified unchanged')
        result[path] = item['sha256']
    return result


def source_binding(manifest_path, manifest_sha256, case_id, official_manifest, fnit_reports):
    require(sha256(manifest_path) == manifest_sha256, 'canonical raw manifest changed')
    original = json.loads(Path(manifest_path).read_text())
    cases = [item for item in original['cases'] if item['case_id'] == case_id]
    require(len(cases) == 1, 'exactly one canonical original raw case required')
    expected = file_map(cases[0]['input_files'])
    for path, digest in expected.items():
        require(Path(path).is_file() and sha256(Path(path)) == digest, 'original raw source bytes changed')
    official = json.loads(Path(official_manifest).read_text())
    require(official.get('case_id') == case_id and official.get('state') == 'completed' and
            official.get('execution_completed') is True and
            official.get('scope') == 'independent official raw DWI plus fresh FS anatomy tracking/downstream',
            'completed independent official wholechain source required; fixed FNIT operator reference is another scope')
    binding = official['raw_case_binding']
    require(binding.get('manifest_sha256') == manifest_sha256 and binding.get('case_id') == case_id and
            binding.get('dataset') == original['dataset'] and binding.get('snapshot') == original['snapshot'] and
            binding.get('rawprep_canonical_dwi_coverage_verified') is True and
            binding.get('anatomy_raw_t1_verified') is True and file_map(binding['files']) == expected,
            'official wholechain original raw provenance differs from canonical')
    producer = Path(binding['official_rawprep_report'])
    require(sha256(producer) == binding['official_rawprep_report_sha256'], 'official rawprep provenance changed')
    rawprep = json.loads(producer.read_text())
    expected_rawprep = {key: value for key, value in expected.items() if not key.endswith('/dataset_description.json')}
    require(rawprep.get('completed') is True and not rawprep.get('error') and
            {str(Path(key).resolve()): value for key, value in rawprep['input_sha256'].items()} == expected_rawprep and
            rawprep.get('commands') and all(item['returncode'] == 0 for item in rawprep['commands']),
            'official rawprep original raw provenance/completion independently differs')
    planned, completed = official['commands'], official['completed_commands']
    require(planned and len(planned) == len(completed) and all(item['returncode'] == 0 and
            all(item.get(key) == value for key, value in plan.items()) for plan, item in zip(planned, completed)),
            'official planned/executed command contract differs or is incomplete')
    reports = []
    for path in fnit_reports:
        report = json.loads(Path(path).read_text())
        require(report.get('status') == 'completed' and report.get('exit_code') == 0 and
                report.get('case_id') == case_id, 'completed same-case FNIT GPU report required')
        require(file_map(report['raw_input_provenance']) == expected and
                file_map(report['input_verification'], verified=True) == expected and
                file_map(report['input_verification_after'], verified=True) == expected,
                'FNIT original raw input identities differ before/after actual pipeline')
        reports.append(report)
    return original, official, reports, {'raw_manifest_sha256': manifest_sha256, 'case_id': case_id,
        'dataset': original['dataset'], 'snapshot': original['snapshot'], 'license': original['license'],
        'canonical_raw_files': cases[0]['input_files'],
        'official_manifest_sha256': sha256(official_manifest),
        'fnit_gpu_report_sha256': [sha256(path) for path in fnit_reports],
        'status': 'actual_original_raw_file_sha_and_producer_before_after_verified',
        'scope': 'same raw case only; intermediate DWI/anatomy/FOD/FA/atlas may differ between independent software chains'}


def semantic_identity(official_items, fnit_items, profile):
    # Preserve the old strict check inside each arm. Cross-arm equality is a
    # separate nodes-semantic contract, never an atlas voxel equality claim.
    official_identity = check_metadata(official_items, profile + ':official')
    fnit_identity = check_metadata(fnit_items, profile + ':fnit')
    all_items = official_items + fnit_items
    first = all_items[0][1]
    directories = [str(Path(meta['directory']).resolve()) for _, meta in all_items]
    require(len(set(directories)) == len(directories), 'duplicate resolved run directories')
    require(first['node_rows'] is not None and first['atlas_sha256'] is not None,
            'wholechain comparison requires actual atlas and canonical node provenance')
    for arrays, meta in all_items:
        require(meta['node_rows'] == first['node_rows'] and meta['nodes'] == first['nodes'] and
                arrays['count'].shape == all_items[0][0]['count'].shape and meta['atlas_sha256'] is not None,
                f'{profile}: canonical index/original label/hemisphere/name semantic mismatch')
    return {'node_rows': {'status': 'verified_equal_semantics', 'columns':
            ['index', 'original_label', 'hemisphere', 'name'], 'nodes': first['nodes']},
            'official_within_arm': official_identity, 'fnit_within_arm': fnit_identity,
            'cross_arm_atlas_sha256': {'official': [meta['atlas_sha256'] for _, meta in official_items],
                                      'fnit': [meta['atlas_sha256'] for _, meta in fnit_items],
                'policy': 'independent wholechain atlas content recorded, not forced equal'},
            'scope': 'raw-case and ordered node semantics; does not satisfy fixed-input image identity'}


def compare(official_root, fnit_dirs, fnit_reports, manifest_path, manifest_sha256, case_id,
            *, fnit_seeds=None):
    official_manifest = Path(official_root) / 'reference_manifest.json'
    original, source, fnit_sources, binding = source_binding(
        manifest_path, manifest_sha256, case_id, official_manifest, fnit_reports)
    official_seeds = source['seeds']
    require(len(official_seeds) >= 3 and len(set(official_seeds)) == len(official_seeds) and
            len(fnit_dirs) == len(fnit_reports) and fnit_dirs, 'official/FNIT repeat counts incomplete')
    fnit_seeds = seed_labels(fnit_seeds, len(fnit_dirs), 'fnit')
    for index, report in enumerate(fnit_sources):
        command = report['command']
        require('--n-seeds' in command and int(command[command.index('--n-seeds') + 1]) ==
                source['parameters']['n_seed_attempts'], 'actual FNIT/official attempted seed counts differ')
        if fnit_seeds is not None:
            require('--seed' in command and int(command[command.index('--seed') + 1]) == fnit_seeds[index],
                    'FNIT seed label differs from actual pipeline command')
    official = [load_profiles(Path(official_root) / f'seed-{seed}') for seed in official_seeds]
    fnit = [load_profiles(Path(path)) for path in fnit_dirs]
    names = list(official[0])
    require(all(set(item) == set(names) for item in (*official, *fnit)), 'wholechain atlas sets differ')
    for seed, profiles in zip(official_seeds, official):
        require({name: meta for name, (_, meta) in profiles.items()} == source['outputs'][str(seed)]['profiles'],
                'official matrix/node/atlas provenance changed after actual run')
    for profiles, report in zip(fnit, fnit_sources):
        expected = report['outputs']['files']
        for arrays, meta in profiles.values():
            directory = Path(meta['directory'])
            for name, digest in meta['matrix_sha256'].items():
                candidates = [directory / f'{prefix}{name}.csv' for prefix in ('connectome_', 'candidate_', '')]
                path = next(path for path in candidates if path.is_file() and sha256(path) == digest)
                require(str(path.resolve()) in expected and expected[str(path.resolve())]['sha256'] == digest and
                        expected[str(path.resolve())]['size_bytes'] == path.stat().st_size,
                        'FNIT matrix not bound to actual completed output report')
            for name, digest in [('nodes.tsv', meta['nodes_tsv_sha256']), ('atlas_dwi.nii.gz', meta['atlas_sha256'])]:
                path = directory / name
                require(path.is_file() and str(path.resolve()) in expected and
                        expected[str(path.resolve())]['sha256'] == digest,
                        'FNIT atlas/node semantics not bound to actual output report')
    report = {'dataset': original['dataset'], 'snapshot': original['snapshot'], 'case_id': case_id,
        'scope': 'independent same-raw-case wholechain matrices, distinct from fixed-input operator envelope',
        'raw_case_binding': binding, 'n_seed_attempts': source['parameters']['n_seed_attempts'],
        'official_seeds': official_seeds, 'fnit_seeds': fnit_seeds,
        'metric_policy': {**METRIC_POLICY, 'scope': 'wholechain final matrices; includes upstream algorithm differences'},
        'profiles': {}, 'population_envelope_status': 'not_assessed',
        'population_reason': 'formal FNIT CLI does not retain TCK/FOD; no tractogram or population metrics fabricated'}
    for name in names:
        left, right = [item[name] for item in official], [item[name] for item in fnit]
        identity = semantic_identity(left, right, name)
        pairs = pairwise([item[0] for item in left], [item[0] for item in right], profile_metrics)
        ranges = {field: envelope([item['metrics'][field] for item in pairs['official']],
                  [item['metrics'][field] for item in pairs['cross']],
                  similarity=field in ('count_support_dice', 'count_pearson')) for field in FIELDS}
        self_ranges = {field: fnit_repeat_envelope([item['metrics'][field] for item in pairs['official']],
                  [item['metrics'][field] for item in pairs['fnit']],
                  similarity=field in ('count_support_dice', 'count_pearson')) for field in FIELDS}
        report['profiles'][name] = {'nodes': left[0][1]['nodes'], 'input_identity': identity,
            'pairwise': pairs, 'ranges': ranges, 'fnit_reproducibility_ranges': self_ranges,
            'comparison_counts': {key: len(value) for key, value in pairs.items()},
            'matrix_envelope_status': overall_status(list(ranges.values())),
            'fnit_reproducibility_status': overall_status(list(self_ranges.values())),
            'provenance': {'official': [meta for _, meta in left], 'fnit': [meta for _, meta in right]}}
    report['matrix_envelope_status'] = overall_status(
        [{'status': item['matrix_envelope_status']} for item in report['profiles'].values()])
    report['fnit_reproducibility_status'] = overall_status(
        [{'status': item['fnit_reproducibility_status']} for item in report['profiles'].values()])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--fnit', type=Path, nargs='+', required=True)
    parser.add_argument('--fnit-gpu-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--fnit-seeds', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.official_root, args.fnit, args.fnit_gpu_reports,
        args.raw_manifest, args.raw_manifest_sha256, args.case_id, fnit_seeds=args.fnit_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('matrix_envelope_status', 'fnit_reproducibility_status')}))


if __name__ == '__main__':
    main()
