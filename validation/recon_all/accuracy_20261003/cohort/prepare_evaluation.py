"""生成冻结10例的三方隔离配置；仅写配置，不启动重建或比较。"""
import argparse
import hashlib
import json
from pathlib import Path

BASELINE = '816e5610417a4c587caf321049438a9554139016'
GPU = 'GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
REQUIRED = ('python', 'code_root', 'code_commit', 'source_archive_sha256',
            'weights', 'assets', 'native_bin_dir', 'fs_license', 'invocation')


def emit(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare(*, manifest_path, bindings_path, output_dir, server_root):
    manifest = json.loads(Path(manifest_path).read_text())
    bindings = json.loads(Path(bindings_path).read_text())
    cases = manifest['cases']
    if len(cases) != 10 or len({(c['dataset'], c['subject']) for c in cases}) != 10:
        raise ValueError('exactly ten frozen distinct subjects required')
    if any(c['download_status'] != 'complete' or not c['sha256'] for c in cases):
        raise ValueError('all inputs must be verified; failures must remain in cohort')
    for role in ('baseline', 'candidate'):
        for key in REQUIRED:
            if not bindings[role].get(key):
                raise ValueError(f'missing frozen {role}.{key}')
        if len(bindings[role]['source_archive_sha256']) != 64 or len(bindings[role]['code_commit']) != 40:
            raise ValueError('full source SHA256 and commit required')
    if bindings['baseline']['code_commit'] != BASELINE:
        raise ValueError('baseline differs from approved main')
    official = bindings['official']
    if official['version'] != 'FreeSurfer 8.2.0 d932c45':
        raise ValueError('official version differs')
    if not official.get('program_manifest_sha256') or len(official['program_manifest_sha256']) != 64:
        raise ValueError('official executable/asset SHA manifest must be frozen')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    server_root = Path(server_root)
    comparisons, runs = [], []
    for case in cases:
        run_paths = {}
        for role in ('baseline', 'candidate'):
            # 每种版本必须从同一原始T1及自己的空目录启动。
            config = {**bindings[role], 'input': case['server_input'], 'input_sha256': case['sha256'],
                      'output': str(server_root / role / 'subjects' / case['id']),
                      'diagnostic_root': str(server_root / role / 'diagnostics' / case['id']),
                      'device': 'cuda:0', 'threads': 4, 'gpu_uuid': GPU,
                      'pipeline_kwargs': {**bindings[role].get('pipeline_kwargs', {}), 'hemisphere_workers': 2}}
            filename = f'{role}_{case["id"]}.json'
            emit(output_dir / filename, config)
            run_paths[role] = str(server_root / 'cohort' / 'configs' / filename)
        ref = str(server_root / 'official' / 'subjects' / case['id'])
        row = {'id': case['id'], 'baseline_config': run_paths['baseline'],
               'candidate_config': run_paths['candidate'], 'official': ref,
               'official_code_version': official['version']}
        comparisons.append(row)
        # 一例一个比较进程，复用完整比较器，避免重复设置Torch interop线程。
        comparison = {**bindings['comparison'], 'threads': 4, 'cases': [row],
                      'data_sources': str(server_root / 'cohort' / 'cohort_verified.json')}
        emit(output_dir / f'comparison_{case["id"]}.json', comparison)
        official_command = [official['recon_all'], '-all', '-i', case['server_input'],
                            '-s', case['id'], '-sd', str(server_root / 'official' / 'subjects'),
                            '-openmp', '4']
        runs.append({'id': case['id'], 'input_sha256': case['sha256'], **run_paths,
                     'official_subject': ref, 'official_command': official_command,
                     'official_command_prerequisites': 'root验证固定程序/资产SHA、私有license、同机计时和线程总预算；不用-parallel'})
    emit(output_dir / 'comparison_all.json', {**bindings['comparison'], 'threads': 4,
         'data_sources': str(server_root / 'cohort' / 'cohort_verified.json'), 'cases': comparisons})
    emit(output_dir / 'evaluation_plan.json', {'schema': 'fnit-recon-ten-case-plan-v1',
         'baseline_commit': BASELINE, 'gpu_uuid': GPU, 'threads': 4, 'lock': '/tmp/fnit-shared-benchmark.lock',
         'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
         'bindings_sha256': hashlib.sha256(Path(bindings_path).read_bytes()).hexdigest(),
         'cases': runs, 'status': 'prepared_not_executed', 'overall_metric_equivalence': 'not_assessed'})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-root', type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(manifest_path=args.manifest, bindings_path=args.bindings,
                   output_dir=args.output, server_root=args.server_root)
    print(json.dumps({'status': 'prepared_not_executed', 'cases': len(rows)}))


if __name__ == '__main__':
    main()
