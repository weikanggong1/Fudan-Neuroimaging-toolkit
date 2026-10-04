"""为已复验10例生成当前FNIT基线和独立官方配置，不生成虚构候选。

输入：cohort_verified.json；私有bindings包含python/code_root/code_commit/
source_archive_sha256/weights/assets/native_bin_dir/fs_license/gpu_uuid及
official_home。输出新目录下10份baseline配置、10份official配置和plan.json。
影像保持原始scanner网格；运行配置仅引用路径，不读取或改变体素。
固定总线程4，前5例预初始化CUDA API，后5例CLI；正式候选须沿用每例方式。
不完整、重复被试、未复验输入或非816基线直接报错，不启动计算。
此函数是benchmark配置生成器，没有独立FreeSurfer等价命令。
"""
import argparse
import json
from pathlib import Path

BASELINE = '816e5610417a4c587caf321049438a9554139016'


def prepare(*, manifest_path, bindings_path, output_directory):
    """使用完整具名路径生成10例配置；输出每例运行方式和配置文件清单。"""
    manifest = json.loads(Path(manifest_path).read_text())
    binding = json.loads(Path(bindings_path).read_text())
    cases = manifest['cases']
    if len(cases) != 10 or len({(c['dataset'], c['subject']) for c in cases}) != 10:
        raise ValueError('exactly ten distinct frozen subjects required')
    if any(c['download_status'] != 'complete' or len(c['sha256']) != 64 for c in cases):
        raise ValueError('all ten inputs must be verified')
    required = ('python', 'code_root', 'code_commit', 'source_archive_sha256',
                'weights', 'assets', 'native_bin_dir', 'fs_license', 'gpu_uuid',
                'server_root', 'official_home')
    if any(not binding.get(k) for k in required):
        raise ValueError('missing runtime binding')
    if binding['code_commit'] != BASELINE or len(binding['source_archive_sha256']) != 64:
        raise ValueError('baseline source differs from frozen public commit')
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(binding['server_root'])
    rows = []
    for index, case in enumerate(cases):
        mode = 'initialized_cuda_api' if index < 5 else 'cli'
        config = {k: binding[k] for k in ('python', 'code_root', 'code_commit',
                  'source_archive_sha256', 'weights', 'assets', 'native_bin_dir',
                  'fs_license', 'gpu_uuid')}
        config.update(input=case['server_input'], input_sha256=case['sha256'],
                      output=str(root / 'baseline/subjects' / case['id']),
                      diagnostic_root=str(root / 'baseline/diagnostics' / case['id']),
                      device='cuda:0', threads=4, invocation=mode,
                      pipeline_kwargs={'hemisphere_workers': 2},
                      pipeline_cli_args=['--hemisphere-workers', '2'])
        official = {'case': case['id'], 'input': case['server_input'],
                    'input_sha256': case['sha256'], 'fs_home': binding['official_home'],
                    'python': binding['python'], 'gpu_uuid': binding['gpu_uuid'],
                    'output': str(root / 'official/subjects' / case['id']),
                    'fs_license': binding['fs_license'],
                    'subjects_dir': str(root / 'official/subjects'),
                    'diagnostic_root': str(root / 'official/diagnostics' / case['id']),
                    'threads': 4, 'itk_threads': 1, 'seed': 1234,
                    'code_version': 'FreeSurfer 8.2.0 d932c45',
                    'program_manifest': str(root / 'official_program_manifest.json')}
        for role, value in (('baseline', config), ('official', official)):
            (output / f'{role}_{case["id"]}.json').write_text(json.dumps(value, indent=2) + '\n')
        rows.append({'id': case['id'], 'input_sha256': case['sha256'],
                     'baseline_config': str(output / f'baseline_{case["id"]}.json'),
                     'official_config': str(output / f'official_{case["id"]}.json'),
                     'invocation': mode, 'status': 'prepared_not_run'})
    plan = {'baseline_commit': BASELINE, 'cases': rows, 'threads': 4,
            'lock': '/tmp/fnit-shared-benchmark.lock',
            'candidate': 'not_frozen; no candidate execution claimed',
            'overall_metric_equivalence': 'not_assessed'}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(manifest_path=args.manifest, bindings_path=args.bindings,
                     output_directory=args.output_directory)
    print(json.dumps({'status': 'prepared_not_run', 'cases': len(result['cases'])}))
