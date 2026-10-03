"""只读 CPU 预检：真实源码/程序/原始数据/配置依赖；未完成 producer 合同明确等待。"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def inspect_program(worker, program, expected, environment):
    """Record a real read-only invocation, including a loader failure; never mark it ready."""
    worker.require(program.resolve() == Path(expected['path']).resolve() and
            os.access(program, os.X_OK) and worker.sha256(program) == expected['sha256'],
            'actual executable dependency differs from audited program')
    before = time.perf_counter()
    result = subprocess.run([str(program), '-version'], env=environment, capture_output=True, text=True)
    return {**expected, 'actual_version_output': (result.stdout + result.stderr).strip(),
            'returncode': result.returncode, 'readonly_invocation_seconds': time.perf_counter() - before,
            'readonly_invocation_ready': result.returncode == 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    path = Path(__file__).with_name('benchmark_connectome_raw_official.py')
    spec = importlib.util.spec_from_file_location('official_raw_dependency_preflight', path)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    worker.require(worker.sha256(args.config) == args.config_sha256, 'frozen source/config changed')
    config = json.loads(args.config.read_text())
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'this preflight must hide GPUs')
    for source, digest in config['source_files'].items():
        worker.require(Path(source).is_file() and worker.sha256(Path(source)) == digest, 'actual source/tool dependency changed')
        if source.endswith('.py'):
            compile(Path(source).read_text(), source, 'exec')
    helper = worker.reference_helper()
    import torch
    import numpy as np
    import nibabel as nib
    worker.require(not torch.cuda.is_initialized(), 'dependency inspection initialized CUDA')
    reference_path = Path(config['verified_reference_manifest'])
    worker.require(worker.sha256(reference_path) == config['verified_reference_manifest_sha256'], 'audited reference manifest changed')
    reference = json.loads(reference_path.read_text())
    worker.require(reference.get('execution_completed') is True and
            worker.sha256(Path(helper.__file__)) == reference['script_sha256'] and
            worker.sha256(Path(helper.__file__).parents[1] / 'connectome_repeat_common.py') == reference['helper_sha256'],
            'actual helper does not match already audited reference')
    programs = {}
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '8',
                   'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
    for name in helper.PROGRAMS:
        program = Path(config['mrtrix_bin']) / name
        expected = reference['programs'][name]
        record = inspect_program(worker, program, expected, environment)
        record['pinned_version_matches'] = (record['actual_version_output'] == reference['mrtrix_version']) if name == 'tckgen' else None
        if name == 'tckgen' and not record['pinned_version_matches']:
            record['readonly_invocation_ready'] = False
        programs[name] = record
    programs_ready = all(row['readonly_invocation_ready'] for row in programs.values())
    manifest_path = Path(config['raw_manifest'])
    worker.require(worker.sha256(manifest_path) == config['raw_manifest_sha256'], 'canonical raw manifest changed')
    manifest = json.loads(manifest_path.read_text())
    checked = {}
    cases = {}
    for case in manifest['cases']:
        identities = []
        for item in case['input_files']:
            source = Path(item['path'])
            if str(source) not in checked:
                worker.require(source.is_file() and worker.sha256(source) == item['sha256'], 'actual canonical raw file differs')
                record = {**item, 'size_bytes': source.stat().st_size, 'actual_sha256_verified': True}
                if source.name.endswith('.nii.gz'):
                    image = nib.load(source)
                    record['actual_image_header'] = {'shape': list(map(int, image.shape)),
                        'spacing': list(map(float, image.header.get_zooms())), 'affine': image.affine.tolist(),
                        'dtype': str(image.get_data_dtype())}
                checked[str(source)] = record
            identities.append(checked[str(source)])
        anatomy_path = Path(config['official_anatomy_root']) / case['case_id'] / 'consumer_contract.json'
        dwi_path = Path(config['official_dwi_root']) / case['case_id'] / 'consumer_contract.json'
        completed = all(p.is_file() and json.loads(p.read_text()).get('state') == 'completed' for p in (anatomy_path, dwi_path))
        row = {'raw_file_identities': identities, 'anatomy_contract_path': str(anatomy_path),
               'official_dwi_contract_path': str(dwi_path), 'actual_completed_contracts_present': completed,
               'required_dependency_ready': completed and programs_ready, 'state': 'waiting_actual_completed_contracts'}
        if completed:
            row['actual_consumer_preflight'] = worker.preflight(anatomy_path, dwi_path, case['case_id'])
            row['state'] = 'actual_completed_consumer_source_checked'
        cases[case['case_id']] = row
    worker.require(not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA')
    report = {'schema_version': 1, 'state': 'readonly_dependency_preflight_completed',
        'scientific_parity': 'not_assessed', 'scope': 'actual source/program/config/raw dependency audit; no tracking/GPU/derivative generation',
        'config_path': str(args.config.resolve()), 'config_sha256': args.config_sha256,
        'script_sha256': worker.sha256(Path(__file__)), 'source_files_sha256_verified': config['source_files'],
        'program_dependencies': programs, 'program_dependencies_ready': programs_ready,
        'raw_inputs_ready': True, 'raw_unique_file_count': len(checked),
        'raw_manifest_sha256': config['raw_manifest_sha256'], 'dataset': manifest['dataset'], 'snapshot': manifest['snapshot'],
        'license': manifest['license'], 'cases': cases,
        'required_dependency_ready': all(row['required_dependency_ready'] for row in cases.values()),
        'not_ready_reason': None if all(row['required_dependency_ready'] for row in cases.values()) else
                ('actual final official DWI/anatomy producer contracts not yet all completed; no fallback or synthetic contracts' if programs_ready else
                 'actual pinned program read-only invocation failed or version differs; see returncode/version output; producer readiness recorded per case'),
        'environment': {'host': platform.node(), 'python': platform.python_version(),
                        'torch': torch.__version__, 'numpy': np.__version__, 'nibabel': nib.__version__,
                        'cuda_initialized': torch.cuda.is_initialized(), 'cuda_visible_devices': ''},
        'seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.require(not args.output.exists(), 'actual preflight output must be new')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in
          ('program_dependencies_ready', 'raw_inputs_ready', 'required_dependency_ready', 'seconds')}))


if __name__ == '__main__':
    main()
