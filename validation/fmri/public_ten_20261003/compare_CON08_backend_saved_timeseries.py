"""只读比较 CON08 官方 FS8.2 CPU 与正式 FNIT GPU 的完整已保存表面时序。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REVISION = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
METRICS_SHA = '047b3505b970351d93a5f5aec99c9ea5a655b7ad0426a3e093543ec390c56f45'
COLD_HELPER_SHA = 'a19408d5b928227aa458b2d9480772c1b3820836508046461fbe3f9b0c49fc54'
RAW_T1 = '2cf6d2f2afc65cb7e87ef68ac95a645387ab5c397f93e080a6b25ac485b810b6'
RAW_BOLD = '24f4c4547182eb8267a455cfbee24593564e96ca97160da58037bbc7bbcbdd4b'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_hashes(source):
    return {path.relative_to(source).as_posix(): sha256(path)
            for path in sorted((source / 'src/fnit').rglob('*.py'))}


def import_fixed(path, name, expected):
    if sha256(path) != expected:
        raise ValueError('a named fixed comparison helper differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_reports(formal, cold):
    for report in (formal, cold):
        allowed = ('complete',) if report is formal else ('complete', 'complete_saved_outputs_verified_late')
        if report.get('status') not in allowed or report.get('subject') != 'CON08' or report.get('source_revision') != REVISION:
            raise ValueError('formal API and cold saved-output boundary must be verified under source1128')
    if any(formal.get(key) is not True for key in ('raw_inputs_unchanged', 'configuration_unchanged',
            'source_unchanged_during_run', 'driver_unchanged', 'provenance_guards_passed')):
        raise ValueError('formal whole source/input guards are incomplete')
    if any(cold.get(key) is not True for key in ('readonly_input_guards_equal', 'frozen_source_guards_equal', 'binding_guard_equal', 'volume_reused')):
        raise ValueError('cold whole source/input guards are incomplete')
    if (formal.get('input_sha256') != {'t1w': RAW_T1, 'bold': RAW_BOLD}
            or cold.get('raw_t1w_sha256') != RAW_T1 or cold.get('raw_bold_sha256') != RAW_BOLD
            or formal.get('frames') != 180 or formal.get('repetition_time') != 2.4
            or cold.get('frame_count') != 180 or cold.get('tr_seconds') != 2.4
            or formal.get('backend') != 'fnit' or cold.get('backend') != 'freesurfer'
            or cold.get('surface_device') != 'cpu' or cold.get('cuda_visible_devices') != ''
            or cold.get('torch_cuda_available') is not False or cold.get('volume_executed') is not False
            or cold.get('reconstruction_reused') is not False):
        raise ValueError('actual same-case raw/device/frame/backend scope differs')


def verify_late_report(original, late, original_path, files_path, source_path):
    """显式晚复核分支：原失败仍为失败，不能把丢失的 API 时钟恢复成成功。"""
    failure = late.get('original_driver_failure', {})
    if (original.get('status') != 'failed' or original.get('error_type') != 'TypeError'
            or original.get('runner_sha256') != COLD_HELPER_SHA
            or late.get('status') != 'complete_saved_outputs_verified_late'
            or late.get('original_driver_status') != 'failed'
            or failure.get('error_type') != 'TypeError'
            or failure.get('message') != 'Object of type PosixPath is not JSON serializable'
            or failure.get('phase') != 'save report/files.private.json after full API return and saved-output checks'
            or failure.get('original_runner_sha256') != COLD_HELPER_SHA
            or failure.get('original_report_sha256') != sha256(original_path)
            or 'full_api_seconds' not in late or late['full_api_seconds'] is not None
            or 'returned_api_total_seconds' not in late or late['returned_api_total_seconds'] is not None
            or any(late.get(key) is not True for key in ('original_source_guard_passed',
                'late_input_guards_equal', 'late_source_guards_equal', 'GIFTI_CIFTI_cortical_values_exact'))
            or late.get('late_input_sha256_before') != late.get('late_input_sha256_after')
            or late.get('source_manifest_sha256') != sha256(source_path)
            or late.get('late_private_filemap_sha256') != sha256(files_path)
            or late.get('original_private_filemap_saved') is not False
            or 'total' in late.get('saved_timing_seconds', {})
            or 'total_before_publication' not in late.get('saved_timing_seconds', {})):
        raise ValueError('the explicit late reporter recovery provenance or lost-clock boundary differs')
    if (late['original_driver_failure_wall_seconds'] != original['driver_through_saved_output_validation_seconds']
            or any(original.get(key) is not True for key in ('readonly_input_guards_equal',
                'frozen_source_guards_equal', 'binding_guard_equal'))
            or any(late[key] != original[key] for key in ('configuration_sha256', 'readonly_inputs_before',
                'readonly_inputs_after', 'raw_t1w_sha256', 'raw_bold_sha256'))):
        raise ValueError('the late boundary does not bind the unchanged original failed attempt')


def protect_output(root, output, protected):
    if output.exists() or output.is_symlink():
        raise FileExistsError('comparison output already exists')
    output = output.resolve()
    if not output.is_relative_to((root / 'runs').resolve()) or output == (root / 'runs').resolve():
        raise ValueError('comparison output must be a new unified runs child')
    for path in protected:
        path = Path(path).resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('comparison output overlaps a protected original entity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root', type=Path, required=True)
    parser.add_argument('--cold-run', type=Path, required=True)
    parser.add_argument('--metric-helper', type=Path, required=True)
    parser.add_argument('--cold-helper', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--late-report', type=Path)
    parser.add_argument('--late-files', type=Path)
    parser.add_argument('--late-source-manifest', type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('read-only posthoc must hide CUDA before Python starts')
    root = args.fnit_root.resolve()
    index = load(root / 'INDEX.json')['active_tasks']['fmri_surface_backends_20261003']
    if index['frozen_source_revision'] != REVISION:
        raise ValueError('current index differs from the actual scientific source')
    cohort = root / 'workspaces/fnit_surface_ten_public_20261003'
    source = cohort / 'source_1128bc52'
    formal_root = (cohort / 'candidate_v4/CON08').resolve()
    cold_root = args.cold_run.resolve()
    formal_path = formal_root / 'report/report.public.json'
    cold_path = cold_root / 'report/report.public.json'
    formal, original_cold = load(formal_path), load(cold_path)
    late_args = (args.late_report, args.late_files, args.late_source_manifest)
    if any(late_args) and not all(late_args):
        raise ValueError('all three explicit late binding paths must be provided together')
    cold = original_cold
    if all(late_args):
        cold = load(args.late_report)
        verify_late_report(original_cold, cold, cold_path, args.late_files, args.late_source_manifest)
    elif original_cold.get('status') != 'complete':
        raise ValueError('an original failed attempt requires explicit independently verified late bindings')
    verify_reports(formal, cold)
    formal_config_path = cohort / 'configs_candidate_v4/CON08.json'
    formal_config, cold_config = load(formal_config_path), load(cold_root / 'config.private.json')
    if (sha256(formal_config_path) != formal['configuration_sha256']
            or sha256(cold_root / 'config.private.json') != cold['configuration_sha256']
            or formal_config.get('device') != 'cuda:0'
            or formal_config['hcp_assets_dir'] != cold_config['hcp_assets_dir']):
        raise ValueError('original whole configuration or GPU/CPU resource selection differs')
    binding = load(cold_root / 'binding.private.json')
    if (cold['readonly_inputs_before'] != cold['readonly_inputs_after']
            or cold['readonly_inputs_before'] != binding['hashes_before']):
        raise ValueError('original cold guard manifest differs from its saved whole report')
    paths = {key: Path(path) for key, path in binding['paths'].items()}
    if {key: sha256(path) for key, path in paths.items()} != binding['hashes_before']:
        raise ValueError('an original input/native/resource/configuration changed after the cold API')
    formal_sources = load(formal_root / 'report/source.private.json')
    if (sha256(formal_root / 'report/source.private.json') != formal['source_sha256']
            or formal_sources != binding['source_hashes_before']):
        raise ValueError('original formal and cold scientific sources differ')
    if args.late_source_manifest and load(args.late_source_manifest) != formal_sources:
        raise ValueError('late saved-output scientific source manifest differs from the original producer')
    source_before = source_hashes(source)
    if source_before != formal_sources:
        raise ValueError('actual scientific source changed before imports or comparison')
    sys.path.insert(0, str(source / 'src'))
    metric = import_fixed(args.metric_helper, 'fixed_saved_surface_metrics', METRICS_SHA)
    helper = import_fixed(args.cold_helper, 'fixed_cold_surface_check', COLD_HELPER_SHA)
    import nibabel as nib
    import numpy as np
    selected = {}
    for role, directory, report in (('formal_FNIT_GPU', formal_root, formal), ('official_FS82_CPU', cold_root, cold)):
        private_path = args.late_files if role == 'official_FS82_CPU' and args.late_files else directory / 'report/files.private.json'
        private = load(private_path)
        selected[role] = private.get('result', private)
        paths[role + '/private_binding'] = private_path
        checks = report['output_checks'] if role == 'formal_FNIT_GPU' else report['outputs']
        for key in ('left', 'right', 'dtseries'):
            path = Path(selected[role][key]).resolve(strict=True)
            if not path.is_relative_to(directory / 'derivatives') or not path.is_file():
                raise ValueError('saved output file map escapes its original derivatives')
            paths[role + '/' + key] = path
        if sha256(paths[role + '/dtseries']) != checks['dtseries']['sha256']:
            raise ValueError('actual saved CIFTI differs from its original whole report')
        if role == 'official_FS82_CPU':
            for key, saved_key in (('left', 'L'), ('right', 'R')):
                if sha256(paths[role + '/' + key]) != checks[saved_key]['sha256']:
                    raise ValueError('actual cold GIFTI differs from its original whole report')
        if role == 'official_FS82_CPU' and args.late_files:
            if (private.get('late_generated_filemap') is not True or private.get('original_private_filemap_saved') is not False
                    or private.get('original_failed_report_sha256') != sha256(cold_path)
                    or private.get('original_binding_sha256') != sha256(cold_root / 'binding.private.json')):
                raise ValueError('late file map is not explicitly bound to this unchanged original failed attempt')
            if (sha256(Path(selected[role]['metadata'])) != cold['metadata_sha256']
                    or sha256(Path(selected[role]['qc_report'])) != cold['qc_sha256']):
                raise ValueError('actual saved metadata/QC differs from the late validation binding')
        if role == 'official_FS82_CPU' and private['configuration'] != cold_config:
            raise ValueError('cold private output configuration differs from its bound request')
        helper.check_cifti_axes(paths[role + '/dtseries'], cold_config['hcp_assets_dir'], 2.4)
    paths.update({'formal/report': formal_path, 'cold/report': cold_path,
                  'cold/binding': cold_root / 'binding.private.json', 'comparison': Path(__file__),
                  'formal/configuration': formal_config_path,
                  'metric_helper': args.metric_helper, 'cold_helper': args.cold_helper})
    if args.late_report:
        paths.update({'cold/late_report': args.late_report, 'cold/late_files': args.late_files,
                      'cold/late_source_manifest': args.late_source_manifest,
                      'cold/original_failure_log': cold_root / 'report/failure.private.txt'})
        if (sha256(paths['cold/original_failure_log']) != cold['original_driver_failure']['failure_log_sha256']
                or cold['late_input_sha256_before']['original_failed_report'] != sha256(cold_path)
                or cold['late_input_sha256_before']['original_binding'] != sha256(cold_root / 'binding.private.json')
                or cold['late_input_sha256_before']['original_failure_log'] != sha256(paths['cold/original_failure_log'])):
            raise ValueError('the preserved original failure/report/binding differs from the late validation')
        if any(sha256(Path(path)) != digest for key, digest in cold['late_input_sha256_before'].items()
               for path in [binding['paths'].get(key)] if path is not None):
            raise ValueError('original inputs differ from the late validation before comparison')
    protect_output(root, args.output_root, [source, cohort / 'raw', formal_root, cold_root,
                   cold_config['hcp_assets_dir'], formal_config_path,
                   *[path.parent for path in late_args if path],
                   args.metric_helper.parent, args.cold_helper.parent,
                   *[path.parent for key, path in paths.items() if key.startswith('native/')],
                   *[path for key, path in paths.items() if key.startswith('resources/')]])
    before = {key: sha256(path) for key, path in paths.items()}
    first = nib.load(str(paths['formal_FNIT_GPU/dtseries']))
    second = nib.load(str(paths['official_FS82_CPU/dtseries']))
    if any(first.header.get_axis(i) != second.header.get_axis(i) for i in (0, 1)):
        raise ValueError('actual full time axis or ordered 21 brain models differ')
    tick = time.perf_counter()
    a, b = np.asarray(first.dataobj), np.asarray(second.dataobj)
    metrics = {'dtseries': metric.data_metrics(a, b), 'brain_models': {}}
    for name, selection, _ in first.header.get_axis(1).iter_structures():
        metrics['brain_models'][name] = metric.data_metrics(a[:, selection], b[:, selection])
    for key, structure in (('left', 'CIFTI_STRUCTURE_CORTEX_LEFT'), ('right', 'CIFTI_STRUCTURE_CORTEX_RIGHT')):
        arrays = []
        for role, image, data in (('formal_FNIT_GPU', first, a), ('official_FS82_CPU', second, b)):
            gifti = nib.load(str(paths[role + '/' + key]))
            if len(gifti.darrays) != 180 or any(row.data.shape != (32492,) for row in gifti.darrays):
                raise ValueError('full fsLR32k GIFTI frame/vertex domain differs')
            values = np.stack([row.data for row in gifti.darrays])
            for name, selection, subset in image.header.get_axis(1).iter_structures():
                if name == structure and not np.array_equal(values[:, subset.vertex], data[:, selection]):
                    raise ValueError('saved GIFTI cortex values differ from their bound CIFTI')
            arrays.append(values)
        metrics[key] = metric.data_metrics(*arrays)
    after = {key: sha256(path) for key, path in paths.items()}
    source_after = source_hashes(source)
    if before != after or source_before != source_after:
        raise ValueError('original data/resources/native/source changed during read-only comparison')
    result = {'status': 'complete_saved_timeseries_comparison', 'subject': 'CON08', 'source_revision': REVISION,
              'scope': 'Same raw and verified formal-v4 volume; original FS8.2 CPU cold backend saved surface vs formal FNIT GPU full surface. Complete saved time series compared after explicit saved-output verification, no MRI rerun, fitting, intensity normalization or native-vertex pairing. CPU/GPU MSM differences confound causal version attribution.',
              'reference_role': 'formal_FNIT_GPU', 'candidate_role': 'official_FS82_CPU',
              'data_metrics': metrics, 'comparison_seconds': time.perf_counter() - tick,
              'CIFTI_time_axis_exact': True, 'CIFTI_brain_models_exact': True,
              'GIFTI_CIFTI_cortical_values_exact': True, 'input_sha256_before': before,
              'input_sha256_after': after, 'input_guards_equal': True,
              'all_original_scientific_source_hashes_equal': True,
              'actual_source_hashes_before': source_before, 'actual_source_hashes_after': source_after,
              'original_saved_GIFTI_binding': {
                  'formal_FNIT_GPU': 'Original private file map and original CIFTI cortical values; formal whole report has no GIFTI byte SHA. Posthoc reads additionally hash the entire GIFTI before/after.',
                  'official_FS82_CPU': 'Independent late validator L/R/CIFTI byte SHA and exact cortical value binding; original failed reporter never saved these output SHAs.' if args.late_report else 'Original whole report L/R byte SHA and original CIFTI cortical values.'},
              'metric_helper_sha256': METRICS_SHA, 'comparison_sha256': before['comparison']}
    if args.late_report:
        result.update(cold_original_driver_status='failed',
            cold_original_failed_report_sha256=sha256(cold_path), cold_late_report_sha256=sha256(args.late_report),
            cold_full_api_seconds=None, cold_returned_api_total_seconds=None,
            cold_original_driver_failure_wall_seconds=cold['original_driver_failure_wall_seconds'],
            cold_saved_prepublication_seconds=cold['saved_timing_seconds']['total_before_publication'],
            cold_late_output_validation_seconds=cold['late_validation_seconds'])
    args.output_root.mkdir(mode=0o700, exist_ok=False)
    (args.output_root / 'report.public.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('CON08_SAVED_TIMESERIES_COMPARISON_COMPLETE', result['comparison_seconds'])


if __name__ == '__main__':
    main()
