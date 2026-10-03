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
        if report.get('status') != 'complete' or report.get('subject') != 'CON08' or report.get('source_revision') != REVISION:
            raise ValueError('both whole APIs must already be complete under source1128')
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
    formal, cold = load(formal_path), load(cold_path)
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
        private_path = directory / 'report/files.private.json'
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
        if role == 'official_FS82_CPU' and private['configuration'] != cold_config:
            raise ValueError('cold private output configuration differs from its bound request')
        helper.check_cifti_axes(paths[role + '/dtseries'], cold_config['hcp_assets_dir'], 2.4)
    paths.update({'formal/report': formal_path, 'cold/report': cold_path,
                  'cold/binding': cold_root / 'binding.private.json', 'comparison': Path(__file__),
                  'formal/configuration': formal_config_path,
                  'metric_helper': args.metric_helper, 'cold_helper': args.cold_helper})
    protect_output(root, args.output_root, [source, cohort / 'raw', formal_root, cold_root,
                   cold_config['hcp_assets_dir'], formal_config_path,
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
              'scope': 'Same raw and verified formal-v4 volume; original FS8.2 CPU cold backend vs formal FNIT GPU full surface. Complete saved time series compared after both APIs, no MRI rerun, fitting, intensity normalization or native-vertex pairing. CPU/GPU MSM differences confound causal version attribution.',
              'reference_role': 'formal_FNIT_GPU', 'candidate_role': 'official_FS82_CPU',
              'data_metrics': metrics, 'comparison_seconds': time.perf_counter() - tick,
              'CIFTI_time_axis_exact': True, 'CIFTI_brain_models_exact': True,
              'GIFTI_CIFTI_cortical_values_exact': True, 'input_sha256_before': before,
              'input_sha256_after': after, 'input_guards_equal': True,
              'all_original_scientific_source_hashes_equal': True,
              'actual_source_hashes_before': source_before, 'actual_source_hashes_after': source_after,
              'original_saved_GIFTI_binding': {
                  'formal_FNIT_GPU': 'Original private file map and original CIFTI cortical values; formal whole report has no GIFTI byte SHA. Posthoc reads additionally hash the entire GIFTI before/after.',
                  'official_FS82_CPU': 'Original whole report L/R byte SHA and original CIFTI cortical values.'},
              'metric_helper_sha256': METRICS_SHA, 'comparison_sha256': before['comparison']}
    args.output_root.mkdir(mode=0o700, exist_ok=False)
    (args.output_root / 'report.public.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('CON08_SAVED_TIMESERIES_COMPARISON_COMPLETE', result['comparison_seconds'])


if __name__ == '__main__':
    main()
