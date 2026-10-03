#!/usr/bin/env python3
"""真实已完成 subject 的旧/新完整统计 writer 回归；不重跑 MRI、不替代 PV。"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

ACTIVE_OUTPUT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verification_passed(modes, comparisons):
    expected_warning = {'category': 'RuntimeWarning', 'message': 'invalid value encountered in scalar divide',
                        'source_basename': 'segstats_wmparc_python.py', 'line': 181}
    return (set(comparisons) == {'aseg', 'wmparc'}
            and all(v['old_new_full_text_byte_equal'] and v['old_original_saved_full_text_byte_equal'] for v in comparisons.values())
            and modes['old']['writers']['aseg']['warnings'] == [expected_warning]
            and not modes['old']['writers']['wmparc']['warnings']
            and not any(v['warnings'] for v in modes['new']['writers'].values())
            and all(m['full_partial_volume_executed'] for m in modes.values()))


def validate_output_root(args):
    output = args.output_root.resolve()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError('Regression output must be a new independent directory')
    raw_parent = Path(os.path.commonpath([args.raw_t1w.resolve().parent, args.raw_bold.resolve().parent]))
    protected = [args.subject.resolve(), args.source.resolve(), raw_parent,
                 args.new_module.resolve().parent, args.candidate_report.resolve().parent,
                 args.source_manifest.resolve().parent, Path(__file__).resolve().parent]
    for directory in protected:
        if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError('Regression output overlaps protected MRI, raw input, report or source directory')
    return output


def worker(config, mode):
    cfg = json.loads(Path(config).read_text())
    source = Path(cfg['source'])
    target = source / 'src/fnit/recon_all/segstats_wmparc_python.py' if mode == 'old' else Path(cfg['new_module'])
    sys.path.insert(0, str(source / 'src'))
    package = importlib.import_module('fnit.recon_all')
    if Path(package.__file__).resolve() != (source / 'src/fnit/recon_all/__init__.py').resolve():
        raise ValueError('Wrong reconstruction package imported')
    name = 'fnit.recon_all.segstats_wmparc_python'
    if name in sys.modules or 'fnit.recon_all.segstats_aseg_python' in sys.modules:
        raise ValueError('Statistics modules imported before controlled source selection')
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    aseg = importlib.import_module('fnit.recon_all.segstats_aseg_python')
    if Path(module.__file__).resolve() != target.resolve() or aseg._statistics is not module._statistics:
        raise ValueError('Actual writer statistics source binding differs from selected module')
    directory = Path(cfg['output_root']) / mode
    directory.mkdir(exist_ok=False)
    results = {'selected_module_sha256': sha(module.__file__), 'module_path_matches_requested': True,
               'aseg_statistics_is_selected_function': True, 'writers': {}}
    private = {'loaded_module_file': str(Path(module.__file__).resolve()),
               'aseg_module_file': str(Path(aseg.__file__).resolve()), 'outputs': {}}
    for label, writer, lut in [('aseg', aseg.write_aseg_stats, cfg['aseg_lut']),
                               ('wmparc', module.write_wmparc_stats, cfg['wmparc_lut'])]:
        output = directory / (label + '.stats')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            started = time.perf_counter()
            writer(cfg['subject'], lut, output)
            elapsed = time.perf_counter() - started
        results['writers'][label] = {'complete_writer_wall_seconds': elapsed, 'sha256': sha(output),
            'warnings': [{'category': w.category.__name__, 'message': str(w.message),
                          'source_basename': Path(w.filename).name, 'line': w.lineno} for w in captured]}
        private['outputs'][label] = str(output)
    results['actual_partial_volume_compiled_signatures'] = [str(s) for s in module._partial_volume.signatures]
    results['full_partial_volume_executed'] = bool(module._partial_volume.signatures)
    save(directory / 'worker.public.json', results)
    save(directory / 'files.private.json', private)


def main():
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('old', 'new'))
    parser.add_argument('--config', type=Path)
    for argument in ('subject', 'source', 'new-module', 'candidate-report', 'source-manifest', 'raw-t1w', 'raw-bold', 'output-root'):
        parser.add_argument('--' + argument, type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.config, args.worker)
        return
    for name in ('subject', 'source', 'new_module', 'candidate_report', 'source_manifest', 'raw_t1w', 'raw_bold', 'output_root'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required')
    args.output_root = validate_output_root(args)
    args.output_root.mkdir(mode=0o700, exist_ok=False)
    ACTIVE_OUTPUT = args.output_root
    started = time.perf_counter()
    candidate = json.loads(args.candidate_report.read_text())
    if candidate['status'] != 'complete' or not candidate['source_unchanged_during_run']:
        raise ValueError('Candidate whole pipeline is not complete with unchanged source')
    expected = json.loads(args.source_manifest.read_text())
    paths = {'source/' + rel: args.source / rel for rel in expected}
    paths.update({'subject/' + str(p.relative_to(args.subject)): p for p in args.subject.rglob('*') if p.is_file()})
    paths.update({'raw/t1w': args.raw_t1w, 'raw/bold': args.raw_bold, 'new_module': args.new_module,
                  'candidate_report': args.candidate_report, 'source_manifest': args.source_manifest,
                  'regression_helper': Path(__file__)})
    cfg = {name: str(getattr(args, name)) for name in ('subject', 'source', 'new_module', 'output_root')}
    for label in ('aseg', 'wmparc'):
        text = (args.subject / 'stats' / (label + '.stats')).read_text()
        lut = Path(next(line[len('# ColorTable '):] for line in text.splitlines() if line.startswith('# ColorTable ')))
        cfg[label + '_lut'] = str(lut)
        paths[label + '_lut'] = lut
    before = {name: sha(path) for name, path in paths.items()}
    for rel, value in expected.items():
        if before['source/' + rel] != value:
            raise ValueError('Frozen source does not match actual whole-pipeline source manifest')
    for key in ('t1w', 'bold'):
        if before['raw/' + key] != candidate['input_sha256'][key]:
            raise ValueError('Raw input hash differs from completed whole pipeline')
    old_sha = before['source/src/fnit/recon_all/segstats_wmparc_python.py']
    new_sha = before['new_module']
    if old_sha != '86422fbb24a997189dc774a942458a3ef8fa8a553841813067fddb50ea9d3751' or new_sha != 'da987f95e5ea46893a8ba4ec5e97f63ac9d9bf9dd8943d77b5d5706af3044a27':
        raise ValueError('Unexpected old/new statistical source')
    config = args.output_root / 'config.private.json'
    save(config, cfg)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                       PYTHONPATH=str(args.source / 'src'))
    modes = {}
    for mode in ('old', 'new'):
        environment['NUMBA_CACHE_DIR'] = str(args.output_root / ('numba-cache-' + mode))
        tick = time.perf_counter()
        with (args.output_root / (mode + '.private.log')).open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), '--worker', mode, '--config', str(config)],
                                    env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Isolated full writer worker failed: ' + mode)
        modes[mode] = json.loads((args.output_root / mode / 'worker.public.json').read_text())
        modes[mode]['isolated_process_wall_seconds'] = time.perf_counter() - tick
        if modes[mode]['selected_module_sha256'] != (old_sha if mode == 'old' else new_sha):
            raise ValueError('Actually imported module SHA differs')
    after = {name: sha(path) for name, path in paths.items()}
    if before != after:
        raise ValueError('Source or actual MRI/input closure changed during regression')
    comparisons = {}
    for name in ('aseg', 'wmparc'):
        old = args.output_root / 'old' / (name + '.stats')
        new = args.output_root / 'new' / (name + '.stats')
        original = args.subject / 'stats' / (name + '.stats')
        comparisons[name] = {'old_new_full_text_byte_equal': old.read_bytes() == new.read_bytes(),
                             'old_original_saved_full_text_byte_equal': old.read_bytes() == original.read_bytes(),
                             'original_saved_sha256': before['subject/stats/' + name + '.stats']}
    passed = verification_passed(modes, comparisons)
    save(args.output_root / 'report.public.json', {
        'status': 'verified' if passed else 'failed_output_or_warning_contract', 'case_id': candidate['subject'],
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'candidate_source_revision': candidate['source_revision'],
        'scope': 'Real CON04 same-input complete FNIT aseg/wmparc writer old/new regression; full original partial-volume calculation, no MRI rerun or new official comparison',
        'cpu_threads_per_worker': 1, 'workers_concurrent': 1, 'cuda_visible_devices': '',
        'old_module_sha256': old_sha, 'new_module_sha256': new_sha, 'helper_sha256': before['regression_helper'],
        'all_frozen_source_and_subject_files_before_after_unchanged': True,
        'guarded_subject_file_count': sum(k.startswith('subject/') for k in before),
        'guarded_frozen_source_file_count': len(expected), 'input_source_sha256_before': before, 'input_source_sha256_after': after,
        'writers': modes, 'comparisons': comparisons, 'diagnostic_wall_seconds': time.perf_counter() - started,
        'timing_boundary': 'Each writer wall includes full input reading, full partial-volume calculation and table saving; first call per isolated source also includes Numba compilation. Diagnostic process/import/hash overhead is separate from production MRI clocks.'})
    print(json.dumps({'status': 'verified' if passed else 'failed_output_or_warning_contract',
                      'report_sha256': sha(args.output_root / 'report.public.json')}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ACTIVE_OUTPUT is not None and not (ACTIVE_OUTPUT / 'report.public.json').exists():
            save(ACTIVE_OUTPUT / 'report.public.json', {'status': 'failed', 'failure_type': type(exc).__name__,
                 'failure': str(exc), 'helper_sha256': sha(Path(__file__)), 'input_source_guards': 'not_completed',
                 'scope': 'Failed independent complete-writer regression; original MRI and frozen source untouched'})
        raise
