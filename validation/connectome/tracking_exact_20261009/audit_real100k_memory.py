"""Single full-input candidate memory audit; never an ABBA timing sample.

Reuse the frozen benchmark's source loading, real input loading, run_once,
CPU snapshot, allocator statistics and output digest. Defaults: 100000 seeds,
batch 8192, seed 0, eager arc. There is no tracking warmup or baseline call.

The report retains source/input hashes and memory samples, but no process ID,
physical GPU identity, raw nvidia-smi output, command arguments or file paths.
Process memory is sampled rather than a proven continuous upper bound.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time


EXPECTED_BENCHMARK_SHA256 = 'bc689db275c72321ea1dc616259f8b7b3d9fba8ef91fcbc19cbfcec721264ff2'
EXPECTED_CANDIDATE_SHA256 = '6ed4e6bedc45fa0eaf2b026541056ec2976eb7d205d397c8dff4f3e899047dbd'
EXPECTED_OUTPUT_SHA256 = '0bb811ee93cca10cd8d7c3995cea860fc9e8f21c69685a5e081f2ed415beca43'
ALLOCATOR_CAP_BYTES = 18_000_000_000
PROCESS_BUDGET_BYTES = 20_000_000_000
REQUIRED_PHASES = {'input_h2d', 'tracking_and_snapshot_d2h'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('benchmark-module', 'candidate-tracking', 'fod-module', 'fod', 'five-tissue', 'gmwmi', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0', help='Explicit CUDA device; CPU execution is not a memory gate')
    parser.add_argument('--n-seeds', type=positive_int, default=100000)
    parser.add_argument('--batch-size', type=positive_int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compile-arc', action='store_true', default=False,
                        help='Default false; enabling compilation makes this a different configuration from the required gate')
    parser.add_argument('--sample-interval-seconds', type=positive_float, default=.5)
    parser.add_argument('--query-timeout-seconds', type=positive_float, default=3.)
    parser.add_argument('--max-sample-gap-seconds', type=positive_float, default=2.,
                        help='Declared monitoring completeness threshold, including query delays; default 2 seconds')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.max_sample_gap_seconds < args.sample_interval_seconds:
        parser.error('--max-sample-gap-seconds must be at least the sample interval')
    return args


def parse_own_process_memory(stdout, own_process_id):
    """Discard all other rows immediately; never retain an identifier or text."""
    values = []
    malformed_rows = 0
    for row in csv.reader(io.StringIO(stdout)):
        if not row or not any(column.strip() for column in row):
            continue
        try:
            row_process_id = int(row[0].strip())
        except (ValueError, IndexError):
            malformed_rows += 1
            continue
        if row_process_id != own_process_id:
            continue
        if len(row) != 2:
            return {'status': 'malformed_own_process_row', 'used_bytes': None,
                    'matching_rows': len(values) + 1, 'malformed_rows': malformed_rows}
        try:
            memory_mib = Decimal(row[1].strip())
            if not memory_mib.is_finite() or memory_mib < 0:
                raise InvalidOperation
            values.append(int(memory_mib * 1_048_576))
        except (InvalidOperation, ValueError):
            return {'status': 'own_process_memory_unavailable', 'used_bytes': None,
                    'matching_rows': len(values) + 1, 'malformed_rows': malformed_rows}
    if not values:
        return {'status': 'process_not_listed', 'used_bytes': None, 'matching_rows': 0,
                'malformed_rows': malformed_rows}
    return {'status': 'ok' if len(values) == 1 and not malformed_rows else 'ambiguous_query_rows',
            'used_bytes': sum(values), 'matching_rows': len(values), 'malformed_rows': malformed_rows}


class OwnProcessMonitor:
    """Only this process on the selected physical device; daemon subprocesses."""

    def __init__(self, interval, query_timeout, started=None):
        self.interval = interval
        self.query_timeout = query_timeout
        self.started = time.perf_counter() if started is None else started
        self._own_process_id = os.getpid()
        self._physical_device = None
        self._phase = 'device_initialization'
        self._state_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._records_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.records = []

    def set_phase(self, phase):
        with self._state_lock:
            self._phase = phase

    def set_physical_device(self, identity):
        # Identity is needed for the query only; it is absent from all records.
        with self._state_lock:
            self._physical_device = identity

    def sample(self, marker=None):
        with self._query_lock:
            with self._state_lock:
                phase_start, target = self._phase, self._physical_device
            started = time.perf_counter()
            command = ['nvidia-smi']
            if target is not None:
                command += ['-i', target]
            command += ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']
            result_record = {'status': 'query_failed', 'used_bytes': None, 'matching_rows': None}
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=self.query_timeout, check=False)
                if result.returncode:
                    result_record.update({'status': 'query_nonzero_exit', 'returncode': result.returncode,
                                          'stderr_sha256': hashlib.sha256(result.stderr.encode('utf-8')).hexdigest()})
                else:
                    result_record = parse_own_process_memory(result.stdout, self._own_process_id)
            except subprocess.TimeoutExpired:
                result_record['status'] = 'query_timeout'
            except OSError as error:
                result_record.update({'status': 'query_os_error', 'exception_type': type(error).__name__,
                                      'errno': getattr(error, 'errno', None)})
            except Exception as error:
                result_record.update({'status': 'query_exception', 'exception_type': type(error).__name__})
            finished = time.perf_counter()
            with self._state_lock:
                phase_end = self._phase
            record = dict(result_record, elapsed_start_seconds=started - self.started,
                          elapsed_end_seconds=finished - self.started,
                          query_seconds=finished - started, phase_start=phase_start,
                          phase_end=phase_end, selected_device_filter_applied=target is not None,
                          marker=marker)
            with self._records_lock:
                record['sample_index'] = len(self.records)
                self.records.append(record)
            return record

    def _loop(self):
        due = time.perf_counter()
        while not self._stop.is_set():
            self.sample()
            due += self.interval
            now = time.perf_counter()
            if due < now:
                due = now
            if self._stop.wait(max(0., due - now)):
                break

    def start(self):
        self._thread = threading.Thread(target=self._loop, name='own-process-memory-monitor', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.query_timeout + 1.)
        return self._thread is None or not self._thread.is_alive()

    def summary(self, scope_start, scope_end, max_gap_allowed, thread_stopped):
        with self._records_lock:
            samples = [dict(row) for row in self.records]
        required = [row for row in samples if row['selected_device_filter_applied'] and
                    (row['phase_start'] in REQUIRED_PHASES or row['phase_end'] in REQUIRED_PHASES)]
        valid = [row for row in required if row['status'] == 'ok']
        all_selected_valid = [row for row in samples if row['selected_device_filter_applied'] and row['status'] == 'ok']
        failures = [row for row in required if row['status'] != 'ok']
        gaps = [later['elapsed_start_seconds'] - earlier['elapsed_start_seconds']
                for earlier, later in zip(samples, samples[1:])]
        required_gaps = [later['elapsed_start_seconds'] - earlier['elapsed_start_seconds']
                         for earlier, later in zip(required, required[1:])]
        required_max_gap = max(required_gaps) if required_gaps else None
        start_bracket = next((row for row in required if row['marker'] == 'before_input_h2d'), None)
        end_bracket = next((row for row in required if row['marker'] == 'after_snapshot_d2h'), None)
        bracketed = (scope_start is not None and scope_end is not None and start_bracket is not None
                     and end_bracket is not None and start_bracket['status'] == 'ok'
                     and end_bracket['status'] == 'ok'
                     and start_bracket['elapsed_end_seconds'] <= scope_start
                     and end_bracket['elapsed_start_seconds'] >= scope_end)
        phase_counts = {phase: sum(row['status'] == 'ok' and
                        (row['phase_start'] == phase or row['phase_end'] == phase) for row in required)
                        for phase in sorted(REQUIRED_PHASES)}
        complete = bool(bracketed and thread_stopped and not failures and all(phase_counts.values())
                        and required_max_gap is not None and required_max_gap <= max_gap_allowed)
        peak = max((row['used_bytes'] for row in valid), default=None)
        whole_peak = max((row['used_bytes'] for row in all_selected_valid), default=None)
        return {'scope': 'only this process on the selected physical CUDA device; other processes excluded',
                'unit_source': 'nvidia-smi used_memory MiB, converted with 1048576 bytes/MiB',
                'requested_interval_seconds': self.interval, 'query_timeout_seconds': self.query_timeout,
                'max_gap_allowed_seconds': max_gap_allowed, 'samples_count': len(samples),
                'required_scope_samples_count': len(required), 'required_scope_valid_samples_count': len(valid),
                'required_scope_failure_count': len(failures),
                'query_error_count': sum(row['status'].startswith('query_') for row in samples),
                'process_missing_count': sum(row['status'] == 'process_not_listed' for row in samples),
                'required_scope_process_missing_count': sum(row['status'] == 'process_not_listed' for row in required),
                'max_interval_start_to_start_seconds': max(gaps) if gaps else None,
                'required_scope_max_interval_start_to_start_seconds': required_max_gap,
                'max_query_seconds': max((row['query_seconds'] for row in samples), default=None),
                'required_scope_elapsed_start_seconds': scope_start, 'required_scope_elapsed_end_seconds': scope_end,
                'scope_bracketed_by_valid_samples': bool(bracketed), 'phase_valid_sample_counts': phase_counts,
                'thread_stopped': thread_stopped, 'coverage_complete_under_declared_gap_limit': complete,
                'required_scope_sampled_peak_used_bytes': peak,
                'required_scope_sampled_peak_used_gb': None if peak is None else peak / 1e9,
                'required_scope_sampled_peak_used_gib': None if peak is None else peak / (1 << 30),
                'whole_audit_selected_process_sampled_peak_bytes': whole_peak,
                'continuous_upper_bound_proven': False,
                'sampling_note': 'Short peaks between samples can be missed. Missing/error rows are retained, never treated as zero. CUDA-init rows may precede process registration; required H2D-through-snapshot rows must be valid.',
                'samples': samples}


def load_benchmark(path):
    name = '_fnit_memory_audit_frozen_benchmark'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('frozen benchmark import unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for name in ('load_sources', 'load_real_inputs', 'run_once', 'sync', 'memory_stats', 'output_summary'):
        if not callable(getattr(module, name, None)):
            raise ValueError('frozen benchmark helper unavailable')
    return module


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                     prefix=output.name + '.', suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2)
        stream.write('\n')
    temporary.replace(output)


def main(argv=None):
    args = parse_args(argv)
    sys.dont_write_bytecode = True
    started = time.perf_counter()
    options = {'n_seeds': args.n_seeds, 'lmax': 8, 'seed': args.seed, 'batch_size': args.batch_size,
               'arc_proposals': 16, 'max_length_mm': 250., 'min_length_mm': None, 'step_mm': None,
               'max_angle_degrees': 45., 'cutoff': .1, 'power': .5, 'compile_arc': args.compile_arc}
    report = {'schema_version': 1, 'status': 'running', 'purpose': 'single candidate full-process memory audit only',
              'audit_only_not_abba': True, 'speedup_assessed': False, 'warmup_calls': 0,
              'tracking_calls': 0, 'baseline_tracking_executed': False,
              'expected_output_sha256': EXPECTED_OUTPUT_SHA256,
              'expected_candidate_sha256': EXPECTED_CANDIDATE_SHA256,
              'expected_benchmark_sha256': EXPECTED_BENCHMARK_SHA256,
              'process_memory_budget_bytes': PROCESS_BUDGET_BYTES,
              'torch_allocator_cap_bytes': ALLOCATOR_CAP_BYTES,
              'options': dict(options), 'source_sha256': {}, 'input_files': {},
              'torch_allocator_peaks': {}, 'timing_seconds': {},
              'timing_note': 'This first-full-call audit includes monitoring subprocess overhead and is excluded from formal ABBA and speedup summaries.',
              'identifiers_and_paths_in_report': False}
    output = args.output.expanduser().resolve()
    monitor = None
    benchmark = None
    device = None
    cuda_initialized = False
    scope_start = None
    scope_end = None
    stage = 'source_validation'
    error = None
    try:
        for name in ('benchmark_module', 'candidate_tracking', 'fod_module', 'fod', 'five_tissue', 'gmwmi'):
            path = getattr(args, name).expanduser().resolve()
            if not path.is_file():
                raise ValueError('required input/source file unavailable')
            setattr(args, name, path)
        report['source_sha256'] = {'audit_script': file_sha256(Path(__file__)),
                                   'frozen_benchmark': file_sha256(args.benchmark_module),
                                   'candidate_tracking': file_sha256(args.candidate_tracking),
                                   'fixed_fod': file_sha256(args.fod_module)}
        if report['source_sha256']['frozen_benchmark'] != EXPECTED_BENCHMARK_SHA256:
            raise ValueError('frozen benchmark SHA mismatch')
        if report['source_sha256']['candidate_tracking'] != EXPECTED_CANDIDATE_SHA256:
            raise ValueError('candidate tracking SHA mismatch')
        benchmark = load_benchmark(args.benchmark_module)
        import torch
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('memory audit requires an explicit CUDA device')
        # This helper requires two modules; both are the same frozen candidate.
        # Only modules['candidate'] is ever executed, exactly once.
        modules, _ = benchmark.load_sources(args.candidate_tracking, args.candidate_tracking, args.fod_module)
        paths = {'fod': args.fod, 'five_tissue': args.five_tissue, 'gmwmi': args.gmwmi}
        stage = 'real_input_loading'
        phase_started = time.perf_counter()
        report['input_files'] = {name: {'sha256': benchmark.file_sha256(path), 'size_bytes': path.stat().st_size}
                                 for name, path in paths.items()}
        report['timing_seconds']['input_hashing'] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        tensors, affines, spacing = benchmark.load_real_inputs(paths)
        report['timing_seconds']['real_input_io'] = time.perf_counter() - phase_started
        report['input_shapes'] = {name: list(value.shape) for name, value in tensors.items()}
        report['input_dtypes'] = {name: str(value.dtype) for name, value in tensors.items()}
        report['affine_dtypes'] = {name: str(value.dtype) for name, value in affines.items()}
        options['five_tissue_spacing_mm'] = spacing
        report['options'] = dict(options)
        report['fixed_fod_function_identity_equal'] = True
        report['runtime'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
                             'cpu_interop_threads': torch.get_num_interop_threads()}
        save_report(report, output)
        monitor = OwnProcessMonitor(args.sample_interval_seconds, args.query_timeout_seconds, started)
        monitor.start()
        stage = 'cuda_initialization'
        phase_started = time.perf_counter()
        torch.cuda.set_device(device)
        torch.cuda.init()
        cuda_initialized = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < ALLOCATOR_CAP_BYTES:
            raise ValueError('selected device is smaller than required allocator cap')
        identity = getattr(properties, 'uuid', None)
        if identity is None:
            raise ValueError('selected physical device filter unavailable')
        monitor.set_physical_device('GPU-' + str(identity).removeprefix('GPU-'))
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / properties.total_memory, device)
        benchmark.sync(device)
        report['timing_seconds']['device_initialization'] = time.perf_counter() - phase_started
        report['device'] = {'logical_device': str(device), 'name': properties.name,
                             'total_memory_bytes': properties.total_memory}
        report['precision'] = {'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                               'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32), 'autocast': False,
                               'inputs': 'float32', 'affines': 'float64'}
        report['torch_allocator_cap_applied'] = True
        stage = 'input_h2d'
        monitor.set_phase(stage)
        monitor.sample(marker='before_input_h2d')
        scope_start = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats(device)
        phase_started = time.perf_counter()
        device_tensors = {name: value.to(device) for name, value in tensors.items()}
        device_affines = {name: value.to(device) for name, value in affines.items()}
        benchmark.sync(device)
        report['timing_seconds']['input_h2d'] = time.perf_counter() - phase_started
        # run_once resets peak counters: save the H2D peak before that reset.
        report['torch_allocator_peaks']['input_h2d'] = benchmark.memory_stats(device)
        monitor.sample(marker='after_input_h2d')
        inputs = {'wm_sh': device_tensors['fod'], 'fod_affine': device_affines['fod'],
                  'five_tissue': device_tensors['five_tissue'], 'five_tissue_affine': device_affines['five_tissue'],
                  'gmwmi': device_tensors['gmwmi']}
        stage = 'tracking_and_snapshot_d2h'
        monitor.set_phase(stage)
        monitor.sample(marker='before_full_tracking')
        report['tracking_calls'] = 1
        record, snapshot = benchmark.run_once(modules['candidate'], 'candidate', 'full_memory_audit', 0,
                                               inputs, options, device, trace_path=None)
        scope_end = time.perf_counter() - started
        monitor.sample(marker='after_snapshot_d2h')
        report['torch_allocator_peaks']['tracking'] = record['tracking_memory']
        report['torch_allocator_peaks']['tracking_and_snapshot_d2h'] = record['tracking_and_d2h_memory']
        report['timing_seconds'].update({'tracking_audit_only': record['tracking_seconds'],
                                         'tracking_process_cpu': record['tracking_process_cpu_seconds'],
                                         'path_pack': record['path_pack_seconds'], 'snapshot_d2h': record['d2h_seconds']})
        report['snapshot_d2h_completed'] = True
        stage = 'output_hash_validation'
        monitor.set_phase(stage)
        phase_started = time.perf_counter()
        summary = benchmark.output_summary(snapshot)
        report['timing_seconds']['output_hashing'] = time.perf_counter() - phase_started
        report['output'] = {name: summary[name] for name in ('accepted_streamlines', 'total_path_points',
                            'seeds_attempted', 'field_sha256', 'output_sha256', 'digest_format')}
        report['output_sha256_matches_expected'] = summary['output_sha256'] == EXPECTED_OUTPUT_SHA256
        monitor.sample(marker='after_output_hash')
    except BaseException as caught:
        error = caught
        report['execution_error'] = {'stage': stage, 'type': type(caught).__name__,
                                    'message': 'Details omitted to keep identifiers and filesystem paths out of the report.'}
    finally:
        if monitor is not None:
            if scope_start is not None and scope_end is None:
                scope_end = time.perf_counter() - started
            if error is not None:
                monitor.sample(marker='failure_final_sample')
            thread_stopped = monitor.stop()
            report['process_memory_monitor'] = monitor.summary(scope_start, scope_end,
                                                                args.max_sample_gap_seconds, thread_stopped)
        if benchmark is not None and cuda_initialized:
            try:
                report['torch_allocator_peaks']['at_audit_end'] = benchmark.memory_stats(device)
            except BaseException as caught:
                report['allocator_stats_error_type'] = type(caught).__name__
        peaks = report['torch_allocator_peaks']
        allocated = [value['peak_allocated_bytes'] for value in peaks.values() if value.get('peak_allocated_bytes') is not None]
        reserved = [value['peak_reserved_bytes'] for value in peaks.values() if value.get('peak_reserved_bytes') is not None]
        report['torch_allocator_peak_allocated_bytes'] = max(allocated) if allocated else None
        report['torch_allocator_peak_reserved_bytes'] = max(reserved) if reserved else None
        process = report.get('process_memory_monitor', {})
        peak = process.get('whole_audit_selected_process_sampled_peak_bytes')
        configuration_equal = args.n_seeds == 100000 and args.batch_size == 8192 and args.seed == 0 and not args.compile_arc
        allocator_ok = (bool(allocated) and bool(reserved) and max(allocated) <= ALLOCATOR_CAP_BYTES
                        and max(reserved) <= ALLOCATOR_CAP_BYTES and report.get('torch_allocator_cap_applied', False))
        process_ok = None if peak is None else peak < PROCESS_BUDGET_BYTES
        coverage_ok = process.get('coverage_complete_under_declared_gap_limit', False)
        digest_ok = report.get('output_sha256_matches_expected', False)
        passed = bool(error is None and configuration_equal and allocator_ok and process_ok and coverage_ok and digest_ok)
        report['gate'] = {'required_configuration_matches': configuration_equal, 'tracking_completed': error is None and digest_ok,
                          'snapshot_d2h_completed': report.get('snapshot_d2h_completed', False),
                          'expected_output_digest_matches': digest_ok, 'torch_allocator_within_18gb_cap': allocator_ok,
                          'sampled_own_process_within_20gb': process_ok, 'sampling_coverage_complete': coverage_ok,
                          'passed_sampled_full_process_memory_gate': passed, 'continuous_memory_upper_bound_proven': False}
        report['status'] = 'passed_sampled_memory_gate' if passed else 'execution_failed' if error is not None else 'memory_gate_not_passed'
        report['timing_seconds']['audit_wall_including_monitoring'] = time.perf_counter() - started
        save_report(report, output)
        print(json.dumps({'status': report['status'], 'output_sha256_matches_expected': digest_ok,
                          'torch_peak_allocated_bytes': report['torch_allocator_peak_allocated_bytes'],
                          'torch_peak_reserved_bytes': report['torch_allocator_peak_reserved_bytes'],
                          'sampled_own_process_peak_bytes': peak, 'monitor_coverage_complete': coverage_ok}), flush=True)
    return 0 if report['gate']['passed_sampled_full_process_memory_gate'] else 3 if error is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
