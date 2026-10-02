"""统一锁内核对实际 logical cuda:0、CUDA UUID 与同 PID 的 NVML/SMI 物理设备身份。"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post-worker', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location('frozen_repeat_device_identity', args.post_worker)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    worker.require(worker.sha256(args.config) == args.config_sha256, 'frozen identity config changed')
    config = json.loads(args.config.read_text())
    source = worker.verify_source(config)
    worker.require(worker.sha256(args.post_worker) == config['tool_files'][str(args.post_worker.resolve())], 'identity worker changed')
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES') == config['gpu_uuid'] and
        os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', 'fixed visible UUID/allocator differs')
    worker.require(not args.output.exists(), 'identity report must be new')
    import torch
    worker.require(not torch.cuda.is_initialized(), 'identity CUDA initialized before lock')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'state': 'cpu_source_verified_waiting_lock', 'pid': os.getpid(),
        'scope': 'device identity only; context plus one-element allocation; no tracking/SIFT2/FA/matrices',
        'scientific_parity': 'not_assessed', 'source': source, 'config_sha256': args.config_sha256,
        'script_sha256': worker.sha256(Path(__file__)), 'expected_gpu_uuid': config['gpu_uuid'],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'logical_device': 'cuda:0',
        'gpu_lock': config['gpu_lock']}
    worker.atomic_json(args.output, report)
    waited = time.perf_counter()
    with Path(config['gpu_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report['lock_wait_seconds'] = time.perf_counter() - waited
        worker.require(not torch.cuda.is_initialized(), 'CUDA initialized while waiting')
        worker.verify_source(config)
        helper = worker.load_module(Path(config['monitor_script']), 'frozen_identity_gpu_monitor')
        monitor = helper.GPUProcessMonitor(torch, 'cuda:0', interval=.25, gpu_uuid=config['gpu_uuid']); monitor.start()
        scratch = None
        try:
            torch.cuda.init(); torch.cuda.reset_peak_memory_stats(0)
            raw_uuid = torch.cuda.get_device_properties(0).uuid
            report['torch_uuid'] = {'type': type(raw_uuid).__name__, 'repr': repr(raw_uuid), 'str': str(raw_uuid)}
            if type(raw_uuid).__name__ == '_CUuuid':
                report['torch_uuid']['uint8_bytes'] = list(raw_uuid.bytes)
            report['torch_uuid']['canonical'] = worker.canonical_gpu_uuid(raw_uuid)
            worker.require(report['torch_uuid']['canonical'] == config['gpu_uuid'], 'actual CUDA identity differs')
            worker.require(torch.cuda.current_device() == 0, 'actual logical device is not zero')
            scratch = torch.empty(1, device='cuda:0', dtype=torch.uint8)
            torch.cuda.synchronize()
            matched = []
            for _ in range(20):
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=3)
                matched = [dict(gpu_uuid=row[0].strip(), pid=int(row[1].strip())) for line in result.stdout.splitlines()
                           if len(row := line.split(',')) == 2 and row[1].strip() == str(os.getpid())]
                if matched:
                    break
                time.sleep(.1)
            report['same_pid_nvidia_smi'] = matched
            worker.require(len(matched) == 1 and matched[0]['gpu_uuid'] == config['gpu_uuid'],
                'same PID physical NVML/SMI mapping is unavailable or differs; no physical index guess')
            report['state'] = 'actual_logical_cuda0_and_physical_uuid_verified'
        except Exception as error:
            report['state'] = 'failed'; report['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            scratch = None
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            measured = monitor.finish()
            peaks = [torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0), measured['peak_process_tree_bytes']]
            report['memory'] = {'peak_cuda_allocated_bytes': peaks[0], 'peak_cuda_reserved_bytes': peaks[1],
                                'process_monitor': measured}
            report['memory_gate'] = worker.memory_gate(peaks, measured['failed_samples'])
            report['total_seconds_including_lock_wait'] = time.perf_counter() - start
            worker.atomic_json(args.output, report)
    worker.require(report['memory_gate'] == 'passed', 'actual identity memory observation failed gate')
    print(json.dumps({key: report[key] for key in ('state', 'torch_uuid', 'same_pid_nvidia_smi', 'memory_gate')}))


if __name__ == '__main__':
    main()
