"""ICLS dispatch diagnostic. Generated systems are regression data, not MRI benchmark."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--voxels', type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    torch.manual_seed(8129)
    design = torch.randn((145, 47), dtype=torch.float64, device=device)
    constraints = torch.randn((302, 47), dtype=torch.float64, device=device)
    # Include an explicitly unconstrained zero voxel alongside difficult systems.
    signal = torch.randn((args.voxels, 145), dtype=torch.float64, device=device)
    signal[0] = 0
    modules = {key: load(path, key) for key, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    def call(key):
        return modules[key]._mrtrix_icls_batch(signal, design, constraints)
    report = {'data_kind': 'generated regression only; not real-data benchmark', 'torch': torch.__version__, 'device': str(device), 'voxels': args.voxels, 'source_sha256': {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in [('baseline',args.baseline),('candidate',args.candidate)]}, 'gpu_uuid': torch.cuda.get_device_properties(device).uuid if device.type == 'cuda' and hasattr(torch.cuda.get_device_properties(device), 'uuid') else None, 'runs': []}
    outputs = {}
    for key in ['baseline', 'candidate', 'candidate', 'baseline']:
        sync()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        outputs[key] = call(key)
        sync()
        report['runs'].append({'version': key, 'wall_s': time.perf_counter()-start, 'allocated_peak_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None, 'reserved_peak_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None})
    delta = (outputs['baseline'] - outputs['candidate']).abs()
    report['error'] = {'neq': int(torch.count_nonzero(delta)), 'max': float(delta.max()), 'rmse': float(delta.square().mean().sqrt()), 'p99': float(torch.quantile(delta.flatten(), .99))}
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
    report['profiles'] = {}
    for key in modules:
        with torch.profiler.profile(activities=activities) as prof:
            call(key); sync()
        report['profiles'][key] = [{'name': e.key, 'calls': e.count, 'cpu_self_us': e.self_cpu_time_total, 'cuda_self_us': e.self_device_time_total} for e in prof.key_averages()]
        prof.export_chrome_trace(str(Path(args.output).with_suffix(f'.{key}.trace.json')))
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'profiles'}, indent=2))
    assert report['error']['neq'] == 0, report['error']

if __name__ == '__main__': main()
