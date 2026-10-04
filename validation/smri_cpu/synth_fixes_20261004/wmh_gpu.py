"""Complete real WMH inference under an unchanged 20,000,000,000 B cap.

Run under the task's shared GPU flock with an explicitly selected GPU UUID.
Private image arrays stay on the server. Timings on a busy shared GPU are
observations and cannot establish a performance guarantee.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

from fnit.wmh_synthseg import WMHSynthSeg
from fnit.wmh_synthseg.pipeline import _write_volumes_csv


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--crop', action='store_true')
    p.add_argument('--source-revision', required=True)
    p.add_argument('--diagnostic-hooks', action='store_true')
    p.add_argument('--reference-uncapped', action='store_true',
                   help='isolated original-code numerical reference only; no production budget claim')
    p.add_argument('--uncapped-control',action='store_true',help='also permit candidate source for an explicitly uncapped numerical control')
    p.add_argument('--profile-kernels',action='store_true',help='record actual cuDNN kernel names; these timings are diagnostics')
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('preserve prior attempts; select a fresh output directory')
    a.output_dir.mkdir(parents=True)
    if a.weights.stat().st_size != 790531383 or digest(a.weights) != '0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988':
        raise ValueError('checkpoint differs from fixed size/SHA manifest')
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    torch.cuda.device_count()
    torch.cuda.init()
    initial_memory = torch.cuda.mem_get_info(0)
    total = torch.cuda.get_device_properties(0).total_memory
    report = {'schema': 'fnit.wmh.real.gpu.memory_fix.v1',
              'source_revision': a.source_revision, 'crop': a.crop,
              'budget_bytes': None if a.reference_uncapped else 20_000_000_000,
              'purpose': ('uncapped candidate numerical control' if a.uncapped_control else 'uncapped old-code numerical reference') if a.reference_uncapped else '20GB candidate/baseline acceptance',
              'device_total_bytes': total,
              'initial_free_bytes': initial_memory[0],
              'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'input_sha256': digest(a.input), 'weight_sha256': digest(a.weights),
              'driver_sha256': digest(__file__), 'torch_version': torch.__version__,
              'timing_scope': 'constructor/API/save separately; synchronized network forwards nested in API',
              'diagnostic_hooks': a.diagnostic_hooks, 'network_calls': [], 'layer_memory': []}
    for name in ('model', 'pipeline', 'spatial'):
        module = __import__('fnit.wmh_synthseg.'+name, fromlist=[''])
        report[name+'_sha256'] = digest(module.__file__)
    samples = []
    stopped = threading.Event()
    def observe():
        while not stopped.is_set():
            started = time.monotonic()
            query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'monotonic': started, 'query_seconds': time.monotonic()-started,
                            'returncode': query.returncode, 'rows': query.stdout.strip().splitlines()})
            stopped.wait(.5)
    monitor = threading.Thread(target=observe, daemon=True)
    try:
        report['failure_stage'] = 'allocator_cap_initialization'
        if a.uncapped_control and not a.reference_uncapped:
            raise ValueError('uncapped control must explicitly request --reference-uncapped')
        if a.reference_uncapped and not a.uncapped_control and report['model_sha256'] != 'dd95cd0816c308ba5a14de4b5bd8ddbeaf640656515d0713bbeb5df11533bf6c':
            raise ValueError('uncapped numerical reference requires the unchanged baseline model')
        if not a.reference_uncapped:
            torch.cuda.set_per_process_memory_fraction(20_000_000_000/total, 0)
        monitor.start()
        report['failure_stage'] = 'model_constructor'
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model = WMHSynthSeg(weights=a.weights, device=device, threads=8)
        torch.cuda.synchronize(device)
        report['constructor_seconds'] = time.perf_counter()-started
        report['constructor_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        report['constructor_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        report['parameters_dtype'] = str(next(model.model.parameters()).dtype)
        report['parameters_device'] = str(next(model.model.parameters()).device)
        original = model.model.forward
        def measured(tensor):
            input_sha256=hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = original(tensor)
            torch.cuda.synchronize(device)
            report['network_calls'].append({'shape': list(tensor.shape), 'input_dtype': str(tensor.dtype), 'input_sha256':input_sha256,
                'output_dtype': str(out.dtype), 'seconds': time.perf_counter()-start,
                'allocated_bytes': torch.cuda.memory_allocated(device),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
            return out
        model.model.forward = measured
        if a.diagnostic_hooks:
            for name, layer in model.model.named_modules():
                if isinstance(layer, torch.nn.GroupNorm):
                    def before(module, tensors, _name=name):
                        report['layer_memory'].append({'layer': _name, 'phase': 'before',
                            'shape': list(tensors[0].shape), 'allocated_bytes': torch.cuda.memory_allocated(device),
                            'reserved_bytes': torch.cuda.memory_reserved(device)})
                    layer.register_forward_pre_hook(before)
        started = time.perf_counter()
        report['failure_stage'] = 'complete_inference'
        if a.profile_kernels:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                result = model(image=a.input, crop=a.crop, save_lesion_probabilities=True)
            def kernels(event):
                names=[k.name for k in event.kernels]
                for child in event.cpu_children:names+=kernels(child)
                return names
            report['actual_cudnn_convolutions']=[{'input_shapes':event.input_shapes,'kernels':kernels(event)} for event in prof.events() if event.name=='aten::cudnn_convolution']
        else:
            result = model(image=a.input, crop=a.crop, save_lesion_probabilities=True)
        torch.cuda.synchronize(device)
        report['api_seconds'] = time.perf_counter()-started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        report['policy'] = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                            'benchmark': torch.backends.cudnn.benchmark,
                            'deterministic': torch.backends.cudnn.deterministic,
                            'autocast_enabled': torch.is_autocast_enabled()}
        started = time.perf_counter()
        result.segmentation.save(a.output_dir/'seg.nii.gz')
        result.lesion_probability.save(a.output_dir/'lesion.nii.gz')
        _write_volumes_csv(result.volumes_mm3, 'anonymous_seg.nii.gz', a.output_dir/'volumes.csv')
        report['save_seconds'] = time.perf_counter()-started
        report['shape'] = list(result.segmentation.shape)
        report['affine'] = result.segmentation.affine.tolist()
        report['volume_values'] = result.volumes_mm3
        report['output_sha256'] = {name: digest(a.output_dir/name) for name in ('seg.nii.gz', 'lesion.nii.gz', 'volumes.csv')}
        report['status'] = 'complete'
        report.pop('failure_stage')
    except Exception as e:
        report['status'] = 'failed'
        report['exception_type'] = type(e).__name__
        report['exception'] = str(e)
        try:
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        except Exception as stats_error:
            report['peak_allocated_bytes'] = report['peak_reserved_bytes'] = None
            report['stats_error'] = str(stats_error)
        raise
    finally:
        stopped.set()
        if monitor.is_alive():
            monitor.join(timeout=5)
        report['device_samples'] = samples
        (a.output_dir/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('device_samples','layer_memory')}), flush=True)


if __name__ == '__main__':
    main()
