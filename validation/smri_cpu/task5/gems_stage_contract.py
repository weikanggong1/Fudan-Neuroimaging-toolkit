"""Save fit parameters after the existing measured real-input stage worker."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import runpy
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--worker', type=Path, required=True)
    args, worker_args = parser.parse_known_args()
    import fnit
    original = fnit.segment_4_subregions
    results = []
    before = {}
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        before = {'device_name': torch.cuda.get_device_name(0),
                  'device_uuid': subprocess.check_output(
                      ['nvidia-smi', '--id', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0],
                       '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip(),
                  'allocated_bytes': torch.cuda.memory_allocated(0),
                  'reserved_bytes': torch.cuda.memory_reserved(0)}

    def retain(*function_args, **function_kwargs):
        result = original(*function_args, **function_kwargs)
        results.append(result)
        return result

    fnit.segment_4_subregions = retain
    sys.argv = [str(args.worker), *worker_args]
    runpy.run_path(str(args.worker), run_name='__main__')
    if len(results) != 1:
        raise ValueError('expected one completed subregion stage')
    result = results[0]
    output = result.output_files['report'].parent
    fits = {}
    arrays = {}
    for index, (structure, fit) in enumerate(sorted(result.structure_results.items())):
        for name, value in [('vertices', fit.vertices),
                            ('means', fit.gaussian_parameters.means),
                            ('covariances', fit.gaussian_parameters.covariances)]:
            arrays[f'fit{index}_{name}'] = value.detach().cpu().numpy()
        arrays[f'fit{index}_objectives'] = np.asarray(fit.objective_history)
        fits[structure] = {'index': index, 'min_jacobian': fit.min_jacobian,
                           'optimization_stats': fit.optimization_stats}
    np.savez_compressed(output / 'fit_parameters.private.npz', **arrays)
    (output / 'fit_contract.private.json').write_text(json.dumps(fits, indent=2) + '\n')
    metadata = {'schema': 'fnit.gems.stage.contract.v1', 'pipeline_completed': True,
                'parameter_export_after_worker_api_timer': True,
                'worker_sha256': hashlib.sha256(args.worker.read_bytes()).hexdigest(),
                'contract_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'torch_intraop': torch.get_num_threads(),
                'torch_interop': torch.get_num_interop_threads()}
    if before:
        metadata['gpu_before'] = before
        metadata['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        metadata['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        metadata['gpu_after_allocated_bytes'] = torch.cuda.memory_allocated(0)
        metadata['gpu_after_reserved_bytes'] = torch.cuda.memory_reserved(0)
    (output / 'stage_contract.public.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
