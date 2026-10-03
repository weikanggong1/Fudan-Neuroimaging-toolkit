"""真实十例同输入 tensor 修复；计时使用原函数，张量抓取另行运行。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import torch

from diagnose_tensor_arithmetic import checked, instrument, sha, summary


class MemorySampler:
    """Sample this process; these tensor calls spawn no CUDA child processes."""
    def __init__(self):
        self.stop = threading.Event(); self.samples = []; self.failures = 0
    def run(self):
        while not self.stop.is_set():
            started = time.perf_counter()
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=5)
                rows = [line.split(',') for line in output.splitlines()]
                own = [row for row in rows if row[0].strip() == str(os.getpid())]
                self.samples.append({'t': started, 'bytes': sum(int(row[1]) * 1024**2 for row in own),
                    'uuid': [row[2].strip() for row in own], 'all_compute_processes': output.splitlines()})
            except Exception:
                self.failures += 1
            self.stop.wait(.1)
    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start(); return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join()
    def result(self):
        return {'peak_bytes': max((r['bytes'] for r in self.samples), default=None), 'samples': self.samples,
            'failed_samples': self.failures, 'max_interval_s': float(max(np.diff([r['t'] for r in self.samples]), default=0)),
            'resolution_bytes': 1024**2, 'pid': os.getpid(), 'gpu_children': 'none; no GPU subprocess in tensor execution'}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-map', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--paired-cases', nargs='*', default=['CON01', 'CON03'])
    parser.add_argument('--abba-rounds', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('fresh output namespace required')
    args.output.mkdir(parents=True)
    mapping = json.loads(args.case_map.read_text())['actual_completed_case_map']
    modules = {name: load_module(path, 'tensor_' + name) for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    torch.set_num_threads(8); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = True
    report = {'scope': 'same official-corrected DWI/gradient/mask ten-case tensor isolation, not raw end-to-end',
        'host': socket.gethostname(), 'pid': os.getpid(), 'torch': torch.__version__,
        'precision': 'unchanged Float32 DWI/tensor image; Float64 solves; TF32 enabled; no iteration/mask/filter change',
        'sources': {name: {'path': str(path), 'sha256': sha(path)} for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]},
        'harness_sha256': sha(__file__), 'diagnostic_helper_sha256': sha(Path(__file__).with_name('diagnose_tensor_arithmetic.py')),
        'case_map_sha256': sha(args.case_map), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'cases': {},
        'timing_policy': 'full solver per call; boundary synchronize; identical resident inputs; ABBA without tensor instrumentation; warmups excluded; shared load retained'}
    for case in args.cases:
        entry = mapping[case]; contract_path = checked(entry['consumer_contract']); producer_path = checked(entry['modeling_report'])
        contract = json.loads(contract_path.read_text()); producer = json.loads(producer_path.read_text())
        if contract.get('state') != 'completed' or contract.get('execution_completed') is not True or producer.get('execution_completed') is not True:
            raise ValueError('actual completed consumer/producer required')
        files = contract['files']; paths = {key: checked(files[key]) for key in ['official_corrected_dwi', 'official_gradient_mrtrix', 'brain_mask', 'FA', 'principal_direction']}
        tensor_path = paths['FA'].parent / 'tensor.nii.gz'
        records = [r for r in producer['commands'] if str(tensor_path) in r.get('output_sha256', {})]
        if len(records) != 1 or records[0]['returncode'] != 0 or records[0]['output_sha256'][str(tensor_path)] != sha(tensor_path):
            raise ValueError('tensor bytes not bound to actual successful official command')
        image = nib.load(paths['official_corrected_dwi']); signal = image.get_fdata(dtype=np.float32)
        mask = np.asarray(nib.load(paths['brain_mask']).dataobj) > 0; gradient = np.loadtxt(paths['official_gradient_mrtrix'])
        references = {}; geometry = {}
        for name, path in [('fa', paths['FA']), ('direction', paths['principal_direction']), ('tensor', tensor_path)]:
            im = nib.load(path)
            if im.shape[:3] != image.shape[:3] or not np.allclose(im.affine, image.affine, rtol=0, atol=1e-6): raise ValueError('reference grid differs')
            references[name] = im.get_fdata(dtype=np.float32)
            geometry[name] = {'path': str(path), 'sha256': sha(path), 'storage_dtype': str(im.get_data_dtype()), 'storage_stride': list(np.asarray(im.dataobj).strides), 'affine_max_abs_header_difference': float(np.abs(im.affine-image.affine).max())}
        case_report = {'contract_sha256': sha(contract_path), 'producer_report_sha256': sha(producer_path), 'official_tensor_command': records[0],
            'inputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()}, 'reference_geometry': geometry,
            'shape': list(signal.shape), 'mask_voxels': int(mask.sum()), 'gradient_max_abs_log_norm2': float(np.abs(np.log(np.sum(gradient[np.linalg.norm(gradient[:, :3], axis=1) > 0, :3] ** 2, axis=1))).max()),
            'runs': [], 'accuracy': {}}
        case_out = args.output / case; case_out.mkdir()
        positive = mask & (signal.min(-1) > 0)
        with MemorySampler() as sampler:
            tensors = [torch.as_tensor(value, device=device) for value in (signal, gradient, mask)]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            if case in args.paired_cases:
                for module in modules.values():
                    values = module.fit_mrtrix_dhollander_tensor(*tensors); torch.cuda.synchronize(); del values
                order = ['baseline', 'candidate', 'candidate', 'baseline'] * args.abba_rounds
                for index, name in enumerate(order):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    values = modules[name].fit_mrtrix_dhollander_tensor(*tensors)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - started; del values
                    case_report['runs'].append({'sequence': index, 'version': name, 'solver_wall_s': elapsed})
            for name, module in modules.items():
                function, function_sha = instrument(module, 'baseline')
                values = function(*tensors); torch.cuda.synchronize()
                arrays = {key: value.cpu().numpy() for key, value in zip(('fa', 'direction', 'tensor'), values)}
                np.savez(case_out / (name + '.npz'), **arrays)
                errors = {group: {key: summary(arrays[key], references[key], selected) for key in arrays} for group, selected in [('all', mask), ('positive_measurements_reporting', positive), ('nonpositive_measurements_reporting', mask & ~positive)]}
                finite = mask & np.isfinite(arrays['direction']).all(-1) & np.isfinite(references['direction']).all(-1)
                left, right = arrays['direction'][finite].astype(np.float64), references['direction'][finite].astype(np.float64)
                norm = np.linalg.norm(left, axis=1)*np.linalg.norm(right, axis=1); valid = norm > 0
                angles = np.degrees(np.arccos(np.clip(np.abs((left[valid]*right[valid]).sum(-1))/norm[valid],0,1)))
                case_report['accuracy'][name] = {'errors': errors, 'direction_antipodal_degrees': {'max': float(angles.max(initial=0)), 'p99': float(np.quantile(angles, .99))}, 'capture_function_sha256': function_sha, 'output_sha256': sha(case_out / (name + '.npz'))}
                del values, arrays
            case_report['allocated_peak_bytes'] = torch.cuda.max_memory_allocated()
            case_report['reserved_peak_bytes'] = torch.cuda.max_memory_reserved()
            del tensors; torch.cuda.synchronize()
        case_report['nvml_process'] = sampler.result()
        peak = case_report['nvml_process']['peak_bytes']
        case_report['sampled_memory_budget_passed'] = peak is not None and 0 < peak < 20_000_000_000 and case_report['allocated_peak_bytes'] < 20_000_000_000 and case_report['reserved_peak_bytes'] < 20_000_000_000
        if any(sha(path) != case_report['inputs'][name]['sha256'] for name, path in paths.items()) or sha(producer_path) != case_report['producer_report_sha256'] or sha(contract_path) != case_report['contract_sha256']:
            raise ValueError('same-case sources changed during solver')
        report['cases'][case] = case_report
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'case': case, 'FA': {name: result['errors']['all']['fa'] for name, result in case_report['accuracy'].items()}, 'memory_passed': case_report['sampled_memory_budget_passed']}), flush=True)
        torch.cuda.empty_cache()
    if any(sha(path) != report['sources'][name]['sha256'] for name,path in [('baseline',args.baseline),('candidate',args.candidate)]) or sha(args.case_map) != report['case_map_sha256']: raise ValueError('frozen source changed')


if __name__ == '__main__': main()
