"""同输入 Double IWLS 诊断；独立算术臂不进入产品实现。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import socket
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(record):
    path = Path(record['path'])
    if sha(path) != record['sha256'] or ('size_bytes' in record and path.stat().st_size != record['size_bytes']):
        raise ValueError('actual consumer input bytes changed: ' + str(path))
    return path


def summary(left, right, selection):
    mask = selection if left.ndim == selection.ndim else np.broadcast_to(selection[..., None], left.shape)
    finite = mask & np.isfinite(left) & np.isfinite(right)
    delta = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
    return {'finite_elements': int(finite.sum()), 'neq': int(np.count_nonzero(delta)),
            'max': float(delta.max(initial=0)), 'p99': float(np.quantile(delta, .99)) if len(delta) else None,
            'p999': float(np.quantile(delta, .999)) if len(delta) else None,
            'rmse': float(np.sqrt(np.mean(delta * delta))) if len(delta) else None,
            'abs_gt_1e_minus5': int((delta > 1e-5).sum()),
            'nonfinite_mismatch': int((mask & ((np.isnan(left) != np.isnan(right)) |
                (np.isposinf(left) != np.isposinf(right)) | (np.isneginf(left) != np.isneginf(right)))).sum())}


def instrument(module, mode):
    source = inspect.getsource(module.fit_mrtrix_dhollander_tensor)
    source = source.replace('def fit_mrtrix_dhollander_tensor(', 'def diagnostic_fit(', 1)
    anchor = '    voxels = torch.nonzero(mask.reshape(-1)).flatten()'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '    flat_tensor = torch.zeros((flat_signal.shape[0], 6), device=device, dtype=torch.float32)\n' + anchor)
    source = source.replace('        flat_vec[block[~valid]] = torch.nan',
                            '        flat_vec[block[~valid]] = torch.nan\n        flat_tensor[block[~valid]] = torch.nan', 1)
    anchor = '        d11, d22, d33, d12, d13, d23 = d.unbind(dim=1)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '        flat_tensor[block] = d.to(torch.float32)\n' + anchor)
    anchor = '    return flat_fa.reshape(signal.shape[:3]), flat_vec.reshape(*signal.shape[:3], 3)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + ', flat_tensor.reshape(*signal.shape[:3], 6)')
    if 'rhs_left' in mode:
        anchor = '            right = weighted_design.transpose(1, 2) @ (weights * log_signal)[:, :, None]'
        assert source.count(anchor) == 1
        source = source.replace(anchor, '            right = (weighted_design.transpose(1, 2) * weights[:, None, :]) @ log_signal[:, :, None]')
    if 'basis_left' in mode:
        source = source.replace('-b * gx.square(), -b * gy.square(), -b * gz.square(),',
                                '-(b * gx * gx), -(b * gy * gy), -(b * gz * gz),', 1)
        source = source.replace('-2 * b * gx * gy, -2 * b * gx * gz, -2 * b * gy * gz,',
                                '-(b * gx * gy * 2), -(b * gx * gz * 2), -(b * gy * gz * 2),', 1)
    if 'grad_reload' in mode:
        anchor = '    gx, gy, gz, b = grad.unbind(dim=1)'
        assert source.count(anchor) == 1
        source = source.replace(anchor, '    grad = grad.clone()\n    squared_norm = grad[:, 0].square() + grad[:, 1].square() + grad[:, 2].square()\n    nonzero = squared_norm > 0\n    grad[nonzero, :3] /= squared_norm[nonzero, None].sqrt()\n    if bool(squared_norm[nonzero].log().abs().max() > 0.01):\n        grad[:, 3] *= squared_norm\n' + anchor)
    namespace = dict(module.__dict__)
    exec(compile(source, '<independent_diagnostic_' + mode + '>', 'exec'), namespace)
    return namespace['diagnostic_fit'], hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--modes', nargs='+', default=['baseline', 'rhs_left', 'grad_reload', 'grad_reload_rhs_left'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic namespace required')
    args.output.mkdir(parents=True)
    contract_sha = sha(args.contract)
    contract = json.loads(args.contract.read_text())
    if contract.get('state') != 'completed' or contract.get('execution_completed') is not True:
        raise ValueError('completed real same-case official model consumer required')
    keys = ['official_corrected_dwi', 'official_gradient_mrtrix', 'brain_mask', 'FA', 'principal_direction']
    paths = {name: checked(contract['files'][name]) for name in keys}
    tensor_path = paths['FA'].parent / 'tensor.nii.gz'
    image = nib.load(paths['official_corrected_dwi'])
    signal = image.get_fdata(dtype=np.float32)
    mask_image = nib.load(paths['brain_mask'])
    mask = np.asarray(mask_image.dataobj) > 0
    references = {}
    grids = {}
    for name, path in [('fa', paths['FA']), ('direction', paths['principal_direction']), ('tensor', tensor_path)]:
        reference = nib.load(path)
        if reference.shape[:3] != image.shape[:3] or not np.allclose(reference.affine, image.affine, rtol=0, atol=1e-6):
            raise ValueError('same-input reference geometry differs: ' + name)
        references[name] = reference.get_fdata(dtype=np.float32)
        grids[name] = {'path': str(path), 'sha256': sha(path), 'storage_dtype': str(reference.get_data_dtype()),
                       'shape': list(reference.shape), 'strides': list(np.asarray(reference.dataobj).strides),
                       'affine': reference.affine.tolist(),
                       'affine_max_abs_header_difference': float(np.abs(reference.affine - image.affine).max()),
                       'geometry_policy': 'same voxel axis/shape; existing CPU diagnostic NIfTI header rounding gate atol=1e-6 mm, no resampling'}
    if mask.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-6):
        raise ValueError('mask geometry differs')
    grad = np.loadtxt(paths['official_gradient_mrtrix'])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tensors = [torch.as_tensor(value, device=device) for value in (signal, grad, mask)]
    spec = importlib.util.spec_from_file_location('accuracy_frozen_response', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {'scope': 'same corrected DWI/gradient/brain mask tensor diagnostic; no official commands and no raw end-to-end',
              'case_id': contract['case_id'], 'host': socket.gethostname(), 'device': str(device),
              'torch': torch.__version__, 'threads': args.threads, 'batch_size': args.batch_size,
              'contract': {'path': str(args.contract), 'sha256': contract_sha},
              'source': {'path': str(args.source), 'sha256': sha(args.source)},
              'script_sha256': sha(__file__), 'inputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()},
              'reference_grids': grids, 'all_mask_voxels': int(mask.sum()),
              'precision': 'corrected Float32, all regression arithmetic Double, default TF32 retained; no regularisation/mask/filter/iteration change',
              'modes': {}}
    positive = mask & (signal.min(axis=-1) > 0)
    for mode in args.modes:
        function, source_sha = instrument(module, mode)
        if device.type == 'cuda':
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        values = function(*tensors, batch_size=args.batch_size)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        arrays = {name: tensor.detach().cpu().numpy() for name, tensor in zip(('fa', 'direction', 'tensor'), values)}
        np.savez(args.output / (mode + '.npz'), **arrays)
        errors = {selection_name: {name: summary(arrays[name], references[name], selected) for name in arrays}
                  for selection_name, selected in [('all', mask), ('positive_only_reporting', positive), ('nonpositive_measurements_reporting', mask & ~positive)]}
        finite = mask & np.isfinite(arrays['direction']).all(-1) & np.isfinite(references['direction']).all(-1)
        left, right = arrays['direction'][finite].astype(np.float64), references['direction'][finite].astype(np.float64)
        norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        valid = norms > 0
        angles = np.degrees(np.arccos(np.clip(np.abs((left[valid] * right[valid]).sum(-1)) / norms[valid], 0, 1)))
        delta = np.where(mask & np.isfinite(arrays['fa']) & np.isfinite(references['fa']), np.abs(arrays['fa'] - references['fa']), -np.inf)
        ranked = np.argsort(delta.reshape(-1))[-12:][::-1]
        report['modes'][mode] = {'instrumented_source_sha256': source_sha, 'fit_seconds': elapsed, 'errors': errors,
            'direction_antipodal_degrees': {'max': float(angles.max(initial=0)), 'p99': float(np.quantile(angles, .99)) if len(angles) else None},
            'worst_FA_voxels': [{'ijk': list(map(int, np.unravel_index(index, mask.shape))), 'abs_error': float(delta.reshape(-1)[index]),
                                'mask_row_rank': int(np.searchsorted(np.flatnonzero(mask.reshape(-1)), index)),
                                'candidate_tensor': arrays['tensor'].reshape(-1, 6)[index].tolist(),
                                'official_tensor': references['tensor'].reshape(-1, 6)[index].tolist()} for index in ranked],
            'allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            'output_sha256': sha(args.output / (mode + '.npz'))}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        del values, arrays
    if sha(args.contract) != contract_sha or any(sha(path) != report['inputs'][name]['sha256'] for name, path in paths.items()):
        raise ValueError('actual input bytes changed during diagnostic')
    print(json.dumps({name: {'fit_seconds': result['fit_seconds'], 'fa': result['errors']['all']['fa'],
                             'direction': result['direction_antipodal_degrees']} for name, result in report['modes'].items()}, indent=2))


if __name__ == '__main__':
    main()
