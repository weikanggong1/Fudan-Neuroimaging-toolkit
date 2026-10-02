"""Same-real-input first-level TOPUP cost/gradient diagnostic, no optimization."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imain', type=Path, required=True)
    parser.add_argument('--datain', type=Path, required=True)
    parser.add_argument('--official-gradient', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-limit-bytes', type=int, default=20_000_000_000)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('report must be new')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_bytes / torch.cuda.get_device_properties(device).total_memory, device)
    from fnit.topup.core import (TOPUPConfig, _TOPUPLevel, _load_inputs, _regrid_images,
                                _average_pool, _gaussian_blur, _cubic_spline_coefficients)
    reference, values, acquisition, axis, voxels = _load_inputs(args.imain, args.datain)
    if np.linalg.det(reference.affine[:3, :3]) > 0:
        values = values[::-1].copy()
    means = values.mean((0, 1, 2), dtype=np.float64)
    images = torch.as_tensor(np.moveaxis(values * (100/means).astype(np.float32)[None,None,None,:], -1,0).copy(),
                             device=args.device, dtype=torch.float32)
    config = TOPUPConfig()
    regridded, source_voxels = _regrid_images(images, voxels, max(config.subsampling), axis)
    factor = config.subsampling[0]
    shape = tuple(int(size//factor) for size in values.shape[:3])
    target_voxels = tuple(float(np.float32(value*factor)) for value in voxels)
    source_voxels = tuple(float(np.float32(value*factor)) for value in source_voxels)
    spacing = tuple(max(1,int(np.floor(config.warp_resolution_mm[0]/value+.5))) for value in target_voxels)
    image_coefficients = _cubic_spline_coefficients(_gaussian_blur(_average_pool(regridded,factor),config.fwhm_mm[0],source_voxels))
    problem = _TOPUPLevel(image_coefficients,shape,spacing,target_voxels,acquisition,axis,factor,
                          config.regularization[0],torch.zeros((2,6),device=args.device,dtype=torch.float64),
                          sampling_voxel_sizes=source_voxels)
    coefficients = torch.zeros(problem.control_shape,device=args.device,dtype=torch.float64)
    parameters = problem.parameters(coefficients,True)
    cost = float(problem.cost(parameters))
    state = problem.state(parameters)
    gradient = problem.gradient(parameters).detach().cpu().numpy()
    # Official NEWMAT coefficient vectors enumerate x inside y inside z.
    field_gradient = gradient[:problem.coefficient_count].reshape(problem.control_shape)
    official_order = np.concatenate((field_gradient.transpose(2,1,0).ravel(),gradient[problem.coefficient_count:]))
    payload = {'schema_version':1,'scope':'first b02b0 level, zero coefficients and movements, no optimization',
               'input_sha256':hashlib.sha256(args.imain.read_bytes()).hexdigest(),
               'datain_sha256':hashlib.sha256(args.datain.read_bytes()).hexdigest(),
               'target_shape':list(shape),'source_shape':list(image_coefficients.shape[1:]),
               'target_voxel_sizes':list(target_voxels),'source_voxel_sizes':list(source_voxels),
               'knot_spacing':list(spacing),'coefficient_shape':list(problem.control_shape),
               'cost':cost,'ssd':float(state['ssd']),'mask_voxels':state['voxels'],
               'gradient_norm':float(np.linalg.norm(official_order)),
               'gradient_max_abs':float(np.max(np.abs(official_order))),
               'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'memory_limit_bytes':args.memory_limit_bytes,
               'movement_gradient':official_order[-5:].tolist(),'source_core_sha256':hashlib.sha256(Path(__import__('fnit.topup.core',fromlist=['']).__file__).read_bytes()).hexdigest()}
    if args.official_gradient is not None:
        expected = np.loadtxt(args.official_gradient).reshape(-1)
        if expected.shape != official_order.shape or not np.isfinite(expected).all():
            raise ValueError('official gradient must be finite with matching parameter count')
        difference = official_order-expected
        payload['gradient_vs_official'] = {'parameter_count':int(expected.size),'pearson_r':float(np.corrcoef(official_order,expected)[0,1]),
                                            'mae':float(np.mean(np.abs(difference))),'rmse':float(np.sqrt(np.mean(difference**2))),
                                            'max_absdiff':float(np.max(np.abs(difference))),
                                            'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(expected)),
                                            'official_movement_gradient':expected[-5:].tolist(),
                                            'movement_absdiff':np.abs(difference[-5:]).tolist()}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(payload),flush=True)

if __name__ == '__main__':
    main()
