"""GPU point/morphology edits for the existing FNIT entorhinal/ACJ WM stage.

This module does not replace native mri_segment or its ordered aseg core.
"""
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from .mgh_compat import save_same_dtype_mgh


def amygdala_cortex_junction_gpu(aseg: torch.Tensor) -> torch.Tensor:
    """Same 7030/7031 junction labels as wm_edits_python, including last-neighbor ties.

    Input is a 3D integer aseg CUDA tensor in x/y/z voxel order. Output is same
    grid, int32 on that device. Neighborhood is 26 neighbors plus the center;
    edge amygdala seeds are excluded. No interpolation, RAS or mm conversion.
    """
    if aseg.ndim != 3 or aseg.device.type != 'cuda' or aseg.dtype.is_floating_point:
        raise ValueError('expected a 3D integer CUDA aseg tensor')
    cortex = (aseg == 3) | (aseg == 42)
    seeds = []
    boundaries = []
    for label in (18, 54):
        seed = (aseg == label).clone()
        seed[[0, -1], :, :] = False
        seed[:, [0, -1], :] = False
        seed[:, :, [0, -1]] = False
        seeds.append(seed)
        boundaries.append((F.max_pool3d(seed.float()[None, None],3,1,1)[0,0] > 0) & cortex)
    result = torch.zeros_like(aseg, dtype=torch.int32)
    result[boundaries[0]] = 7030
    result[boundaries[1]] = 7031
    overlap = boundaries[0] & boundaries[1]
    # Preserve the CPU C-order c/r/s scan: the last encountered amygdala wins.
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                offsets=(dx,dy,dz)
                target=tuple(slice(max(0,-d),min(size,size-d)) for d,size in zip(offsets,aseg.shape))
                source=tuple(slice(max(0,d),min(size,size+d)) for d,size in zip(offsets,aseg.shape))
                local=result[target]
                local[overlap[target] & seeds[0][source]]=7030
                local[overlap[target] & seeds[1][source]]=7031
    return result


@torch.no_grad()
def fix_ento_wm_gpu(input_file: str | Path, label_file: str | Path,
                    output_file: str | Path, *, level: int,
                    left_value: int, right_value: int,
                    device: str | torch.device, acj: bool = False) -> int:
    """GPU equivalent of FNIT fix_ento_wm; same file API plus explicit device.

    input_file and label_file: matching 3D MGH/MGZ voxel grids, affines in mm
    within 1e-4. output_file preserves input dtype/header/footer. level 1/2/3
    chooses entorhinal/subiculum/both; ACJ uses 7030/7031 instead of 3201/4201.
    left_value/right_value are assigned in input dtype. acj=False by default.
    Return edited voxel count. Missing files, invalid grids/device or CUDA
    failure propagate exceptions; no CPU fallback. Reading/writing uses nibabel.
    """
    device=torch.device(device)
    if device.type!='cuda': raise ValueError('explicit CUDA device required')
    image=nib.load(str(input_file)); labels=nib.load(str(label_file))
    if (image.shape!=labels.shape or len(image.shape)!=3 or
        not np.allclose(image.affine,labels.affine,rtol=0,atol=1e-4)):
        raise ValueError('Input and label volumes must share a 3D grid')
    # MGH multi-byte storage is big-endian; torch requires native byte order.
    source=np.asarray(image.dataobj)
    source=source.astype(source.dtype.newbyteorder('='),copy=False)
    voxels=torch.tensor(source,device=device)
    segmentation=torch.tensor(np.asarray(labels.dataobj).astype(np.int32),device=device)
    if acj: segmentation=amygdala_cortex_junction_gpu(segmentation)
    left=torch.zeros(image.shape,dtype=torch.bool,device=device)
    right=torch.zeros_like(left)
    if level in (1,3):
        left|=segmentation==3006;right|=segmentation==4006
    if level in (2,3):
        left|=segmentation==(7030 if acj else 3201)
        right|=segmentation==(7031 if acj else 4201)
    voxels[left]=left_value;voxels[right]=right_value
    save_same_dtype_mgh(input_file,output_file,voxels.cpu().numpy())
    return int((left.sum()+right.sum()).item())
