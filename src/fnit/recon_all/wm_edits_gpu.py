"""GPU point/morphology edits for the existing FNIT entorhinal/ACJ WM stage.

This module does not replace native mri_segment or its ordered aseg core.
"""
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from .mgh_compat import save_same_dtype_mgh


def _amygdala_cortex_junction_torch(aseg: torch.Tensor) -> torch.Tensor:
    """复用ACJ内核生成7030/7031标签，保留最后邻居决定同分的规则。

    aseg为3D整数标签张量，按x/y/z体素顺序；私有诊断内核允许CPU或
    CUDA。输出同网格、同设备int32。邻域为26邻居与中心，排除边缘
    杏仁核种子；无插值、RAS变换或mm换算。无可调参数，不修改输入。
    shape或dtype错误抛ValueError；设备运行故障原样传播。
    """
    if aseg.ndim != 3 or aseg.dtype.is_floating_point or aseg.dtype == torch.bool:
        raise ValueError('expected a 3D integer aseg tensor')
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


def amygdala_cortex_junction_gpu(aseg: torch.Tensor) -> torch.Tensor:
    """返回CUDA上同网格int32的7030/7031 ACJ标签，不修改aseg。

    aseg须3D整数CUDA张量，坐标为x/y/z体素；无可调参数、插值或
    空间变换。CPU、shape或dtype错误抛ValueError。CPU公共接口保留
    在原模块；此接口不会静默回退。
    """
    if aseg.device.type != 'cuda':
        raise ValueError('expected a 3D integer CUDA aseg tensor')
    return _amygdala_cortex_junction_torch(aseg)


def _native_voxel_array(image):
    """Preserve values/dtype while converting MGH byte order for torch upload."""
    source=np.asarray(image.dataobj)
    return np.ascontiguousarray(source,dtype=source.dtype.newbyteorder('='))


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
    voxels=torch.tensor(_native_voxel_array(image),device=device)
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
