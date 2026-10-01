"""MNI BOLD → fsLR32k 采样，以及皮层标签到原体积网格的映射。"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..connectome.atlas_surface import _nearest_surface_vertices
from .core import load_assets, parcellate, profiles_from_timeseries


def _surface_points(left_surface, right_surface):
    points = []
    for path in (left_surface, right_surface):
        image = nib.load(str(path))
        arrays = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        if len(arrays) != 1 or arrays[0].data.shape != (32492, 3):
            raise ValueError("each anatomical fsLR32k surface must contain [32492,3] points")
        points.append(np.asarray(arrays[0].data, dtype=np.float64))
    points = np.concatenate(points)
    if not np.isfinite(points).all():
        raise ValueError("surface coordinates must be finite")
    return points


def project_volume(volume, left_surface, right_surface, *, assets=None,
                   device="cuda:0", frame_chunk=8):
    """在两个解剖中层表面采样，返回 float32 [time,59412]。

    volume 为已配准的 4D NIfTI；左右表面坐标必须与其 scanner-RAS mm
    位于同一空间，顶点次序为 fsLR32k。使用三线性插值，不平滑时间。
    frame_chunk 控制一次传入 GPU 的帧数；表面越界时报错。
    """
    assets = load_assets() if assets is None else assets
    image = nib.load(str(volume))
    if image.ndim != 4 or image.shape[-1] < 8 or frame_chunk < 1:
        raise ValueError("volume must be 4D with at least 8 frames; frame_chunk must be positive")
    points = _surface_points(left_surface, right_surface)[assets["cortex_mask"]]
    inverse = np.linalg.inv(image.affine)
    voxel = points @ inverse[:3, :3].T + inverse[:3, 3]
    shape = np.asarray(image.shape[:3])
    if np.any(shape < 2) or np.any(voxel < 0) or np.any(voxel > shape - 1):
        raise ValueError("cortical surface points lie outside the input volume grid")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    grid = torch.as_tensor(2 * voxel / (shape - 1) - 1, dtype=torch.float32,
                           device=selected).reshape(1, len(points), 1, 1, 3)
    # Read compressed NIfTI once; repeated proxy slices would decompress it again.
    data = np.asarray(image.dataobj, dtype=np.float32)
    series = np.empty((image.shape[-1], len(points)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, image.shape[-1], frame_chunk):
            stop = min(start + frame_chunk, image.shape[-1])
            chunk = np.ascontiguousarray(data[..., start:stop].transpose(3, 2, 1, 0))
            tensor = torch.from_numpy(chunk)[None].to(selected)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("volume contains nonfinite values")
            sampled = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=True)[0, :, :, 0, 0]
            series[start:stop] = sampled.cpu().numpy()
    return series


def labels_to_volume(labels, reference, left_surface, right_surface, cortical_mask,
                     output, *, device="cuda:0", max_distance_mm=3.0):
    """将 [64984] 标签写到 reference 的 3D 网格，返回输出 Path。

    cortical_mask 必须与 reference 同网格；非零体素才参与映射。
    每个体素取最近皮层中层顶点的整数标签；距离大于 max_distance_mm
    的体素为 0。背景、medial wall、皮层下和小脑不应包含在 mask 内。
    """
    image = nib.load(str(reference))
    mask_image = nib.load(str(cortical_mask))
    mask = np.asarray(mask_image.dataobj)
    labels = np.asarray(labels)
    if labels.shape != (64984,) or not np.isin(labels, np.arange(18)).all():
        raise ValueError("labels must contain 64984 integers from 0 to 17")
    if (mask.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine,
                                                       atol=1e-5, rtol=0)):
        raise ValueError("cortical_mask and reference must share one voxel grid and affine")
    if not np.isfinite(mask).all() or max_distance_mm <= 0:
        raise ValueError("mask must be finite and max_distance_mm must be positive")
    indices = np.argwhere(mask != 0)
    points = _surface_points(left_surface, right_surface)
    # Include medial wall in the nearest search so its label 0 is preserved.
    selected = torch.device(device)
    vertices = torch.as_tensor(points, dtype=torch.float64, device=selected)
    query = torch.as_tensor(indices @ image.affine[:3, :3].T + image.affine[:3, 3],
                            dtype=torch.float64, device=selected)
    nearest = _nearest_surface_vertices(query, vertices, bin_width_mm=3.0,
                                         query_batch=2048)
    values = torch.as_tensor(labels, device=selected)[nearest]
    values[torch.linalg.vector_norm(query - vertices[nearest], dim=1) > max_distance_mm] = 0
    result = np.zeros(image.shape[:3], dtype=np.uint8)
    result[tuple(indices.T)] = values.cpu().numpy()
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_intent("label")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, image.affine, header), str(destination))
    return destination


def parcellate_volume(volume, left_surface, right_surface, cortical_mask, output_dir,
                      *, assets=None, censor=None, w=200.0, c=50.0, device="cuda:0",
                      frame_chunk=8, max_distance_mm=3.0):
    """单被试体积入口：GPU 采样、原 MS-HBM CPU 推断、GPU 标签映射。

    返回 (labels[64984], history)；写出标签数组、皮层 CIFTI、17 网络时序/
    相关矩阵及 labels_mni.nii.gz。无需调用外部影像软件。
    """
    from .output import save_results

    assets = load_assets() if assets is None else assets
    series = project_volume(volume, left_surface, right_surface, assets=assets,
                            device=device, frame_chunk=frame_chunk)
    profiles = profiles_from_timeseries(series, assets, censor)
    labels, history = parcellate(profiles, assets, w=w, c=c)
    save_results(output_dir, labels, series, assets["cortex_mask"], censor=censor)
    labels_to_volume(labels, volume, left_surface, right_surface, cortical_mask,
                     Path(output_dir) / "labels_mni.nii.gz", device=device,
                     max_distance_mm=max_distance_mm)
    return labels, history
