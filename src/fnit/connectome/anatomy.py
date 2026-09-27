"""FreeSurfer parcellation to MRtrix 5TT and GM/WM interface on a torch device.

The numeric LUT follows MRtrix3 3.0.3 ``5ttgen freesurfer -nocrop
-sgm_amyg_hipp`` without its older ``-first`` extension. This module uses a
precomputed FreeSurfer 8.2 name-to-ID match; it does not produce aparc+aseg.
"""

from importlib.resources import files

import torch


def freesurfer_five_tissue(segmentation: torch.Tensor) -> torch.Tensor:
    """Return float32 [X,Y,Z,5] in cortical GM, subcortical GM, WM, CSF, path order."""
    if segmentation.ndim != 3:
        raise ValueError("segmentation must be a 3D FreeSurfer label image")
    if not bool(torch.isfinite(segmentation).all()) or not bool(
        torch.equal(segmentation, segmentation.round())
    ) or bool((segmentation < 0).any()):
        raise ValueError("segmentation must contain finite non-negative integer labels")
    mapping = torch.zeros(16384, dtype=torch.int8)
    for line in files(__package__).joinpath("FreeSurfer2ACT_sgm_amyg_hipp_ids.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            label, tissue = (int(value) for value in line.split())
            mapping[label] = tissue
    mapping = mapping.to(segmentation.device)
    indices = segmentation.long()
    tissue = mapping[indices.clamp_max(len(mapping) - 1)]
    tissue = torch.where(indices < len(mapping), tissue, 0)
    return torch.stack([(tissue == channel).to(torch.float32) for channel in range(1, 6)], -1)


def gmwmi_from_five_tissue(five_tissue: torch.Tensor) -> torch.Tensor:
    """Port the MRtrix 5tt2gmwmi central-difference tissue-gradient rule."""
    if five_tissue.ndim != 4 or five_tissue.shape[-1] != 5:
        raise ValueError("five_tissue must have shape [X,Y,Z,5]")
    gm = five_tissue[..., 0] + five_tissue[..., 1]
    wm = five_tissue[..., 2]
    gradient_squared = torch.zeros_like(gm)
    for axis in range(3):
        negative_gm = gm.roll(1, axis)
        positive_gm = gm.roll(-1, axis)
        negative_wm = wm.roll(1, axis)
        positive_wm = wm.roll(-1, axis)
        negative_gm.select(axis, 0).copy_(gm.select(axis, 0))
        positive_gm.select(axis, -1).copy_(gm.select(axis, -1))
        negative_wm.select(axis, 0).copy_(wm.select(axis, 0))
        positive_wm.select(axis, -1).copy_(wm.select(axis, -1))
        difference = torch.minimum((positive_gm - negative_gm).abs(),
                                   (positive_wm - negative_wm).abs())
        difference *= 0.5
        difference.select(axis, 0).mul_(2)
        difference.select(axis, -1).mul_(2)
        gradient_squared += difference.square()
    return gradient_squared.sqrt()


def resample_labels_nearest(
    labels: torch.Tensor,
    source_affine: torch.Tensor,
    target_shape: tuple[int, int, int],
    target_affine: torch.Tensor,
    target_to_source_world: torch.Tensor,
) -> torch.Tensor:
    """Sample an integer atlas on a target grid using a target-to-source RAS transform."""
    if labels.ndim != 3 or any(size < 1 for size in target_shape):
        raise ValueError("labels and target_shape must be three-dimensional")
    device = labels.device
    source_affine = torch.as_tensor(source_affine, dtype=torch.float64, device=device)
    target_affine = torch.as_tensor(target_affine, dtype=torch.float64, device=device)
    target_to_source_world = torch.as_tensor(
        target_to_source_world, dtype=torch.float64, device=device
    )
    if any(matrix.shape != (4, 4) for matrix in
           (source_affine, target_affine, target_to_source_world)):
        raise ValueError("affines and transform must be 4x4")
    axes = [torch.arange(size, device=device, dtype=torch.float64)
            for size in target_shape]
    voxel = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    target_world = voxel @ target_affine[:3, :3].T + target_affine[:3, 3]
    source_world = target_world @ target_to_source_world[:3, :3].T + target_to_source_world[:3, 3]
    inverse = torch.linalg.inv(source_affine)
    source_voxel = source_world @ inverse[:3, :3].T + inverse[:3, 3]
    nearest = source_voxel.round().long()
    inside = torch.ones(len(nearest), dtype=torch.bool, device=device)
    for axis, size in enumerate(labels.shape):
        inside &= (nearest[:, axis] >= 0) & (nearest[:, axis] < size)
    safe = torch.stack([nearest[:, axis].clamp(0, size - 1)
                        for axis, size in enumerate(labels.shape)], -1)
    sampled = labels[safe[:, 0], safe[:, 1], safe[:, 2]]
    return torch.where(inside, sampled, 0).reshape(target_shape)


def combine_cortical_subcortical(
    cortical: torch.Tensor, subcortical: torch.Tensor, cortical_max_label: int
) -> torch.Tensor:
    """Match UKB-connectomics precedence and subcortical label shift."""
    if cortical.shape != subcortical.shape or cortical.ndim != 3:
        raise ValueError("cortical and subcortical atlas grids must match")
    if cortical_max_label < int(cortical.max()):
        raise ValueError("cortical_max_label must cover every cortical label")
    return torch.where(cortical > 0, cortical,
                       torch.where(subcortical > 0,
                                   subcortical + cortical_max_label, 0))
