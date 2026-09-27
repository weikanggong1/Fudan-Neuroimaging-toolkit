"""将 FSL FIRST 的皮层下组织融合到 FreeSurfer 五组织图。

对应 MRtrix3 eeab681 的 ``lib/mrtrix3/_5ttgen/freesurfer.py`` 中
两个 FIRST 分支。目录模式需要 14 张由 FIRST 网格生成的 PVE。
"""

import torch


# ``-first -sgm_amyg_hipp`` 对应的 14 个结构标签。
FIRST_LABELS = (26, 58, 11, 50, 13, 52, 12, 51, 10, 49, 18, 54, 17, 53)


def freesurfer_first_five_tissue(
    five_tissue: torch.Tensor,
    *,
    first_pve: torch.Tensor | None = None,
    first_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 ACT 顺序的 ``float32 [X,Y,Z,5]`` FIRST 融合图。

    ``five_tissue`` 是目标 T1 网格上的基础 FreeSurfer 5TT。必须且只能
    提供一种同网格 FIRST 输入：``first_pve`` 是按 ``FIRST_LABELS``
    顺序堆叠的 ``[X,Y,Z,14]`` 网格 PVE；``first_labels`` 是备用整数
    ``*_firstseg`` 分割图 ``[X,Y,Z]``。
    """
    if five_tissue.ndim != 4 or five_tissue.shape[-1] != 5 or five_tissue.dtype != torch.float32:
        raise ValueError("five_tissue must be float32 [X,Y,Z,5]")
    if (first_pve is None) == (first_labels is None):
        raise ValueError("supply exactly one of first_pve or first_labels")
    shape = five_tissue.shape[:3]
    first = first_pve if first_pve is not None else first_labels
    if first.device != five_tissue.device or first.shape[:3] != shape:
        raise ValueError("FIRST data and five_tissue must have the same grid and device")

    if first_pve is not None:
        if first_pve.shape != (*shape, len(FIRST_LABELS)):
            raise ValueError("first_pve must have 14 structure channels")
        sgm = first_pve.to(torch.float32).sum(-1).clamp_max_(1)
        sgm *= five_tissue.sum(-1) > 0
        multiplier = 1 - sgm
        output = torch.empty_like(five_tissue)
        output[..., 0] = five_tissue[..., 0] * multiplier
        output[..., 1] = sgm
        output[..., 2] = (five_tissue[..., 2] + five_tissue[..., 1]) * multiplier
        output[..., 3] = five_tissue[..., 3] * multiplier
        output[..., 4] = five_tissue[..., 4] * multiplier
    else:
        if first_labels.ndim != 3:
            raise ValueError("first_labels must be a 3D integer label image")
        mask = torch.zeros(shape, dtype=torch.bool, device=five_tissue.device)
        for label in FIRST_LABELS:
            mask |= first_labels == label
        sgm = mask.to(torch.float32)
        output = torch.empty_like(five_tissue)
        output[..., 0] = (five_tissue[..., 0] - sgm).clamp_min_(0)
        output[..., 1] = sgm
        output[..., 2] = (five_tissue[..., 2] + five_tissue[..., 1] - sgm).clamp_min_(0)
        output[..., 3] = (five_tissue[..., 3] - sgm).clamp_min_(0)
        output[..., 4] = (five_tissue[..., 4] - sgm).clamp_min_(0)
    return output
