"""FreeSurfer subject inputs and native Desikan/Destrieux atlas labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


@dataclass(frozen=True)
class FreeSurferSubject:
    """An already completed recon-all directory; no FreeSurfer command is run."""

    subject_dir: Path

    def __post_init__(self) -> None:
        root = Path(self.subject_dir)
        for relative in ("mri/brain.mgz", "mri/aparc+aseg.mgz"):
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
        object.__setattr__(self, "subject_dir", root)

    @property
    def brain(self) -> Path:
        """Skull-stripped T1 registration target, ``mri/brain.mgz``."""
        return self.subject_dir / "mri/brain.mgz"

    @property
    def aparc_aseg(self) -> Path:
        """Desikan cortical and subcortical labels, ``mri/aparc+aseg.mgz``."""
        return self.subject_dir / "mri/aparc+aseg.mgz"

    @property
    def aparc_a2009s_aseg(self) -> Path:
        """Destrieux and subcortical labels, ``mri/aparc.a2009s+aseg.mgz``."""
        return self.subject_dir / "mri/aparc.a2009s+aseg.mgz"


@dataclass(frozen=True)
class ConnectomeNode:
    """One matrix row: contiguous index, FreeSurfer source ID, side, name."""

    index: int
    original_label: int
    hemisphere: str
    name: str


@lru_cache(maxsize=1)
def fs_aparc_nodes() -> tuple[ConnectomeNode, ...]:
    """Return the fixed 84 rows of MRtrix ``fs_default.txt`` order."""
    resource = files("fnit.connectome").joinpath("data/fs_aparc84.tsv")
    with resource.open("r", encoding="utf-8") as stream:
        rows = tuple(ConnectomeNode(
            index=int(row["index"]),
            original_label=int(row["original_label"]),
            hemisphere=row["hemisphere"],
            name=row["name"],
        ) for row in csv.DictReader(stream, delimiter="\t"))
    if tuple(node.index for node in rows) != tuple(range(1, 85)):
        raise ValueError("fs_aparc84.tsv must define contiguous nodes 1..84")
    return rows


def fs_aparc_atlas(segmentation: torch.Tensor) -> tuple[torch.Tensor, tuple[ConnectomeNode, ...]]:
    """Relabel FreeSurfer ``aparc+aseg`` [X,Y,Z] to int32 nodes 0..84.

    Each returned node defines one matrix row. Background and unused
    FreeSurfer IDs map to 0. Equivalent reference: ``labelconvert`` with
    FreeSurferColorLUT.txt and MRtrix ``fs_default.txt``.
    """
    if segmentation.ndim != 3 or not bool(torch.isfinite(segmentation).all()) or not bool(
        torch.equal(segmentation, segmentation.round())
    ) or bool((segmentation < 0).any()):
        raise ValueError("segmentation must be a finite nonnegative 3D integer image")
    nodes = fs_aparc_nodes()
    limit = max(node.original_label for node in nodes) + 1
    mapping = torch.zeros(limit, dtype=torch.int32, device=segmentation.device)
    for node in nodes:
        mapping[node.original_label] = node.index
    labels = segmentation.long()
    atlas = mapping[labels.clamp_max(limit - 1)]
    atlas = torch.where(labels < limit, atlas, 0)
    return atlas, nodes


def fs_aparc_a2009s_atlas(
    segmentation: torch.Tensor, subject_dir: str | Path,
) -> tuple[torch.Tensor, tuple[ConnectomeNode, ...]]:
    """把官方 a2009s+aseg 体积映射成同设备连续节点图。

    ``segmentation`` 是整数标签张量 [X,Y,Z]；``subject_dir`` 是含双半球
    ``label/*.aparc.a2009s.annot`` 的官方 recon-all 目录。返回 int32
    标签图 0..K 与定义矩阵行列的节点表。11100+i/12100+i 对应左右注释
    编号 i，皮层后接已有 fs-aparc 表的 16 个皮层下/小脑节点。
    """
    if segmentation.ndim != 3 or not bool(torch.isfinite(segmentation).all()) or not bool(
        torch.equal(segmentation, segmentation.round())
    ) or bool((segmentation < 0).any()):
        raise ValueError("segmentation must be a finite nonnegative 3D integer image")
    root = Path(subject_dir)
    annotations = [nib.freesurfer.read_annot(str(
        root / f"label/{hemi}.aparc.a2009s.annot"))
        for hemi in ("lh", "rh")]
    present = sorted(int(index) for index in np.unique(annotations[0][0]) if index > 0)
    if not present or any(index >= len(names) for _, _, names in annotations
                          for index in present):
        raise ValueError("a2009s annotation names do not cover cortical labels")
    nodes = []
    for hemisphere, offset, (_, _, names) in zip(
        ("L", "R"), (11100, 12100), annotations,
    ):
        for index in present:
            nodes.append(ConnectomeNode(
                len(nodes) + 1, offset + index, hemisphere,
                f"ctx-{'lh' if hemisphere == 'L' else 'rh'}-{names[index].decode('utf-8')}",
            ))
    for node in fs_aparc_nodes():
        if node.original_label < 1000:
            nodes.append(ConnectomeNode(
                len(nodes) + 1, node.original_label, node.hemisphere, node.name,
            ))
    mapping = torch.zeros(12200, dtype=torch.int32, device=segmentation.device)
    for node in nodes:
        mapping[node.original_label] = node.index
    source = segmentation.long()
    atlas = mapping[source.clamp_max(len(mapping) - 1)]
    cortical = ((source > 11100) & (source < 11200)) | ((source > 12100) & (source < 12200))
    if not bool(cortical.any()):
        raise ValueError("a2009s volume contains no expected 11100/12100 cortical labels")
    if bool((cortical & (atlas == 0)).any()):
        raise ValueError("a2009s volume contains labels absent from its native annotation")
    return torch.where(source < len(mapping), atlas, 0), tuple(nodes)
