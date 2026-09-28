"""FreeSurfer subject inputs and the 84-node Desikan atlas used by MRtrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

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
