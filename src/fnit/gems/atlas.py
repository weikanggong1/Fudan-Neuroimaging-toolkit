"""FreeSurfer GEMS tetrahedral probabilistic-atlas I/O.

The reader follows ``kvlAtlasMeshCollection::Read``: AtlasMesh(.gz) stores a
reference mesh, one or more mesh positions, tetrahedral topology and per-node
anatomical label probabilities (alphas).  FNIT keeps these data in ordinary
NumPy arrays so no FreeSurfer/ITK runtime is required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
from pathlib import Path
from typing import Iterable

import numpy as np


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _header_value(line: str, key: str, cast=int):
    if key not in line:
        raise ValueError(f"Expected {key!r}, got {line.rstrip()!r}")
    return cast(line.split(key, 1)[1].strip())


def read_compression_lut(path: str | Path, n_labels: int | None = None):
    """Read a GEMS ``compressionLookupTable.txt``.

    Rows are ``FreeSurferID compressedIndex name R G B A``.  The returned
    arrays are ordered by the compressed index used by AtlasMesh alphas.
    """
    rows: dict[int, tuple[int, str]] = {}
    with Path(path).open("rt", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            fs_id, compressed = int(fields[0]), int(fields[1])
            rows[compressed] = (fs_id, fields[2])
    if not rows:
        raise ValueError(f"No labels found in {path}")
    size = max(rows) + 1 if n_labels is None else int(n_labels)
    if set(rows) != set(range(size)):
        missing = sorted(set(range(size)) - set(rows))
        raise ValueError(f"Compression LUT does not cover 0..{size-1}; missing {missing[:8]}")
    ids = np.asarray([rows[i][0] for i in range(size)], dtype=np.int32)
    names = tuple(rows[i][1] for i in range(size))
    return ids, names


@dataclass(frozen=True)
class GEMSAtlas:
    """A tetrahedral probabilistic atlas independent of FreeSurfer runtime.

    Coordinates are voxel-center coordinates in the atlas domain.  ``vertices``
    is the initial/current mesh position, while ``reference_vertices`` defines
    the deformation prior. ``alphas[v, k]`` is P(label=k) at mesh node v.
    """

    reference_vertices: np.ndarray
    vertices: np.ndarray
    tetrahedra: np.ndarray
    alphas: np.ndarray
    stiffness: float
    can_move: np.ndarray
    label_ids: np.ndarray
    label_names: tuple[str, ...]

    def __post_init__(self):
        rv = np.asarray(self.reference_vertices, dtype=np.float64)
        vv = np.asarray(self.vertices, dtype=np.float64)
        tt = np.asarray(self.tetrahedra, dtype=np.int64)
        aa = np.asarray(self.alphas, dtype=np.float32)
        cm = np.asarray(self.can_move, dtype=bool)
        ids = np.asarray(self.label_ids, dtype=np.int32)
        if rv.ndim != 2 or rv.shape[1] != 3 or vv.shape != rv.shape:
            raise ValueError("reference_vertices and vertices must both be [V,3]")
        if tt.ndim != 2 or tt.shape[1] != 4:
            raise ValueError("tetrahedra must be [T,4]")
        if tt.size and (tt.min() < 0 or tt.max() >= len(rv)):
            raise ValueError("tetrahedron vertex index out of range")
        if aa.ndim != 2 or aa.shape[0] != len(rv):
            raise ValueError("alphas must be [V,K]")
        if cm.shape != rv.shape:
            raise ValueError("can_move must be [V,3]")
        if ids.shape != (aa.shape[1],) or len(self.label_names) != aa.shape[1]:
            raise ValueError("label metadata must match alpha channels")
        sums = aa.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError("every mesh node must carry non-zero alpha mass")
        aa = aa / sums
        object.__setattr__(self, "reference_vertices", rv)
        object.__setattr__(self, "vertices", vv)
        object.__setattr__(self, "tetrahedra", tt)
        object.__setattr__(self, "alphas", aa)
        object.__setattr__(self, "can_move", cm)
        object.__setattr__(self, "label_ids", ids)
        object.__setattr__(self, "label_names", tuple(map(str, self.label_names)))

    @property
    def n_labels(self) -> int:
        return int(self.alphas.shape[1])

    def with_vertices(self, vertices: np.ndarray) -> "GEMSAtlas":
        return replace(self, vertices=np.asarray(vertices, dtype=np.float64))

    def transformed(self, matrix: np.ndarray, *, transform_reference: bool = True) -> "GEMSAtlas":
        """Apply a 4x4 voxel-coordinate affine to mesh coordinates."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        def apply(points):
            hom = np.concatenate((points, np.ones((len(points), 1))), axis=1)
            return (hom @ matrix.T)[:, :3]
        reference = apply(self.reference_vertices) if transform_reference else self.reference_vertices
        return replace(self, reference_vertices=reference, vertices=apply(self.vertices))

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            reference_vertices=self.reference_vertices,
            vertices=self.vertices,
            tetrahedra=self.tetrahedra,
            alphas=self.alphas,
            stiffness=np.asarray(self.stiffness, dtype=np.float64),
            can_move=self.can_move.astype(np.uint8),
            label_ids=self.label_ids,
            label_names=np.asarray(self.label_names, dtype="U"),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "GEMSAtlas":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["reference_vertices"], data["vertices"], data["tetrahedra"],
                data["alphas"], float(data["stiffness"]), data["can_move"].astype(bool),
                data["label_ids"], tuple(data["label_names"].tolist()),
            )

    @classmethod
    def from_freesurfer(
        cls,
        mesh: str | Path,
        compression_lut: str | Path | None = None,
        *,
        mesh_index: int = 0,
    ) -> "GEMSAtlas":
        """Read FreeSurfer ``AtlasMesh.gz`` without FreeSurfer or ITK.

        The text/gzip format is implemented directly from
        ``kvlAtlasMeshCollection::Read``. Non-tetrahedral cells are read and
        discarded because only tetrahedra contribute volumetric priors.
        """
        with _open_text(mesh) as stream:
            n_points = _header_value(stream.readline(), "Number of points:")
            n_cells = _header_value(stream.readline(), "Number of cells:")
            n_labels = _header_value(stream.readline(), "Number of labels:")
            n_meshes = _header_value(stream.readline(), "Number of meshes:")
            if mesh_index < 0 or mesh_index >= n_meshes:
                raise IndexError(f"mesh_index {mesh_index} outside 0..{n_meshes-1}")

            marker = stream.readline().strip().lower()
            if not marker.startswith("reference position"):
                raise ValueError("AtlasMesh missing Reference position section")
            reference = np.empty((n_points, 3), dtype=np.float64)
            point_map: dict[int, int] = {}
            for i in range(n_points):
                fields = stream.readline().split()
                old_id = int(fields[0])
                point_map[old_id] = i
                reference[i] = tuple(map(float, fields[1:4]))

            stiffness = _header_value(stream.readline(), "K:", float)
            positions: list[np.ndarray] = []
            for m in range(n_meshes):
                marker = stream.readline().strip().lower()
                if not marker.startswith("position"):
                    raise ValueError(f"AtlasMesh missing Position {m} section")
                position = np.empty((n_points, 3), dtype=np.float64)
                for _ in range(n_points):
                    fields = stream.readline().split()
                    position[point_map[int(fields[0])]] = tuple(map(float, fields[1:4]))
                positions.append(position)

            if not stream.readline().strip().lower().startswith("cells"):
                raise ValueError("AtlasMesh missing Cells section")
            tetrahedra: list[list[int]] = []
            for _ in range(n_cells):
                fields = stream.readline().split()
                if len(fields) < 3:
                    raise ValueError("Malformed AtlasMesh cell")
                cell_type = fields[1].upper()
                if cell_type not in {"VERTEX", "LINE", "TRIANGLE"}:
                    if len(fields) < 6:
                        raise ValueError("Malformed tetrahedral cell")
                    tetrahedra.append([point_map[int(v)] for v in fields[2:6]])

            if not stream.readline().strip().lower().startswith("point parameters"):
                raise ValueError("AtlasMesh missing Point parameters section")
            alphas = np.empty((n_points, n_labels), dtype=np.float32)
            can_move = np.empty((n_points, 3), dtype=bool)
            for _ in range(n_points):
                fields = stream.readline().split()
                idx = point_map[int(fields[0])]
                alphas[idx] = np.asarray(fields[1:1+n_labels], dtype=np.float32)
                flags = fields[1+n_labels:1+n_labels+4]
                if len(flags) != 4:
                    raise ValueError("Malformed AtlasMesh point parameters")
                can_move[idx] = [flag.lower() == "true" for flag in flags[1:4]]

        if compression_lut is None:
            label_ids = np.arange(n_labels, dtype=np.int32)
            label_names = tuple(f"label-{i}" for i in range(n_labels))
        else:
            label_ids, label_names = read_compression_lut(compression_lut, n_labels)
        return cls(reference, positions[mesh_index], np.asarray(tetrahedra, dtype=np.int64),
                   alphas, stiffness, can_move, label_ids, label_names)
