"""Plot significant BWAS voxel connections on a gray-matter outline."""

from __future__ import annotations

import csv
import gzip
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


_EDGE_SUFFIX = "_desc-BWASedges_relmat.tsv.gz"
_CLUSTER_SUFFIX = "_desc-BWASclusters_stat.tsv"
_MA_SUFFIX = "_desc-BWASMA_statmap.nii.gz"


def _result_files(output_root: Path) -> tuple[Path, Path, Path]:
    folder = output_root / "group" / "func"
    edges = sorted(folder.glob(f"*{_EDGE_SUFFIX}"))
    if len(edges) != 1:
        raise ValueError(f"expected exactly one BWAS edge file in {folder}; found {len(edges)}")
    prefix = edges[0].name.removesuffix(_EDGE_SUFFIX)
    clusters = folder / f"{prefix}{_CLUSTER_SUFFIX}"
    ma_map = folder / f"{prefix}{_MA_SUFFIX}"
    if not clusters.is_file() or not ma_map.is_file():
        raise FileNotFoundError("matching BWAS clusters TSV and MA NIfTI are required")
    return edges[0], clusters, ma_map


def _significant_clusters(path: Path, p_max: float) -> set[int]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"cluster", "p_fwer"}.issubset(reader.fieldnames or []):
            raise ValueError("BWAS cluster TSV requires cluster and p_fwer columns")
        selected = set()
        for row in reader:
            cluster, p = int(row["cluster"]), float(row["p_fwer"])
            if cluster < 1 or not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid BWAS cluster identifier or FWER p value")
            if p < p_max:
                selected.add(cluster)
    return selected


def _top_edges(path: Path, clusters: set[int], top_k: int,
               min_abs_z: float | None) -> list[tuple[tuple[int, ...], float, int]]:
    """Keep only top-k edges in memory, even for million-row gzip files."""
    if not clusters:
        return []
    fields = ["voxel1_i", "voxel1_j", "voxel1_k", "voxel2_i", "voxel2_j",
              "voxel2_k", "z", "cluster"]
    heap: list[tuple[float, int, tuple[int, ...], float, int]] = []
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError("BWAS edge TSV has missing coordinate, z, or cluster columns")
        for index, row in enumerate(reader):
            cluster = int(row["cluster"])
            if cluster not in clusters:
                continue
            z = float(row["z"])
            if not np.isfinite(z):
                raise ValueError("BWAS edge z values must be finite")
            score = abs(z)
            if min_abs_z is not None and score < min_abs_z:
                continue
            if len(heap) == top_k and (score, -index) <= heap[0][:2]:
                continue
            coordinates = tuple(int(row[field]) for field in fields[:6])
            item = (score, -index, coordinates, z, cluster)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
    return [(item[2], item[3], item[4]) for item in sorted(heap, reverse=True)]


def _gray_matter_surface(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Build a translucent closed mesh from the supplied analysis mask on CPU."""
    smooth = gaussian_filter(np.pad(mask.astype(np.float32), 1), sigma=1.0)
    vertices, faces, _, _ = marching_cubes(smooth, level=0.25, step_size=2)
    return nib.affines.apply_affine(affine, vertices - 1)[faces]


def _draw_gray_matter(ax, triangles: np.ndarray) -> None:
    ax.add_collection3d(Poly3DCollection(
        triangles, linewidths=0, facecolors=(0.55, 0.58, 0.60, 0.07),
        edgecolors="none", zsort="average"))


def plot_bwas_connectivity(
    bwas_output_root: str | Path,
    gray_matter_mask_file: str | Path,
    output_png: str | Path,
    *,
    top_k: int = 500,
    cluster_p_max: float = 0.05,
    min_abs_z: float | None = None,
    view: str = "montage",
) -> Path:
    """Create a PNG of significant BWAS edges on the 2 mm gray-matter mask.

    The single BWAS group result is found below ``bwas_output_root/group/func``.
    Only the strongest ``top_k`` significant edges are retained in memory.
    Blue means negative z, while red means positive z; endpoint size
    follows the MA significant-edge count. The plot is illustrative, not a
    tractography or anatomical fiber map.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not 0 < cluster_p_max <= 1:
        raise ValueError("cluster_p_max must be in (0, 1]")
    if min_abs_z is not None and (not np.isfinite(min_abs_z) or min_abs_z < 0):
        raise ValueError("min_abs_z must be finite and nonnegative")
    views = {"superior": (90, -90), "left": (0, 180), "right": (0, 0),
             "anterior": (0, 90), "oblique": (25, -60)}
    if view != "montage" and view not in views:
        raise ValueError(f"view must be montage or one of {', '.join(views)}")
    output_png = Path(output_png).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError("output_png must have a .png extension")
    if output_png.exists():
        raise FileExistsError(f"visualization already exists: {output_png}")

    edges_file, clusters_file, ma_file = _result_files(Path(bwas_output_root))
    mask_img, ma_img = nib.load(str(gray_matter_mask_file)), nib.load(str(ma_file))
    if mask_img.ndim != 3 or ma_img.ndim != 3 or mask_img.shape != ma_img.shape:
        raise ValueError("gray-matter mask and BWAS MA map must be matching 3D NIfTI")
    if not np.allclose(mask_img.affine, ma_img.affine, atol=1e-3):
        raise ValueError("gray-matter mask and BWAS MA map affines differ")
    if not np.allclose(np.linalg.norm(mask_img.affine[:3, :3], axis=0), 2.0, atol=0.01):
        raise ValueError("gray-matter mask must be on a 2 mm grid")
    mask = np.asarray(mask_img.dataobj) != 0
    ma = np.asarray(ma_img.dataobj)
    if not mask.any() or not np.isfinite(ma).all() or np.any(ma < 0):
        raise ValueError("gray-matter mask must be nonempty and MA values nonnegative/finite")

    significant = _significant_clusters(clusters_file, cluster_p_max)
    edges = _top_edges(edges_file, significant, top_k, min_abs_z)
    if edges:
        coordinates = np.asarray([entry[0] for entry in edges], dtype=np.int64).reshape(-1, 2, 3)
        if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(mask.shape)):
            raise ValueError("BWAS edge endpoint lies outside the gray-matter mask grid")
        if not np.all(mask[tuple(coordinates.reshape(-1, 3).T)]):
            raise ValueError("BWAS edge endpoint lies outside the gray-matter mask")

    color_map = LinearSegmentedColormap.from_list("BWAS signed z", [
        (0.00, "#173b86"), (0.40, "#76b7df"), (0.499, "#edf6fb"),
        (0.501, "#fceae7"), (0.60, "#e17b72"), (1.00, "#b91f2d"),
    ])
    canonical = nib.as_closest_canonical(mask_img)
    if not np.allclose(canonical.affine[:3, :3],
                       np.diag(np.diag(canonical.affine[:3, :3])), atol=1e-3):
        raise ValueError("gray-matter mask must have axes aligned to MNI coordinates")
    display_mask = np.asarray(canonical.dataobj) != 0
    surface_triangles = _gray_matter_surface(display_mask, canonical.affine)
    panel_views = ("superior", "left", "right", "oblique") if view == "montage" else (view,)
    figure = Figure(figsize=(12, 10) if view == "montage" else (8, 8),
                    dpi=180, facecolor="white")
    FigureCanvasAgg(figure)
    world = nib.affines.apply_affine(mask_img.affine, coordinates) if edges else None
    if edges:
        values = np.asarray([entry[1] for entry in edges])
        limit = max(1.0, float(np.max(np.abs(values))))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        strength = np.abs(values) / limit
        colors = color_map(norm(values))
        colors[:, 3] = 0.08 + 0.28 * strength
        endpoints = np.unique(coordinates.reshape(-1, 3), axis=0)
        points = nib.affines.apply_affine(mask_img.affine, endpoints)
        counts = ma[tuple(endpoints.T)]
        denominator = np.log1p(max(1.0, float(counts.max())))
        point_sizes = 1.0 + 5.0 * np.log1p(counts) / denominator
        color_axis = figure.add_axes((0.91, 0.29, 0.018, 0.42) if view == "montage"
                                     else (0.89, 0.28, 0.025, 0.43))
        figure.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=color_axis,
                        label="signed z (blue: negative; red: positive)")
    else:
        figure.text(0.5, 0.07, "No edges pass the cluster and z thresholds",
                    ha="center", fontsize=10)
    corners = np.array(np.meshgrid(*[(0, s - 1) for s in mask.shape], indexing="ij"))
    world_corners = nib.affines.apply_affine(mask_img.affine, corners.reshape(3, -1).T)
    minimum, maximum = world_corners.min(axis=0), world_corners.max(axis=0)
    for panel_index, panel_view in enumerate(panel_views):
        if view == "montage":
            left = (0.01, 0.44)[panel_index % 2]
            bottom = (0.52, 0.10)[panel_index // 2]
            bounds = (left, bottom, 0.42, 0.39)
        else:
            bounds = (0.01, 0.08, 0.86, 0.84)
        ax = figure.add_axes(bounds, projection="3d")
        elevation, azimuth = views[panel_view]
        _draw_gray_matter(ax, surface_triangles)
        if edges:
            ax.add_collection3d(Line3DCollection(
                world, colors=colors, linewidths=0.25 + 0.55 * strength, zorder=10))
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_sizes,
                       c="#30373b", alpha=0.38, depthshade=False, zorder=11)
        ax.set(xlim=(minimum[0], maximum[0]), ylim=(minimum[1], maximum[1]),
               zlim=(minimum[2], maximum[2]))
        ax.set_box_aspect(maximum - minimum, zoom=1.5 if view == "montage" else 1.45)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_axis_off()
        if view == "montage":
            ax.set_title(panel_view.capitalize(), fontsize=11, pad=0)
    figure.suptitle(f"BWAS voxel-pair associations · top {len(edges):,} edges\n"
                     f"cluster FWER p < {cluster_p_max:g}", fontsize=13, y=0.96)
    figure.text(0.03, 0.025, "Straight lines: voxel-pair associations, not anatomical fibers.\n"
                "Endpoint size ∝ log(1 + MA significant-edge count).",
                fontsize=8, color="#4d5356")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180, facecolor="white")
    return output_png
