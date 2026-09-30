"""Plot significant BWAS voxel connections on a gray-matter surface."""

from __future__ import annotations

import csv
import gzip
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


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


def _gray_matter_surface(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed mesh from the supplied analysis mask."""
    import pyvista as pv

    smooth = gaussian_filter(np.pad(mask.astype(np.float32), 1), sigma=1.0)
    coarse = smooth[::2, ::2, ::2]
    grid = pv.ImageData(dimensions=coarse.shape, spacing=(2, 2, 2),
                        origin=(-1, -1, -1))
    grid.point_data["mask"] = coarse.ravel(order="F")
    surface = grid.contour([0.25], scalars="mask").triangulate()
    return nib.affines.apply_affine(affine, surface.points), surface.faces.reshape(-1, 4)[:, 1:]


def _brainnet_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an externally supplied BrainNet Viewer MNI surface (.nv)."""
    if path.suffix.lower() != ".nv":
        raise ValueError("brain_surface_file must be a BrainNet Viewer .nv mesh")
    numbers = np.fromstring("\n".join(line.partition("#")[0]
                                      for line in path.read_text().splitlines()), sep=" ")
    if len(numbers) < 5 or numbers[0] != int(numbers[0]):
        raise ValueError("invalid BrainNet Viewer surface vertex count")
    vertex_count = int(numbers[0])
    face_count_position = 1 + 3 * vertex_count
    if (vertex_count < 4 or len(numbers) <= face_count_position or
            numbers[face_count_position] != int(numbers[face_count_position])):
        raise ValueError("invalid BrainNet Viewer surface face count")
    face_count = int(numbers[face_count_position])
    if face_count < 1 or len(numbers) != face_count_position + 1 + 3 * face_count:
        raise ValueError("BrainNet Viewer surface has incomplete coordinates or faces")
    vertices = numbers[1:face_count_position].reshape(-1, 3)
    face_values = numbers[face_count_position + 1:]
    if not np.isfinite(vertices).all() or not np.isfinite(face_values).all():
        raise ValueError("BrainNet Viewer surface contains nonfinite values")
    faces = face_values.astype(np.int64).reshape(-1, 3) - 1
    if np.any(face_values != faces.ravel() + 1) or np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError("BrainNet Viewer surface has invalid 1-based face indices")
    return vertices, faces


def _cluster_bundles(path: Path, clusters: set[int], affine: np.ndarray):
    """Stream every edge and represent each significant cluster and sign once."""
    if not clusters:
        return []
    fields = ("voxel1_i", "voxel1_j", "voxel1_k", "voxel2_i", "voxel2_j", "voxel2_k")
    groups = {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {*fields, "z", "cluster"}.issubset(reader.fieldnames or []):
            raise ValueError("BWAS edge TSV has missing coordinate, z, or cluster columns")
        for row in reader:
            cluster = int(row["cluster"])
            if cluster not in clusters:
                continue
            z = float(row["z"])
            if not np.isfinite(z):
                raise ValueError("BWAS edge z values must be finite")
            key = (cluster, z >= 0)
            group = groups.get(key)
            if group is None:
                group = [0, np.zeros((2, 3), dtype=np.float64), 0.0]
                groups[key] = group
            group[0] += 1
            group[1] += np.asarray([int(row[field]) for field in fields]).reshape(2, 3)
            group[2] += z
    if {cluster for cluster, _ in groups} != clusters:
        raise ValueError("a significant BWAS cluster has no matching edges")
    return [(cluster, group[0], group[2] / group[0],
             nib.affines.apply_affine(affine, group[1] / group[0]))
            for (cluster, _), group in sorted(groups.items())]


def _connection_paths(world: np.ndarray, cluster_ids: np.ndarray,
                      z_values: np.ndarray, bundle_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Optionally bend display lines toward same-cluster, same-sign centroids."""
    steps = np.linspace(0.0, 1.0, 9 if bundle_strength else 2)
    paths = (world[:, 0, None, :] * (1.0 - steps)[None, :, None] +
             world[:, 1, None, :] * steps[None, :, None])
    if bundle_strength:
        weight = bundle_strength * np.sin(np.pi * steps) ** 2
        for cluster_id, positive in set(zip(cluster_ids, z_values >= 0)):
            group = np.flatnonzero((cluster_ids == cluster_id) & ((z_values >= 0) == positive))
            if len(group) < 2:
                continue
            centers = world[group].mean(axis=0)
            spine = (centers[0, None, :] * (1.0 - steps)[:, None] +
                     centers[1, None, :] * steps[:, None])
            paths[group] += weight[None, :, None] * (spine[None, :, :] - paths[group])
        paths[:, 0] = world[:, 0]
        paths[:, -1] = world[:, 1]
    width = len(steps)
    lines = np.column_stack((np.full(len(world), width),
                             np.arange(len(world) * width).reshape(-1, width))).ravel()
    return paths.reshape(-1, 3), lines


def plot_bwas_connectivity(
    bwas_output_root: str | Path,
    gray_matter_mask_file: str | Path,
    output_png: str | Path,
    *,
    top_k: int = 500,
    cluster_p_max: float = 0.05,
    cluster_id: int | None = None,
    min_abs_z: float | None = None,
    bundle_strength: float = 0.0,
    all_clusters: bool = False,
    brain_surface_file: str | Path | None = None,
    view: str = "montage",
) -> Path:
    """Create a PNG of significant BWAS edges on the 2 mm gray-matter mask.

    The single BWAS group result is found below ``bwas_output_root/group/func``.
    Only the strongest ``top_k`` significant edges are retained in memory.
    ``cluster_id`` selects one significant 6D cluster; ``bundle_strength``
    bends display lines within a cluster and sign without moving endpoints.
    ``all_clusters`` draws one weighted representative tube per cluster and
    sign using every significant edge, regardless of ``top_k``.
    Blue means negative z, while red means positive z. In edge mode endpoint
    size follows MA counts; in cluster mode tube width follows cluster size.
    Curves are not anatomical fiber paths.
    """
    if not isinstance(all_clusters, bool):
        raise ValueError("all_clusters must be a boolean")
    if not all_clusters and (isinstance(top_k, bool) or
                             not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if not 0 < cluster_p_max <= 1:
        raise ValueError("cluster_p_max must be in (0, 1]")
    if cluster_id is not None and (isinstance(cluster_id, bool) or
                                   not isinstance(cluster_id, (int, np.integer)) or
                                   cluster_id < 1):
        raise ValueError("cluster_id must be a positive integer or None")
    if min_abs_z is not None and (not np.isfinite(min_abs_z) or min_abs_z < 0):
        raise ValueError("min_abs_z must be finite and nonnegative")
    if not np.isfinite(bundle_strength) or not 0 <= bundle_strength <= 1:
        raise ValueError("bundle_strength must be in [0, 1]")
    if all_clusters and (cluster_id is not None or min_abs_z is not None or bundle_strength):
        raise ValueError("all_clusters cannot be combined with edge filters or bundle_strength")
    views = {
        "superior": ((0, 0, 1), (0, 1, 0)),
        "left": ((-1, 0, 0), (0, 0, 1)),
        "right": ((1, 0, 0), (0, 0, 1)),
        "anterior": ((0, 1, 0), (0, 0, 1)),
        "posterior": ((0, -1, 0), (0, 0, 1)),
        "inferior": ((0, 0, -1), (0, 1, 0)),
        "oblique": ((1, -1, 0.7), (0, 0, 1)),
    }
    if view not in {"montage", "six", "signed_six", *views}:
        raise ValueError(f"view must be montage, six, signed_six, or one of {', '.join(views)}")
    if view == "signed_six" and not all_clusters:
        raise ValueError("signed_six requires all_clusters=True")
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
    if cluster_id is not None:
        if cluster_id not in significant:
            raise ValueError("cluster_id is not significant at cluster_p_max")
        significant = {cluster_id}
    bundles = _cluster_bundles(edges_file, significant, mask_img.affine) if all_clusters else []
    edges = [] if all_clusters else _top_edges(edges_file, significant, top_k, min_abs_z)
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
    if brain_surface_file is None:
        display_mask = np.asarray(canonical.dataobj) != 0
        surface_vertices, surface_faces = _gray_matter_surface(display_mask, canonical.affine)
    else:
        surface_vertices, surface_faces = _brainnet_surface(Path(brain_surface_file))
    if view == "montage":
        panel_views = ("superior", "left", "right", "oblique")
    elif view == "six":
        panel_views = ("left", "superior", "right", "posterior", "inferior", "anterior")
    elif view == "signed_six":
        panel_views = ("left", "superior", "right") * 2
    else:
        panel_views = (view,)
    import pyvista as pv

    surface = pv.PolyData(surface_vertices, np.column_stack((
        np.full(len(surface_faces), 3), surface_faces)).ravel())
    center = np.asarray(surface.center)
    extent = np.asarray(surface.bounds).reshape(3, 2)
    spans = extent[:, 1] - extent[:, 0]
    if bundles:
        max_count = max(bundle[1] for bundle in bundles)
        tubes = {False: [], True: []}
        endpoint_positions = {False: [], True: []}
        endpoint_radii = {False: [], True: []}
        for _, count, z, ends in bundles:
            positive = z >= 0
            radius = 0.2 + 1.0 * np.log1p(count) / np.log1p(max_count)
            tube = pv.Line(ends[0], ends[1]).tube(radius=radius, n_sides=8, capping=True)
            tube.cell_data["signed_z"] = np.full(tube.n_cells, z)
            tubes[positive].append(tube)
            endpoint_positions[positive].extend(ends)
            endpoint_radii[positive].extend((1.2 + radius, 1.2 + radius))
        connections_by_sign = {sign: pv.merge(meshes, merge_points=False)
                               for sign, meshes in tubes.items() if meshes}
        connections = pv.merge(list(connections_by_sign.values()), merge_points=False)
        values = np.asarray([bundle[2] for bundle in bundles])
        limit = max(1.0, float(np.max(np.abs(values))))
        markers_by_sign = {}
        for sign, points in endpoint_positions.items():
            if not points:
                continue
            point_cloud = pv.PolyData(np.asarray(points))
            point_cloud["radius"] = endpoint_radii[sign]
            markers_by_sign[sign] = point_cloud.glyph(scale="radius", geom=pv.Sphere(
                theta_resolution=10, phi_resolution=10), orient=False)
        markers = pv.merge(list(markers_by_sign.values()), merge_points=False)
    elif edges:
        world = nib.affines.apply_affine(mask_img.affine, coordinates)
        values = np.asarray([entry[1] for entry in edges])
        limit = max(1.0, float(np.max(np.abs(values))))
        cluster_ids = np.asarray([entry[2] for entry in edges])
        line_points, lines = _connection_paths(world, cluster_ids, values, bundle_strength)
        connections = pv.PolyData(line_points, lines=lines)
        connections.cell_data["signed_z"] = values
        endpoints = np.unique(coordinates.reshape(-1, 3), axis=0)
        points = nib.affines.apply_affine(mask_img.affine, endpoints)
        counts = ma[tuple(endpoints.T)]
        point_cloud = pv.PolyData(points)
        point_cloud["radius"] = 0.6 + 1.3 * np.log1p(counts) / np.log1p(max(1.0, float(counts.max())))
        markers = point_cloud.glyph(scale="radius", geom=pv.Sphere(
            theta_resolution=8, phi_resolution=8), orient=False)
    has_connections = bool(bundles or edges)
    shape = (2, 3) if view in {"six", "signed_six"} else (2, 2) if view == "montage" else (1, 1)
    window_size = ((2700, 1800) if view in {"six", "signed_six"} else
                   (2160, 1800) if view == "montage" else (1440, 1440))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=shape, off_screen=True,
                         window_size=window_size, border=False)
    try:
        plotter.set_background("white", all_renderers=True)
        for panel_index, panel_view in enumerate(panel_views):
            plotter.subplot(panel_index // shape[1], panel_index % shape[1])
            plotter.add_mesh(surface, color="#aeb3b5", opacity=0.23 if brain_surface_file else 0.12,
                             show_edges=False, smooth_shading=True, lighting=True,
                             ambient=0.35, diffuse=0.65, specular=0.15)
            if has_connections:
                sign = panel_index < 3 if view == "signed_six" else None
                panel_connections = connections_by_sign.get(sign) if sign is not None else connections
                panel_markers = markers_by_sign.get(sign) if sign is not None else markers
                if panel_connections is not None:
                    plotter.add_mesh(panel_connections, scalars="signed_z", cmap=color_map,
                                     clim=(-limit, limit), line_width=1.5,
                                     render_lines_as_tubes=True, opacity=0.65 if bundles else 0.75,
                                     lighting=False,
                                     show_scalar_bar=panel_index == len(panel_views) - 1,
                                     scalar_bar_args={"title": "signed z", "vertical": False,
                                                      "position_x": 0.18, "position_y": 0.06,
                                                      "width": 0.64, "height": 0.09,
                                                      "title_font_size": 21,
                                                      "label_font_size": 17,
                                                      "n_labels": 2, "fmt": "%.1f"})
                if panel_markers is not None:
                    plotter.add_mesh(panel_markers, color="#d6ad20" if bundles else "#373d41",
                                     opacity=0.85 if bundles else 0.65,
                                     lighting=False, show_scalar_bar=False)
            direction, up = (np.asarray(value, dtype=float) for value in views[panel_view])
            direction /= np.linalg.norm(direction)
            plotter.camera_position = (center + direction * max(spans) * 3, center, up)
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = max(spans) * 0.55
            title = (("Positive " if panel_index < 3 else "Negative ") + panel_view.capitalize()
                     if view == "signed_six" else panel_view.capitalize())
            plotter.add_text(title, position="upper_left",
                             font_size=13, color="#30363a")
            if not has_connections:
                plotter.add_text("No edges pass the cluster and z thresholds",
                                 position="lower_left", font_size=10, color="#555555")
        plotter.screenshot(output_png)
    finally:
        plotter.close()
    return output_png
