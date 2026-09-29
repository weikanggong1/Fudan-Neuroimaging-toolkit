"""GPU sulcal registration from a native sphere to the fsLR sphere.

This is an independent smooth spherical optimizer, not the MSM binary. The
input sulcal metric and initial rotation follow ``prepare_msmsulc_inputs``.
"""

from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from .surface_registration import MSMSulcInputs


def _surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    points = np.asarray(image.darrays[0].data, dtype=np.float32)
    faces = np.asarray(image.darrays[1].data, dtype=np.int32)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not np.isfinite(points).all()):
        raise ValueError(f"invalid sphere: {path}")
    return points, faces


def _smoothing_graph(faces: np.ndarray, count: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    graph = sparse.csr_matrix((np.ones(len(row), np.float32), (row, col)),
                              shape=(count, count))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("sphere has isolated vertices")
    return sparse.diags(1 / degree) @ graph, edges


def _smooth_metric(values: np.ndarray, graph: sparse.csr_matrix, steps: int) -> np.ndarray:
    result = values.copy()
    for _ in range(steps):
        result = 0.5 * (result + graph @ result)
    result -= result.mean()
    scale = result.std()
    if scale <= 0:
        raise ValueError("sulcal metric has no variation")
    return (result / scale).astype(np.float32)


def _control_points(count: int, radius: float) -> np.ndarray:
    angle = np.arange(count) * np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * (np.arange(count) + 0.5) / count
    return (radius * np.column_stack((np.sqrt(1 - z * z) * np.cos(angle),
                                      np.sqrt(1 - z * z) * np.sin(angle), z))).astype(np.float32)


def run_msmsulc(
    inputs: dict[str, MSMSulcInputs],
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Path]:
    """Estimate L/R sulcus-aligned native spheres without FSL or MSM.

    ``inputs`` is the two-hemisphere result of ``prepare_msmsulc_inputs``.
    The returned ``L`` and ``R`` GIFTI paths retain each native mesh's vertex
    order. A JSON report records objective terms, wall time, GPU memory, and
    folded-triangle counts. Workbench may use these spheres for fsLR sampling.
    """
    if set(inputs) != {"L", "R"}:
        raise ValueError("inputs must contain L and R MSMSulc inputs")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    report = {}
    for hemi in ("L", "R"):
        entry = inputs[hemi]
        initial, faces = _surface(entry.rotated_sphere)
        target, target_faces = _surface(entry.reference_sphere)
        native = np.asarray(nib.load(str(entry.native_sulc)).darrays[0].data,
                            dtype=np.float32)
        reference = np.asarray(nib.load(str(entry.reference_sulc)).darrays[0].data,
                               dtype=np.float32)
        if (native.shape != (len(initial),) or reference.shape != (len(target),)
                or not np.isfinite(native).all() or not np.isfinite(reference).all()
                or np.min(faces) < 0 or np.max(faces) >= len(initial)
                or np.min(target_faces) < 0 or np.max(target_faces) >= len(target)):
            raise ValueError(f"{hemi} sulc and sphere topology differ")
        radius = float(np.linalg.norm(initial, axis=1).mean())
        if not 90 <= radius <= 110:
            raise ValueError(f"{hemi} rotated sphere must have approximately 100-mm radius")
        source_graph, edges = _smoothing_graph(faces, len(initial))
        target_graph, _ = _smoothing_graph(target_faces, len(target))
        base = torch.from_numpy(initial).to(selected)
        target_xyz = torch.from_numpy(target).to(selected)
        edge_i = torch.from_numpy(edges[:, 0].astype(np.int64)).to(selected)
        edge_j = torch.from_numpy(edges[:, 1].astype(np.int64)).to(selected)
        face = torch.from_numpy(faces.astype(np.int64)).to(selected)
        edge_length = torch.linalg.vector_norm(base[edge_i] - base[edge_j], dim=1)
        triangle = base[face]
        signed_base = (torch.cross(triangle[:, 1] - triangle[:, 0],
                                   triangle[:, 2] - triangle[:, 0], dim=1)
                       * triangle[:, 0]).sum(dim=1)
        if bool((signed_base.abs() < 1e-5).any()):
            raise ValueError(f"{hemi} sphere has degenerate triangles")
        controls = _control_points(25000, radius)
        distances, indices = cKDTree(controls).query(initial, k=8)
        weights = np.exp(-distances ** 2 / (2 * 2.8 ** 2)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        control_indices = torch.from_numpy(indices.astype(np.int64)).to(selected)
        control_weights = torch.from_numpy(weights).to(selected)
        delta_parameters = torch.nn.Parameter(torch.zeros((len(controls), 3), device=selected))
        optimizer = torch.optim.Adam([delta_parameters], lr=0.1)
        tree = cKDTree(target)
        stages = []
        start = time.perf_counter()
        if selected.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected)
        for smoothing, sigma, iterations, regularization in (
            (40, 8.0, 100, 0.03), (18, 5.0, 100, 0.02),
            (6, 3.0, 150, 0.01), (0, 2.0, 200, 0.005),
        ):
            source = torch.from_numpy(_smooth_metric(native, source_graph, smoothing)).to(selected)
            reference_metric = torch.from_numpy(
                _smooth_metric(reference, target_graph, smoothing)).to(selected)
            neighbors = None
            for step in range(iterations):
                delta = (delta_parameters[control_indices] * control_weights[:, :, None]).sum(dim=1)
                xyz = F.normalize(base + delta, dim=1) * radius
                if step % 5 == 0:
                    neighbors = torch.from_numpy(tree.query(
                        xyz.detach().cpu().numpy(), k=12, workers=4,
                    )[1].astype(np.int64)).to(selected)
                candidates = target_xyz[neighbors]
                distance_sq = ((xyz[:, None, :] - candidates) ** 2).sum(dim=-1)
                interpolation = torch.softmax(-distance_sq / (2 * sigma * sigma), dim=1)
                sampled = (interpolation * reference_metric[neighbors]).sum(dim=1)
                similarity = (sampled - source).square().mean()
                smoothness = (delta[edge_i] - delta[edge_j]).square().sum(dim=1).mean()
                displacement = delta.square().sum(dim=1).mean()
                strained = torch.linalg.vector_norm(xyz[edge_i] - xyz[edge_j], dim=1)
                strain = torch.log(strained / edge_length).square().mean()
                triangle = xyz[face]
                signed = (torch.cross(triangle[:, 1] - triangle[:, 0],
                                      triangle[:, 2] - triangle[:, 0], dim=1)
                          * triangle[:, 0]).sum(dim=1)
                area_ratio = signed / signed_base
                barrier = torch.relu(0.8 - area_ratio).square().mean()
                loss = (similarity + regularization * smoothness
                        + 0.0005 * displacement + 0.2 * strain + 20 * barrier)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            stages.append({"sulc_mse": float(similarity.detach()),
                           "edge_strain": float(strain.detach()),
                           "folded_triangles": int((area_ratio <= 0).sum())})
        with torch.no_grad():
            delta = (delta_parameters[control_indices] * control_weights[:, :, None]).sum(dim=1)
            xyz = F.normalize(base + delta, dim=1) * radius
            triangle = xyz[face]
            ratio = ((torch.cross(triangle[:, 1] - triangle[:, 0],
                                  triangle[:, 2] - triangle[:, 0], dim=1)
                      * triangle[:, 0]).sum(dim=1) / signed_base)
            folded = int((ratio <= 0).sum())
            if folded:
                raise RuntimeError(f"{hemi} registration folded {folded} triangles")
            vertices = xyz.cpu().numpy().astype(np.float32)
        destination = output / f"{hemi}.sphere.sulc_registered.native.surf.gii"
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(vertices, intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]), str(destination))
        result[hemi] = destination
        report[hemi] = {"vertices": len(vertices), "faces": len(faces),
                        "folded_triangles": folded,
                        "seconds": time.perf_counter() - start,
                        "peak_gpu_reserved_gb": (torch.cuda.max_memory_reserved(selected) / 1e9
                                                 if selected.type == "cuda" else 0),
                        "stages": stages}
        del delta_parameters, optimizer, base, target_xyz
        if selected.type == "cuda":
            torch.cuda.empty_cache()
    (output / "registration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return result
