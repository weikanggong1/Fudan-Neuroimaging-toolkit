"""Shared affine initialization for FNIT MSMSulc."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch


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


def _affine_initialization(initial: np.ndarray, target: np.ndarray,
                           native: np.ndarray, reference: np.ndarray,
                           source_graph: sparse.csr_matrix,
                           target_graph: sparse.csr_matrix,
                           device: torch.device) -> tuple[np.ndarray, np.ndarray, list[float]]:
    base = torch.from_numpy(initial).to(device)
    target_xyz = torch.from_numpy(target).to(device)
    source = torch.from_numpy(_smooth_metric(native, source_graph, 8)).to(device)
    reference_metric = torch.from_numpy(_smooth_metric(reference, target_graph, 8)).to(device)
    tree = cKDTree(target)
    angles = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([angles], lr=0.001)
    neighbors = None
    for step in range(200):
        x, y, z = angles.unbind()
        skew = torch.stack((torch.stack((x * 0, -z, y)),
                            torch.stack((z, y * 0, -x)),
                            torch.stack((-y, x, z * 0))))
        rotated = base @ torch.matrix_exp(skew).T
        if step % 5 == 0:
            neighbors = torch.from_numpy(tree.query(
                rotated.detach().cpu().numpy(), k=12, workers=4,
            )[1].astype(np.int64)).to(device)
        distance_sq = ((rotated[:, None, :] - target_xyz[neighbors]) ** 2).sum(dim=-1)
        weights = torch.softmax(-distance_sq / (2 * 5.0 ** 2), dim=1)
        sampled = (weights * reference_metric[neighbors]).sum(dim=1)
        loss = (sampled - source).square().mean() + 0.1 * angles.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    matrix = torch.matrix_exp(skew).detach().cpu().numpy().astype(np.float32)
    return target @ matrix, matrix, (angles.detach().cpu().numpy() * 180 / np.pi).tolist()
