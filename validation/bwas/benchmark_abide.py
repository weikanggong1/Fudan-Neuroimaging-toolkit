"""Compare FNIT BWAS with functions loaded from the original BWAS_cpu.py.

The original file is read at runtime and is never bundled with FNIT.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import scipy
from scipy import sparse, stats
from scipy.spatial.distance import cdist
import torch

from fnit.bwas.core import _cluster_p, _clusters, _fisher_block, _smoothness


def load_original(path):
    source = path.read_text()
    names = {"BWAS_correlation", "BWAS_fisher_z", "BWAS_regression_online1",
             "BWAS_regression_online2", "BWAS_regression_online3",
             "BWAS_est_fwhm", "BWAS_GRF_6D_density",
             "BWAS_whole_brain_cluster_p", "BWAS_get_sparse_dismat"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "scipy": scipy, "norm": stats.norm,
                 "cdist": cdist, "sparse": sparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--fnit-result", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cdt", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    original, source_hash = load_original(args.upstream_source)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = perf_counter()
    with (args.bids_root / "participants.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    mask_img = nib.load(str(next(args.bids_root.glob("*analysis_mask.nii.gz"))))
    mask = np.asarray(mask_img.dataobj) != 0
    coords = np.column_stack(np.where(mask))
    design = np.float32([[float(row["case"]), float(row["age"]),
                          float(row["sex"]), 1] for row in rows])
    matrices, smoothness, smoothness_error = [], [], []
    for row in rows:
        file = next((args.bids_root / row["participant_id"]).glob("**/*desc-clean_bold.nii.gz"))
        data = np.asarray(nib.load(str(file)).dataobj, dtype=np.float32)
        original_width = float(np.mean(original["BWAS_est_fwhm"](data)))
        smoothness.append(original_width)
        smoothness_error.append(abs(original_width-_smoothness(data)))
        series = data[mask].T.copy()
        series = (series-series.mean(axis=0))/series.std(axis=0)
        matrices.append(series)
    n, v = len(rows), len(coords)
    y = np.empty((n, v*v), dtype=np.float32)
    for s, series in enumerate(matrices):
        r = original["BWAS_correlation"](series, series)
        r[r > 0.9999] = 0
        y[s] = original["BWAS_fisher_z"](r).reshape(-1)
    beta = original["BWAS_regression_online1"](
        design, y, 0, n, np.zeros((design.shape[1], v*v), dtype=np.float32))
    sigma = original["BWAS_regression_online2"](
        design, y, 0, n, beta, np.zeros((1, v*v), dtype=np.float32))
    t = original["BWAS_regression_online3"](design, beta, sigma).reshape(v, v)
    z = stats.norm.ppf(stats.t.cdf(t, n-design.shape[1]-1))
    edge_i, edge_j = np.where(np.triu(np.abs(z) > args.cdt, k=1))
    reference_edges = {(int(i), int(j)): float(z[i, j])
                       for i, j in zip(edge_i, edge_j)}
    adjacency = original["BWAS_get_sparse_dismat"](coords)
    if len(edge_i):
        combined = (adjacency[edge_i].T[edge_i] + adjacency[edge_j].T[edge_j]) == 2
        _, labels = sparse.csgraph.connected_components(combined, directed=False)
        reference_sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    else:
        reference_sizes = []
    reference_seconds = perf_counter()-start

    meta = json.loads(next(args.fnit_result.rglob("*_statmap.json")).read_text())
    edge_file = next(args.fnit_result.rglob("*BWASedges_relmat.tsv.gz"))
    lookup = {tuple(coord): i for i, coord in enumerate(coords)}
    with gzip.open(edge_file, "rt") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fnit_edges = {}
        for row in reader:
            first = lookup[tuple(int(row[f"voxel1_{axis}"]) for axis in "ijk")]
            second = lookup[tuple(int(row[f"voxel2_{axis}"]) for axis in "ijk")]
            fnit_edges[first, second] = float(row["z"])
    with next(args.fnit_result.rglob("*BWASclusters_stat.tsv")).open() as stream:
        fnit_sizes = sorted((int(row["edges"]) for row in
                             csv.DictReader(stream, delimiter="\t")), reverse=True)
    _, our_reference_table = _clusters(
        [(i, j, value) for (i, j), value in reference_edges.items()],
        coords, args.cdt, meta["FWHMInVoxels"])
    our_reference_sizes = sorted((int(row[1]) for row in our_reference_table), reverse=True)
    fnit_i = np.array([key[0] for key in fnit_edges], dtype=int)
    fnit_j = np.array([key[1] for key in fnit_edges], dtype=int)
    if len(fnit_i):
        fnit_adjacency = (adjacency[fnit_i].T[fnit_i] + adjacency[fnit_j].T[fnit_j]) == 2
        _, fnit_reference_labels = sparse.csgraph.connected_components(
            fnit_adjacency, directed=False)
        reference_on_fnit_sizes = sorted(np.bincount(fnit_reference_labels).tolist(), reverse=True)
    else:
        reference_on_fnit_sizes = []
    common = reference_edges.keys() & fnit_edges.keys()

    # The same FNIT Fisher values and design are fitted either all at once or
    # in subject chunks. This tests the user's required separable GLM identity.
    dev = torch.device(args.device)
    x = torch.as_tensor(design, dtype=torch.float32, device=dev)
    nuisance, _ = torch.linalg.qr(x[:, 1:], mode="reduced")
    phenotype_vector = x[:, 0] - nuisance @ (nuisance.T @ x[:, 0])
    x = torch.column_stack((phenotype_vector / torch.linalg.vector_norm(phenotype_vector),
                            nuisance))
    width = v
    full = _fisher_block(matrices, 0, n, 0, 0, width, width, dev,
                         dtype=torch.float32)
    full_xy = x.T @ full
    full_sigma = (full**2).sum(dim=0) - (full_xy**2).sum(dim=0)
    batch_xy = torch.zeros_like(full_xy)
    batch_yy = torch.zeros_like(full_sigma)
    for st in range(0, n, 7):
        en = min(st+7, n)
        batch_values = _fisher_block(matrices, st, en, 0, 0, width, width, dev,
                                     dtype=torch.float32)
        batch_xy += x[st:en].T @ batch_values
        batch_yy += (batch_values**2).sum(dim=0)
    batch_sigma = batch_yy - (batch_xy**2).sum(dim=0)
    df = n-x.shape[1]
    full_t = full_xy[0]/torch.sqrt(full_sigma/df)
    batch_t = batch_xy[0]/torch.sqrt(batch_sigma/df)
    finite = torch.isfinite(full_t) & torch.isfinite(batch_t)
    full_z = stats.norm.ppf(stats.t.cdf(full_t.cpu().numpy(), int(df)-1))
    batch_z = stats.norm.ppf(stats.t.cdf(batch_t.cpu().numpy(), int(df)-1))
    upper = np.triu_indices(v, k=1)
    original_z = z[upper]
    blocked_z = batch_z.reshape(v, v)[upper]
    valid_z = np.isfinite(original_z) & np.isfinite(blocked_z)
    original_z, blocked_z = original_z[valid_z], blocked_z[valid_z]

    representative_size = reference_sizes[0] if reference_sizes else 1
    original_p = original["BWAS_whole_brain_cluster_p"](
        v/np.sqrt(2), v/np.sqrt(2), args.cdt, representative_size,
        [meta["FWHMInVoxels"]]*3)[0]
    fnit_p = _cluster_p(v, args.cdt, representative_size,
                        meta["FWHMInVoxels"])[0]
    summary = {
        "dataset": "ABIDE II KKI, 2 mm resampled real preprocessed BOLD crop",
        "subjects": n, "case_count": int(design[:, 0].sum()),
        "control_count": int(n-design[:, 0].sum()),
        "voxels": v, "cdt": args.cdt,
        "original_source_sha256": source_hash,
        "reference_core_seconds": reference_seconds,
        "fnit_full_seconds": meta["ElapsedSeconds"],
        "reference_edges": len(reference_edges), "fnit_edges": len(fnit_edges),
        "edge_jaccard": len(common)/len(reference_edges.keys() | fnit_edges.keys())
                        if reference_edges or fnit_edges else 1.0,
        "common_edge_z_mae": float(np.mean([abs(reference_edges[k]-fnit_edges[k])
                                              for k in common])) if common else None,
        "common_edge_z_max_error": max((abs(reference_edges[k]-fnit_edges[k])
                                        for k in common), default=None),
        "all_voxel_pair_z_count": len(original_z),
        "all_voxel_pair_z_mae": float(np.mean(np.abs(original_z-blocked_z))),
        "all_voxel_pair_z_max_error": float(np.max(np.abs(original_z-blocked_z))),
        "all_voxel_pair_cdt_disagreements": int(np.count_nonzero(
            (np.abs(original_z) > args.cdt) != (np.abs(blocked_z) > args.cdt))),
        "cluster_sizes_equal": reference_sizes == fnit_sizes,
        "cluster_rules_equal_on_reference_edges": reference_sizes == our_reference_sizes,
        "cluster_rules_equal_on_fnit_edges": reference_on_fnit_sizes == fnit_sizes,
        "reference_cluster_count": len(reference_sizes),
        "fnit_cluster_count": len(fnit_sizes),
        "fwhm_mean_original": max(2.0, float(np.mean(smoothness))),
        "fwhm_abs_difference": abs(max(2.0, float(np.mean(smoothness)))-meta["FWHMInVoxels"]),
        "fwhm_subject_max_error": max(smoothness_error),
        "representative_cluster_p_abs_error": abs(float(original_p)-fnit_p),
        "subject_block_projection_max_error": float((full_xy-batch_xy).abs().max()),
        "subject_block_sigma_max_error": float((full_sigma-batch_sigma).abs().max()),
        "subject_block_t_max_error": float((full_t[finite]-batch_t[finite]).abs().max()),
        "subject_block_z_max_error": float(np.max(np.abs(full_z[finite.cpu().numpy()]
                                                       -batch_z[finite.cpu().numpy()]))),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
