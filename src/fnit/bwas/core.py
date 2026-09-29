"""Blockwise BWAS GLM and six-dimensional random-field cluster inference.

Independent implementation of Gong et al. (Medical Image Analysis, 2018).
"""

from __future__ import annotations

import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import scipy.signal
from scipy import special, stats
import torch

from .. import __version__


@dataclass(frozen=True)
class BWASResult:
    clusters: Path
    edges: Path
    ma_map: Path
    metadata: Path
    subjects: int
    voxels: int
    suprathreshold_edges: int


def _inputs(bids_root: Path, participants: Path, mask_path: Path,
            phenotype: str, covariates: tuple[str, ...]):
    with participants.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"participant_id", phenotype, *covariates}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"participants TSV requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("participants TSV is empty")
    ids = [row["participant_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not x.startswith("sub-") for x in ids):
        raise ValueError("participant_id must be unique BIDS sub-* identifiers")
    files = []
    if not (bids_root / "dataset_description.json").is_file():
        raise ValueError("bids_root must contain dataset_description.json")
    for subject in ids:
        matches = list((bids_root / subject).glob(
            "**/*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one clean 2 mm BOLD for {subject}; found {len(matches)}")
        files.append(matches[0])
    design = np.array([[float(row[phenotype]),
                        *(float(row[name]) for name in covariates), 1.0]
                       for row in rows], dtype=np.float64)
    if not np.isfinite(design).all() or len(rows) <= design.shape[1]:
        raise ValueError("design requires finite values and more subjects than columns")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("phenotype, covariates and intercept must have full rank")
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj) != 0
    if mask.ndim != 3 or not mask.any():
        raise ValueError("mask must be a nonempty 3D NIfTI")
    if not np.allclose(mask_img.header.get_zooms()[:3], 2.0, atol=0.01):
        raise ValueError("mask must be on a 2 mm grid")
    for file in files:
        img = nib.load(str(file))
        if img.ndim != 4 or img.shape[:3] != mask.shape or img.shape[3] < 4:
            raise ValueError(f"BOLD shape does not match 3D mask: {file}")
        if not np.allclose(img.affine, mask_img.affine, atol=1e-3):
            raise ValueError(f"BOLD affine does not match mask: {file}")
    tasks = {part for file in files for part in file.name.split("_")
             if part.startswith("task-")}
    if len(tasks) != 1:
        raise ValueError("all BOLD files must have the same BIDS task entity")
    return files, ids, design, mask_img, mask, tasks.pop()


def _smoothness(data: np.ndarray) -> float:
    """Original BWAS temporal detrending and per-frame spatial FWHM, in voxels."""
    valid = np.any(data != 0, axis=3)
    series = scipy.signal.detrend(data[valid], axis=1).astype(np.float32, copy=False)
    series[series == 0] = np.nan
    variance = np.nanvar(series, axis=0)
    index = np.full(data.shape[:3], -1, dtype=np.int32)
    index[valid] = np.arange(valid.sum())
    gradient = []
    for axis in range(3):
        first = [slice(None)]*3
        second = [slice(None)]*3
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        left, right = index[tuple(first)], index[tuple(second)]
        adjacent = (left >= 0) & (right >= 0)
        difference = series[right[adjacent]]-series[left[adjacent]]
        gradient.append(np.nanvar(difference, axis=0))
    ratio = np.cbrt(gradient[0] * gradient[1] * gradient[2]) / (2 * variance)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwhm = np.sqrt(-2 * np.log(2) / np.log1p(-ratio))
    return float(np.nanmean(fwhm))


def _ec_density(t: float) -> np.ndarray:
    a, b = 4 * np.log(2), np.exp(-t * t / 2)
    return np.array([
        stats.norm.sf(t), a**0.5 / (2*np.pi) * b,
        a / (2*np.pi)**1.5 * b * t,
        a**1.5 / (2*np.pi)**2 * b * (t*t-1),
        a**2 / (2*np.pi)**2.5 * b * (t**3-3*t),
        a**2.5 / (2*np.pi)**3 * b * (t**4-6*t*t+3),
        a**3 / (2*np.pi)**3.5 * b * (t**5-10*t**3+15*t),
    ])


def _cluster_p(nvoxels: int, cdt: float, size: int, fwhm: float) -> tuple[float, float]:
    n = nvoxels / np.sqrt(2)
    radius = (n / (4*np.pi/3)) ** (1/3)
    r = np.array([1, 4*radius/fwhm, 2*np.pi*(radius/fwhm)**2,
                  n/fwhm**3])
    el = float(sum(r[i]*r[j]*_ec_density(cdt)[i+j]
                   for i in range(4) for j in range(4)))
    if el < 0:
        return 1.0, 1.0
    en = n*n*stats.norm.sf(abs(cdt))
    phi = (special.gamma(4) * el / en) ** (1/3)
    uncorrected = float(np.exp(-phi * size**(1/3)))
    return float(-np.expm1(-el*uncorrected)), uncorrected


def _clusters(edges: list[tuple[int, int, float]], coords: np.ndarray,
              cdt: float, fwhm: float):
    lookup = {(i, j): k for k, (i, j, _) in enumerate(edges)}
    voxel_lookup = {tuple(coord): i for i, coord in enumerate(coords)}
    neighbors = []
    offsets = np.array([(x, y, z) for x in (-1, 0, 1)
                        for y in (-1, 0, 1) for z in (-1, 0, 1)
                        if x*x+y*y+z*z < 1.5**2])
    for coord in coords:
        neighbors.append([voxel_lookup[tuple(other)] for other in coord + offsets
                          if tuple(other) in voxel_lookup])
    parent = np.arange(len(edges))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k, (i, j, _) in enumerate(edges):
        for ni in neighbors[i]:
            for nj in neighbors[j]:
                other = lookup.get((ni, nj))
                if other is not None:
                    parent[find(other)] = find(k)
    members = {}
    for k in range(len(edges)):
        members.setdefault(find(k), []).append(k)
    ordered = sorted(members.values(), key=lambda m: (-len(m), m[0]))
    labels = np.empty(len(edges), dtype=np.int32)
    table = []
    for number, member in enumerate(ordered, 1):
        labels[member] = number
        selected = [edges[k] for k in member]
        p, raw_p = _cluster_p(len(coords), cdt, len(member), fwhm)
        table.append((number, len(member), p, raw_p,
                      max(edge[2] for edge in selected),
                      len({edge[0] for edge in selected}),
                      len({edge[1] for edge in selected})))
    return labels, table


def _fisher_block(matrices, start: int, stop: int, i: int, j: int,
                  n_i: int, n_j: int, device: torch.device,
                  voxel_major: bool = False) -> torch.Tensor:
    count = stop-start
    lengths = np.array([matrices[subject].shape[1] if voxel_major
                        else len(matrices[subject]) for subject in range(start, stop)])
    max_time = int(lengths.max())
    a = np.zeros((count, max_time, n_i), dtype=np.float32)
    b = np.zeros((count, max_time, n_j), dtype=np.float32)
    for row, subject in enumerate(range(start, stop)):
        series = matrices[subject]
        if voxel_major:
            a[row, :lengths[row]] = series[i:i+n_i].T
            b[row, :lengths[row]] = series[j:j+n_j].T
        else:
            a[row, :lengths[row]] = series[:, i:i+n_i]
            b[row, :lengths[row]] = series[:, j:j+n_j]
    left = torch.from_numpy(a).to(device=device, dtype=torch.float64)
    right = torch.from_numpy(b).to(device=device, dtype=torch.float64)
    r = torch.bmm(left.transpose(1, 2), right)
    r /= torch.as_tensor(lengths, device=device, dtype=torch.float64)[:, None, None]
    r = torch.where(r > 0.9999, 0, r).clamp(-0.999999, 0.999999)
    return torch.atanh(r).reshape(count, -1)


def run_bwas(bids_root: str | Path, participants_tsv: str | Path,
             mask_file: str | Path, output_root: str | Path, *,
             phenotype: str, covariates: tuple[str, ...] = ("age", "sex"),
             cdt: float = 5.0, block_size: int = 128,
             subject_block_size: int = 16,
             num_workers: int = 1,
             device: str = "cuda:0", fwhm: float | None = None,
             validate_direct_ols: bool = False,
             cache_root: str | Path | None = None,
             _prepared_cache_dir: str | Path | None = None) -> BWASResult:
    """Run one group BWAS on 2 mm clean BIDS Derivatives BOLD images."""
    start = perf_counter()
    if not 0 < cdt < 20 or min(block_size, subject_block_size, num_workers) < 1:
        raise ValueError("cdt must be in (0, 20) and block sizes/workers must be positive")
    files, ids, design, mask_img, mask, task = _inputs(
        Path(bids_root), Path(participants_tsv), Path(mask_file), phenotype,
        tuple(covariates))
    if len(files) > 512:
        try:
            import resource
        except ImportError:
            pass
        else:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            required = len(files)+128
            if hard != resource.RLIM_INFINITY and required > hard:
                raise OSError(f"open-file hard limit {hard} is below required {required}")
            if required > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"BWAS output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_parent = Path(cache_root).expanduser().resolve() if cache_root else output_root
    cache_parent.mkdir(parents=True, exist_ok=True)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    coords = np.column_stack(np.where(mask))
    x = torch.as_tensor(design, dtype=torch.float64, device=dev)
    inv = torch.linalg.pinv(x.T @ x)
    df = len(ids) - x.shape[1]
    scale = torch.sqrt(inv[0, 0] / df)
    nvox = len(coords)
    edges = []
    validation = {"count": 0, "sum_abs_z_error": 0.0, "max_abs_z_error": 0.0,
                  "threshold_disagreements": 0, "reference_edges": 0,
                  "seconds": 0.0} if validate_direct_ols else None
    direct_inverse = np.linalg.pinv(design.T @ design) if validate_direct_ols else None
    direct_weights = direct_inverse @ design.T if validate_direct_ols else None
    direct_gpu_weights = (torch.as_tensor(direct_weights, device=dev)
                          if validate_direct_ols and dev.type == "cuda" else None)
    if _prepared_cache_dir is not None and fwhm is None:
        raise ValueError("a prepared cache requires the measured fwhm")
    cache_context = (nullcontext(Path(_prepared_cache_dir)) if _prepared_cache_dir
                     else TemporaryDirectory(prefix="bwas-cache-", dir=cache_parent))
    with cache_context as scratch:
        matrices, widths = [], []
        def prepare_subject(item):
            subject, file = item
            data = np.asarray(nib.load(str(file)).dataobj, dtype=np.float32)
            subject_width = _smoothness(data) if fwhm is None else None
            series = data[mask].T.copy()
            del data
            series -= series.mean(axis=0)
            std = series.std(axis=0)
            if not np.isfinite(series).all() or np.any(std <= 0):
                raise ValueError(f"nonfinite or constant masked BOLD voxels: {file}")
            series /= std
            cache_file = Path(scratch) / f"subject-{subject}.npy"
            np.save(cache_file, np.ascontiguousarray(series.T))
            return cache_file, subject_width

        if _prepared_cache_dir is None:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for subject, (cache_file, subject_width) in enumerate(
                        executor.map(prepare_subject, enumerate(files))):
                    if subject_width is not None:
                        widths.append(subject_width)
                    matrices.append(np.load(cache_file, mmap_mode="r"))
                    if (subject+1) % 100 == 0 or subject+1 == len(files):
                        print(f"BWAS prepared {subject+1}/{len(files)} BOLD images", flush=True)
        else:
            for subject, file in enumerate(files):
                series = np.load(Path(scratch) / f"subject-{subject}.npy", mmap_mode="r")
                if series.shape != (nvox, nib.load(str(file)).shape[3]):
                    raise ValueError(f"prepared cache shape differs from BOLD: {file}")
                matrices.append(series)
        width = float(max(2.0, np.mean(widths))) if fwhm is None else float(fwhm)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("fwhm must be positive and finite")
        nblocks = (nvox+block_size-1)//block_size
        total_tiles = nblocks*(nblocks+1)//2
        completed_tiles = 0
        for i in range(0, nvox, block_size):
            for j in range(i, nvox, block_size):
                n_i, n_j = min(block_size, nvox-i), min(block_size, nvox-j)
                xy = torch.zeros((x.shape[1], n_i*n_j), device=dev, dtype=torch.float64)
                yy = torch.zeros(n_i*n_j, device=dev, dtype=torch.float64)
                direct_y = (np.empty((len(ids), n_i*n_j), dtype=np.float64)
                            if validate_direct_ols else None)
                for st in range(0, len(ids), subject_block_size):
                    en = min(st+subject_block_size, len(ids))
                    values = _fisher_block(matrices, st, en, i, j, n_i, n_j,
                                           dev, voxel_major=True)
                    xy += x[st:en].T @ values
                    yy += (values * values).sum(dim=0)
                    if direct_y is not None:
                        direct_y[st:en] = values.cpu().numpy()
                beta = inv @ xy
                sigma = yy - (beta * xy).sum(dim=0)
                t = (beta[0] / (torch.sqrt(sigma) * scale)).cpu().numpy()
                z = stats.norm.ppf(stats.t.cdf(t, df-1)).reshape(n_i, n_j)
                if direct_y is not None:
                    check_start = perf_counter()
                    flat_z = z.reshape(-1)
                    for lo in range(0, n_i*n_j, 65536):
                        hi = min(lo+65536, n_i*n_j)
                        if direct_gpu_weights is None:
                            observed = direct_y[:, lo:hi]
                            direct_beta = direct_weights @ observed
                            residual = observed - design @ direct_beta
                            direct_sigma = np.sum(residual*residual, axis=0)
                            direct_t = direct_beta[0] / np.sqrt(
                                direct_sigma * (direct_inverse[0, 0] / df))
                        else:
                            observed = torch.as_tensor(
                                np.ascontiguousarray(direct_y[:, lo:hi]), device=dev)
                            direct_beta = direct_gpu_weights @ observed
                            residual = observed - x @ direct_beta
                            direct_sigma = (residual*residual).sum(dim=0)
                            direct_t = (direct_beta[0] / torch.sqrt(
                                direct_sigma * (direct_inverse[0, 0] / df))).cpu().numpy()
                        direct_z = stats.norm.ppf(stats.t.cdf(direct_t, df-1))
                        positions = np.arange(lo, hi)
                        keep = ((i + positions // n_j) < (j + positions % n_j))
                        keep &= np.isfinite(direct_z) & np.isfinite(flat_z[lo:hi])
                        difference = np.abs(flat_z[lo:hi][keep]-direct_z[keep])
                        validation["count"] += int(keep.sum())
                        validation["sum_abs_z_error"] += float(difference.sum())
                        if len(difference):
                            validation["max_abs_z_error"] = max(
                                validation["max_abs_z_error"], float(difference.max()))
                        validation["threshold_disagreements"] += int(np.count_nonzero(
                            (np.abs(flat_z[lo:hi][keep]) > cdt) !=
                            (np.abs(direct_z[keep]) > cdt)))
                        validation["reference_edges"] += int(np.count_nonzero(
                            np.abs(direct_z[keep]) > cdt))
                    validation["seconds"] += perf_counter()-check_start
                if i == j:
                    rr, cc = np.where(np.triu(np.abs(z) > cdt, k=1))
                else:
                    rr, cc = np.where(np.abs(z) > cdt)
                edges.extend((i+int(a), j+int(b), float(z[a, b])) for a, b in zip(rr, cc))
                completed_tiles += 1
                if completed_tiles % 50 == 0 or completed_tiles == total_tiles:
                    print(f"BWAS voxel tiles {completed_tiles}/{total_tiles}", flush=True)
        matrices.clear()
    labels, table = _clusters(edges, coords, cdt, width)
    group_dir = output_root / "group" / "func"
    group_dir.mkdir(parents=True)
    prefix = f"{task}_space-MNI152NLin6Asym_res-2"
    cluster_file = group_dir / f"{prefix}_desc-BWASclusters_stat.tsv"
    edge_file = group_dir / f"{prefix}_desc-BWASedges_relmat.tsv.gz"
    ma_file = group_dir / f"{prefix}_desc-BWASMA_statmap.nii.gz"
    metadata_file = group_dir / f"{prefix}_desc-BWASMA_statmap.json"
    with cluster_file.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["cluster", "edges", "p_fwer", "p_uncorrected",
                         "max_z", "region1_voxels", "region2_voxels"])
        writer.writerows(table)
    with gzip.open(edge_file, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["voxel1_i", "voxel1_j", "voxel1_k", "voxel2_i",
                         "voxel2_j", "voxel2_k", "z", "cluster"])
        for (i, j, z), label in zip(edges, labels):
            writer.writerow([*coords[i], *coords[j], z, int(label)])
    ma = np.zeros(mask.shape, dtype=np.float32)
    significant = {row[0] for row in table if row[2] < 0.05}
    for (i, j, _), label in zip(edges, labels):
        if int(label) in significant:
            ma[tuple(coords[i])] += 1
            ma[tuple(coords[j])] += 1
    header = mask_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(ma, mask_img.affine, header), str(ma_file))
    metadata = {
        "Name": "FNIT BWAS", "BIDSVersion": "1.11.1", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fudan-neuroimaging-toolkit", "Version": __version__}],
        "Phenotype": phenotype, "Covariates": list(covariates),
        "Participants": ids, "CDT": cdt, "FWHMInVoxels": width,
        "DegreesOfFreedom": int(df), "VoxelCount": nvox,
        "SuprathresholdEdgeCount": len(edges), "Device": str(dev),
        "VoxelBlockSize": block_size, "SubjectBlockSize": subject_block_size,
        "PreparationWorkers": num_workers,
        "PeakCUDAAllocatedBytes": torch.cuda.max_memory_allocated(dev)
        if dev.type == "cuda" else None,
        "ElapsedSeconds": perf_counter()-start,
        "Source": "https://github.com/weikanggong/BWAS",
    }
    if validation is not None:
        metadata["DirectOLSValidation"] = {
            "EdgeCount": validation["count"],
            "ZMeanAbsoluteDifference": (validation["sum_abs_z_error"] /
                                        validation["count"]),
            "ZMaxAbsoluteDifference": validation["max_abs_z_error"],
            "CDTDisagreements": validation["threshold_disagreements"],
            "ReferenceSuprathresholdEdges": validation["reference_edges"],
            "ReferenceRegressionSeconds": validation["seconds"],
        }
    (output_root / "dataset_description.json").write_text(json.dumps({
        key: metadata[key] for key in ("Name", "BIDSVersion", "DatasetType", "GeneratedBy")
    }, indent=2) + "\n")
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return BWASResult(cluster_file, edge_file, ma_file, metadata_file,
                      len(ids), nvox, len(edges))
