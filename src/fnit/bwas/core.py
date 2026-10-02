"""Blockwise BWAS GLM and six-dimensional random-field cluster inference.

Independent implementation of Gong et al. (Medical Image Analysis, 2018).
"""

from __future__ import annotations

import csv
import gc
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
from .cache import build_packed_cache, open_packed_cache
from .cluster_union import edge_components
from .packed_loader import PackedTileLoader


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
    roots = edge_components(edges, coords)
    members = {}
    for k, root in enumerate(roots):
        members.setdefault(int(root), []).append(k)
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
                  voxel_major: bool = False,
                  dtype: torch.dtype = torch.float64) -> torch.Tensor:
    count = stop-start
    lengths = np.array([matrices[subject].shape[1] if voxel_major
                        else len(matrices[subject]) for subject in range(start, stop)])
    max_time = int(lengths.max())
    if voxel_major:
        a = np.zeros((count, n_i, max_time), dtype=np.float32)
        b = np.zeros((count, n_j, max_time), dtype=np.float32)
    else:
        a = np.zeros((count, max_time, n_i), dtype=np.float32)
        b = np.zeros((count, max_time, n_j), dtype=np.float32)
    for row, subject in enumerate(range(start, stop)):
        series = matrices[subject]
        if voxel_major:
            a[row, :, :lengths[row]] = series[i:i+n_i]
            b[row, :, :lengths[row]] = series[j:j+n_j]
        else:
            a[row, :lengths[row]] = series[:, i:i+n_i]
            b[row, :lengths[row]] = series[:, j:j+n_j]
    left = torch.from_numpy(a)
    right = torch.from_numpy(b)
    if device.type == "cuda":
        left = left.pin_memory()
        right = right.pin_memory()
    left = left.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
    right = right.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
    r = (torch.bmm(left, right.transpose(1, 2)) if voxel_major else
         torch.bmm(left.transpose(1, 2), right))
    r /= torch.as_tensor(lengths, device=device, dtype=dtype)[:, None, None]
    r.masked_fill_(r > 0.9999, 0).clamp_(-0.999999, 0.999999)
    return r.atanh_().reshape(count, -1)


def _glm_blocks(blocks, subjects: int):
    """Keep the original 16-subject GLM reduction order across I/O batches."""
    pending = {}
    for column, start, stop, values in blocks:
        position = 0
        while start < stop:
            end = min(stop, (start//16+1)*16)
            offset = start % 16
            count = end-start
            part = values[position:position+count]
            if offset == 0 and (end % 16 == 0 or end == subjects):
                yield column, start, end, part
            else:
                if column not in pending:
                    pending[column] = values.new_empty((16, values.shape[1]))
                pending[column][offset:offset+count].copy_(part)
                if end % 16 == 0 or end == subjects:
                    yield column, start-offset, end, pending[column][:offset+count]
            del part
            position += count
            start = end
        del values


def _tune_subject_block(matrices, x: torch.Tensor, nvox: int,
                        device: torch.device):
    # Contiguous windows preserve the cohort's actual scan-length grouping.
    starts = sorted({int(start)//32*32 for start in
                     np.linspace(0, max(0, len(matrices)-32), 4)})
    indices = [subject for start in starts
               for subject in range(start, min(start+32, len(matrices)))]
    sample = [matrices[subject] for subject in indices]
    design = x[indices]
    width = min(4096, max(1, nvox//3))
    span = min(3*width, nvox)
    columns = [(j, min(width, nvox-j)) for j in range(width, span, width)] or [(0, width)]
    candidates = sorted({min(count, len(sample)) for count in (8, 16, 32)})
    timings = {count: [] for count in candidates}
    for repetition in range(3):
        for count in candidates if repetition < 2 else candidates[::-1]:
            shards = []
            for st in range(0, len(sample), count):
                group = sample[st:st+count]
                lengths = np.asarray([matrix.shape[1] for matrix in group], dtype=np.int32)
                packed = np.zeros((span, len(group), int(lengths.max())), dtype=np.float32)
                for subject, matrix in enumerate(group):
                    packed[:, subject, :lengths[subject]] = matrix[:span]
                shards.append((packed, lengths))
            loader = PackedTileLoader(shards, device, async_h2d=True)
            torch.cuda.synchronize(device)
            begin = perf_counter()
            xy_blocks = [torch.zeros((x.shape[1], width*n_j), device=device)
                         for _, n_j in columns]
            yy_blocks = [torch.zeros(width*n_j, device=device) for _, n_j in columns]
            for column, st, en, values in _glm_blocks(
                    loader.fisher_tiles(0, columns, width), len(sample)):
                xy_blocks[column].addmm_(design[st:en].T, values)
                yy_blocks[column] += values.square_().sum(0)
            del values
            for xy, yy in zip(xy_blocks, yy_blocks):
                sigma = yy-(xy*xy).sum(0)
                t = xy[0]/torch.sqrt(sigma)
                torch.nonzero(torch.abs(t) > 5).cpu()
            torch.cuda.synchronize(device)
            if repetition:
                timings[count].append(perf_counter()-begin)
            del loader, shards, xy_blocks, yy_blocks, xy, yy, sigma, t
            torch.cuda.empty_cache()
    return min(timings, key=lambda count: np.median(timings[count])), timings


def _tune_voxel_block(shards, x: torch.Tensor, nvox: int,
                      device: torch.device, fixed_width: int | None = None,
                      fixed_columns: int | None = None,
                      fixed_cache_row: bool | None = None):
    longest = max(range(len(shards)), key=lambda k: int(shards[k][1].max()))
    groups = list(range(min(8, len(shards))))
    if longest not in groups:
        groups.append(longest)
    starts = np.cumsum([0]+[len(lengths) for _, lengths in shards])
    sampled_shards = [shards[k] for k in groups]
    sampled_design = torch.cat([x[int(starts[k]):int(starts[k+1])]
                                for k in groups], dim=0)
    candidates = ([min(nvox, fixed_width)] if fixed_width is not None else
                  sorted({min(nvox, size) for size in
                          (512, 1024, 2048, 4096, 5120, 6144, 7168, 8192)}))
    timings = {}
    peak_allocated = peak_reserved = 0
    def pilot(width, columns, cache_row):
        torch.cuda.synchronize(device)
        loader = PackedTileLoader(shards if cache_row else sampled_shards, device,
                                  async_h2d=True, cache_row=cache_row)
        preload_seconds = 0.0
        if cache_row:
            preload_start = perf_counter()
            loader._load_row(0, width)
            preload_seconds = perf_counter()-preload_start
            loader.row_views = [loader.row_views[k] for k in groups]
            loader.shards = sampled_shards
        begin = perf_counter()
        xy_blocks = [torch.zeros((x.shape[1], width*n_j), device=device)
                     for _, n_j in columns]
        yy_blocks = [torch.zeros(width*n_j, device=device) for _, n_j in columns]
        for column, st, en, values in _glm_blocks(
                loader.fisher_tiles(0, columns, width), len(sampled_design)):
            xy_blocks[column].addmm_(sampled_design[st:en].T, values)
            yy_blocks[column] += values.square_().sum(0)
        del values
        for column, (_, n_j) in enumerate(columns):
            xy, yy = xy_blocks[column], yy_blocks[column]
            sigma = yy-(xy*xy).sum(0)
            t = (xy[0]/torch.sqrt(sigma)).reshape(width, n_j)
            selected = torch.nonzero(torch.abs(t) > 5)
            selected.cpu()
            xy_blocks[column] = yy_blocks[column] = None
            del xy, yy, sigma, t, selected
        torch.cuda.synchronize(device)
        # Extrapolate the sampled groups and amortize one full row preload.
        pilot_seconds = perf_counter()-begin
        seconds = pilot_seconds*len(x)/len(sampled_design)
        row_batches = ((nvox+width-1)//width+1)/(2*len(columns))
        return seconds + preload_seconds/row_batches, pilot_seconds, preload_seconds

    configurations = [(width, count, cache_row) for width in candidates
                      for count in ((1, 2) if fixed_columns is None else (fixed_columns,))
                      for cache_row in ((False, True) if fixed_cache_row is None
                                        else (fixed_cache_row,))]
    for repetition in range(3):
        order = configurations if repetition < 2 else configurations[::-1]
        for width, count, cache_row in order:
            key = f"{width}x{count}-row{int(cache_row)}"
            if repetition == 2 and key not in timings:
                continue
            j = width if nvox >= (count+1)*width else 0
            columns = [(start, min(width, nvox-start))
                       for start in range(j, min(j+count*width, nvox), width)]
            try:
                torch.cuda.reset_peak_memory_stats(device)
                seconds, measured, preload = pilot(width, columns, cache_row)
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                # Leave space below the allocator cap for final sparse output.
                if reserved < 16_500_000_000:
                    entry = timings.setdefault(key, {"block_size": width,
                        "column_tiles": count, "gpu_row_cache": cache_row,
                        "pilot_seconds": [], "row_preload_seconds": [],
                        "estimated_full_tile_seconds": [],
                        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0})
                    if repetition:
                        entry["pilot_seconds"].append(measured)
                        entry["row_preload_seconds"].append(preload)
                        entry["estimated_full_tile_seconds"].append(seconds)
                    entry["peak_allocated_bytes"] = max(entry["peak_allocated_bytes"], allocated)
                    entry["peak_reserved_bytes"] = max(entry["peak_reserved_bytes"], reserved)
                    if repetition:
                        entry["estimated_pairs_per_second"] = (
                            width*sum(n_j for _, n_j in columns) /
                            float(np.median(entry["estimated_full_tile_seconds"])))
                else:
                    timings.pop(key, None)
            except torch.cuda.OutOfMemoryError:
                timings.pop(key, None)
            finally:
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                gc.collect()
                torch.cuda.empty_cache()
    if not timings:
        raise MemoryError("no BWAS voxel block fits the 17 GB CUDA cap")
    best = max(timings.values(), key=lambda item: item["estimated_pairs_per_second"])
    return best["block_size"], best["column_tiles"], best["gpu_row_cache"], {
        "Candidates": timings, "SampledSubjects": len(sampled_design),
        "CohortSubjects": len(x), "PeakAllocatedBytes": peak_allocated,
        "PeakReservedBytes": peak_reserved}


def run_bwas(bids_root: str | Path, participants_tsv: str | Path,
             mask_file: str | Path, output_root: str | Path, *,
             phenotype: str, covariates: tuple[str, ...] = ("age", "sex"),
             cdt: float = 5.0, block_size: int | None = None,
             subject_block_size: int | None = None,
             num_workers: int = 1,
             device: str = "cuda:0", fwhm: float | None = None,
             validate_direct_ols: bool = False,
             cache_root: str | Path | None = None,
             column_tiles: int | None = None,
             gpu_row_cache: bool | None = None,
             _prepared_cache_dir: str | Path | None = None,
             _prepared_packed_cache_dir: str | Path | None = None) -> BWASResult:
    """Run one group BWAS on 2 mm clean BIDS Derivatives BOLD images."""
    start = perf_counter()
    if (not 0 < cdt < 20 or num_workers < 1 or
            (block_size is not None and block_size < 1) or
            (subject_block_size is not None and subject_block_size < 1) or
            column_tiles not in (None, 1, 2)):
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
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_parent = Path(cache_root).expanduser().resolve() if cache_root else output_root
    cache_parent.mkdir(parents=True, exist_ok=True)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        total_memory = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(17_000_000_000 / total_memory, 1.0), dev)
        torch.cuda.reset_peak_memory_stats(dev)
    coords = np.column_stack(np.where(mask))
    x = torch.as_tensor(design, dtype=torch.float32, device=dev)
    nuisance, _ = torch.linalg.qr(x[:, 1:], mode="reduced")
    phenotype_vector = x[:, 0] - nuisance @ (nuisance.T @ x[:, 0])
    x = torch.column_stack((phenotype_vector / torch.linalg.vector_norm(phenotype_vector),
                            nuisance))
    df = len(ids) - x.shape[1]
    t_threshold = float(stats.t.isf(stats.norm.sf(cdt), df-1))
    nvox = len(coords)
    edges = []
    validation = {"count": 0, "sum_abs_z_error": 0.0, "max_abs_z_error": 0.0,
                  "threshold_disagreements": 0, "reference_edges": 0,
                  "seconds": 0.0} if validate_direct_ols else None
    direct_inverse = np.linalg.pinv(design.T @ design) if validate_direct_ols else None
    direct_weights = direct_inverse @ design.T if validate_direct_ols else None
    direct_gpu_weights = (torch.as_tensor(direct_weights, device=dev)
                          if validate_direct_ols and dev.type == "cuda" else None)
    direct_gpu_design = (torch.as_tensor(design, device=dev)
                         if direct_gpu_weights is not None else None)
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
        tuning = {}
        if subject_block_size is None:
            if _prepared_packed_cache_dir is not None:
                manifest = json.loads((Path(_prepared_packed_cache_dir) /
                                       "manifest.json").read_text())
                subject_block_size = int(manifest["subject_block_size"])
            elif dev.type == "cuda":
                subject_block_size, timings = _tune_subject_block(
                    matrices, x, nvox, dev)
                tuning["SubjectBlockPilotSeconds"] = timings
            else:
                subject_block_size = 16
        cache_build_seconds = 0.0
        loader = None
        if dev.type == "cuda" and not validate_direct_ols:
            packed_dir = (Path(_prepared_packed_cache_dir) if _prepared_packed_cache_dir
                          else Path(scratch) / f"packed-b{subject_block_size}")
            if _prepared_packed_cache_dir is None and not packed_dir.exists():
                cache_build_seconds = build_packed_cache(
                    matrices, packed_dir, subject_block_size)
            matrices.clear()
            shards = open_packed_cache(packed_dir, len(ids), nvox, subject_block_size)
            loader = PackedTileLoader(shards, dev, async_h2d=True, measure=True,
                                      cache_row=bool(gpu_row_cache))
        if loader is not None and (block_size is None or column_tiles is None or
                                   gpu_row_cache is None):
            block_size, tuned_columns, cache_row, timings = _tune_voxel_block(
                shards, x, nvox, dev, fixed_width=block_size,
                fixed_columns=column_tiles, fixed_cache_row=gpu_row_cache)
            tuning["VoxelBlockPilot"] = timings
            loader.cache_row = cache_row
            column_tiles = tuned_columns
        elif block_size is None:
            block_size = 128
        stages = {key: 0.0 for key in ("cache_read", "h2d", "correlation",
                                      "fisher_glm", "threshold", "clustering")}
        nblocks = (nvox+block_size-1)//block_size
        total_tiles = nblocks*(nblocks+1)//2
        completed_tiles = 0
        column_tiles = (column_tiles or 1) if loader is not None else 1
        print(f"BWAS blocks: voxel={block_size}, subjects={subject_block_size}, "
              f"columns={column_tiles}, GPU row={bool(getattr(loader, 'cache_row', False))}",
              flush=True)
        for i in range(0, nvox, block_size):
            n_i = min(block_size, nvox-i)
            for first_j in range(i, nvox, block_size*column_tiles):
                columns = [(j, min(block_size, nvox-j))
                           for j in range(first_j, min(first_j+block_size*column_tiles,
                                                      nvox), block_size)]
                xy_blocks = [torch.zeros((x.shape[1], n_i*n_j), device=dev)
                             for _, n_j in columns]
                yy_blocks = [torch.zeros(n_i*n_j, device=dev) for _, n_j in columns]
                j, n_j = columns[0]
                direct_y = (np.empty((len(ids), n_i*n_j), dtype=np.float64)
                            if validate_direct_ols else None)
                if loader is None:
                    blocks = ((0, _st, min(_st+subject_block_size, len(ids)),
                               _fisher_block(matrices, _st,
                                             min(_st+subject_block_size, len(ids)),
                                             i, j, n_i, n_j, dev, voxel_major=True,
                                             dtype=torch.float32))
                              for _st in range(0, len(ids), subject_block_size))
                else:
                    blocks = _glm_blocks(loader.fisher_tiles(i, columns, n_i), len(ids))
                glm_events = []
                for column, st, en, values in blocks:
                    xy, yy = xy_blocks[column], yy_blocks[column]
                    glm_start = perf_counter()
                    if loader is not None:
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                    xy.addmm_(x[st:en].T, values)
                    if direct_y is not None:
                        yy += (values * values).sum(dim=0)
                        direct_y[st:en] = values.cpu().numpy()
                    else:
                        yy += values.square_().sum(dim=0)
                    if loader is not None:
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_end.record()
                        glm_events.append((gpu_start, gpu_end))
                    else:
                        stages["fisher_glm"] += perf_counter()-glm_start
                    del values
                del xy, yy
                if loader is not None:
                    measured = loader.drain_timings()
                    for name, duration in measured.items():
                        stages["fisher_glm" if name == "fisher" else name] += duration
                    stages["fisher_glm"] += sum(
                        first.elapsed_time(last) for first, last in glm_events) / 1000
                for column, (j, n_j) in enumerate(columns):
                    xy, yy = xy_blocks[column], yy_blocks[column]
                    threshold_start = perf_counter()
                    sigma = yy - (xy * xy).sum(dim=0)
                    t_gpu = (xy[0] / torch.sqrt(sigma / df)).reshape(n_i, n_j)
                    if direct_y is not None:
                        t = t_gpu.cpu().numpy()
                        check_start = perf_counter()
                        z = stats.norm.ppf(stats.t.cdf(t, df-1))
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
                                residual = observed - direct_gpu_design @ direct_beta
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
                    if dev.type == "cuda" and direct_y is None:
                        keep = torch.abs(t_gpu) > t_threshold
                        if i == j:
                            keep.triu_(diagonal=1)
                        selected = torch.nonzero(keep)
                        rr = selected[:, 0].cpu().numpy()
                        cc = selected[:, 1].cpu().numpy()
                        t_selected = t_gpu[keep].cpu().numpy()
                    else:
                        if direct_y is None:
                            t = t_gpu.cpu().numpy()
                        if i == j:
                            rr, cc = np.where(np.triu(np.abs(t) > t_threshold, k=1))
                        else:
                            rr, cc = np.where(np.abs(t) > t_threshold)
                        t_selected = t[rr, cc]
                    selected_z = stats.norm.ppf(stats.t.cdf(t_selected, df-1))
                    edges.extend((i+int(a), j+int(b), float(value))
                                 for a, b, value in zip(rr, cc, selected_z))
                    stages["threshold"] += perf_counter()-threshold_start
                    completed_tiles += 1
                    if completed_tiles % 10 == 0 or completed_tiles == total_tiles:
                        print(f"BWAS voxel tiles {completed_tiles}/{total_tiles}", flush=True)
                    xy_blocks[column] = yy_blocks[column] = None
                    del xy, yy, sigma, t_gpu
                    if dev.type == "cuda" and direct_y is None:
                        del keep, selected
                del xy_blocks, yy_blocks, direct_y
        matrices.clear()
    cluster_start = perf_counter()
    labels, table = _clusters(edges, coords, cdt, width)
    stages["clustering"] = perf_counter()-cluster_start
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
        "GLMSubjectBlockSize": 16 if loader is not None else subject_block_size,
        "ColumnTilesPerRowBatch": column_tiles,
        "GPUResidentRowBlock": bool(getattr(loader, "cache_row", False)),
        "Precision": "float32", "TF32Enabled": False,
        "DesignOrthogonalization": "QR",
        "RuntimeVersions": {"numpy": np.__version__, "scipy": scipy.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda},
        "PreparationWorkers": num_workers,
        "PeakCUDAAllocatedBytes": max(torch.cuda.max_memory_allocated(dev),
            tuning.get("VoxelBlockPilot", {}).get("PeakAllocatedBytes", 0))
        if dev.type == "cuda" else None,
        "PeakCUDAReservedBytes": max(torch.cuda.max_memory_reserved(dev),
            tuning.get("VoxelBlockPilot", {}).get("PeakReservedBytes", 0))
        if dev.type == "cuda" else None,
        "PackedCacheBuildSeconds": cache_build_seconds,
        "StageSeconds": stages,
        "BlockTuning": tuning,
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
