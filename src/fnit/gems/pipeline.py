"""One-call native-space aggregation of multiple GEMS subregion atlases."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import logging
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch

from .._dmri import configure_device
from .._nib import FNITNifti1Image, new_image
from .atlas import GEMSAtlas
from .core import TorchGEMSResult
from .gaussian import GaussianParameters
from .precision import restore_parallel_precision, set_parallel_precision


logger = logging.getLogger(__name__)


@dataclass
class SubregionResult:
    labels: FNITNifti1Image
    label_table: dict[int, str]
    structure_results: dict[str, TorchGEMSResult]
    confidence: torch.Tensor
    initialization: dict[str, dict]
    volumes: dict[int, dict[str, float]] | None = None
    label_metadata: dict[int, "SubregionLabel"] | None = None
    input_source: str | None = None
    output_files: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def save(self, output_dir: str | Path, *, save_highres: bool = True,
             save_posteriors: bool = False) -> dict[str, Path]:
        """Save native labels, metadata, volumes, report and optional fine grids."""
        from .output import save_subregion_result
        self.output_files = save_subregion_result(
            self, output_dir, save_highres=save_highres, save_posteriors=save_posteriors)
        return self.output_files

    def mask(self, label: int | str) -> np.ndarray:
        if isinstance(label, str):
            matches = [idx for idx, name in self.label_table.items() if name == label]
            if len(matches) != 1:
                raise KeyError(label)
            label = matches[0]
        return np.asanyarray(self.labels.dataobj) == int(label)


@dataclass(frozen=True)
class SubregionLabel:
    id: int
    name: str
    parent: str
    family: str
    hemisphere: str | None
    source: str


_CANONICAL = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")

# Keep the final merge in ``_CANONICAL`` order, but submit the longest recipes
# first.  With two workers this avoids filling both streams with the short
# brainstem job before the two high-resolution hippocampal jobs are queued.
# The values are only a scheduling hint; recipe output and numerical settings
# are unchanged.
_PARALLEL_PRIORITY = {
    "thalamus": 0,
    "hippo-amygdala-left": 1,
    "hippo-amygdala-right": 2,
    "brainstem": 3,
}


def _parallel_batches(selected: list[str], workers: int) -> list[list[str]]:
    """Build conservative CUDA batches under the 20 GB process budget.

    The hippocampus/amygdala recipes use substantially larger 0.333-mm
    working grids than brainstem and thalamus.  Running both high-resolution
    recipes together exceeded the measured driver budget on H100, so they are
    deliberately kept in separate batches.  The two lighter recipes can
    overlap and still preserve the canonical merge order.
    """
    ordered = sorted(selected, key=lambda name: _PARALLEL_PRIORITY.get(name, 99))
    light = [name for name in ordered if not name.startswith("hippo-amygdala-")]
    heavy = [name for name in ordered if name.startswith("hippo-amygdala-")]
    batches: list[list[str]] = []
    if light:
        for index in range(0, len(light), max(1, workers)):
            batches.append(light[index:index + max(1, workers)])
    batches.extend([name_list] for name_list in heavy)
    return batches


def _merge_native(combined: np.ndarray, best_conf: np.ndarray,
                  candidate: np.ndarray, confidence: np.ndarray,
                  support: np.ndarray) -> None:
    take = (candidate != 0) & support
    take &= (combined == 0) | (confidence > best_conf)
    combined[take] = candidate[take]
    best_conf[take] = confidence[take]


def _outcome_on_native_grid(outcome, context, native_image):
    same_grid = context.image.shape == native_image.shape and np.allclose(
        context.image.affine, native_image.affine, atol=1e-5, rtol=0)
    if same_grid:
        return outcome.native_labels, outcome.native_confidence, outcome.native_support
    grid = (native_image.shape, native_image.affine)
    # Preserve each recipe's standard processing-grid winner. Confidence and
    # support follow the same nearest processing voxel on the original grid.
    def sample(array, dtype):
        source = nib.Nifti1Image(np.asarray(array, dtype=dtype), context.image.affine)
        return np.asarray(resample_from_to(source, grid, order=0).dataobj, dtype=dtype)
    labels = sample(outcome.native_labels, np.int32)
    confidence = sample(outcome.native_confidence, np.float32)
    support = sample(outcome.native_support, np.uint8).astype(bool)
    return labels, confidence, support


def _expand_structures(structures) -> list[str]:
    requested = _CANONICAL if structures == "all" else (
        [structures] if isinstance(structures, str) else list(structures))
    expanded = []
    for name in requested:
        members = (_CANONICAL if name == "all" else
                   ("hippo-amygdala-left", "hippo-amygdala-right") if name == "hippo-amygdala" else (name,))
        for member in members:
            if member not in expanded:
                expanded.append(member)
    if not expanded:
        raise ValueError("structures must not be empty")
    unknown = sorted(set(expanded) - set(_CANONICAL))
    if unknown:
        raise ValueError(f"Unknown subregion structures: {unknown}; choose from {_CANONICAL}")
    return expanded


def _cpu_thread_scoped(function):
    """CPU calls restore Torch threads and constrain the caller's Numba mask.

    CUDA retains the existing pipeline path, including its thread policy.
    Native BLAS threads and process affinity remain the caller's responsibility.
    """
    @wraps(function)
    def scoped(*args, **kwargs):
        device = torch.device(kwargs.get("device", "cuda:0"))
        if device.type != "cpu":
            return function(*args, **kwargs)
        threads = kwargs.get("threads", 4)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        import numba
        before_torch = torch.get_num_threads()
        before_numba = numba.get_num_threads()
        # A previously imported Numba pool cannot grow beyond its capacity.
        # Preserve the existing API's Torch budget while capping that pool.
        numba_threads = min(threads, int(numba.config.NUMBA_NUM_THREADS))
        try:
            torch.set_num_threads(threads)
            numba.set_num_threads(numba_threads)
            return function(*args, **kwargs)
        finally:
            try:
                numba.set_num_threads(before_numba)
            finally:
                torch.set_num_threads(before_torch)
    return scoped


@_cpu_thread_scoped
def segment_4_subregions(
    t1: str | Path | nib.spatialimages.SpatialImage,
    atlas_root: str | Path | None = None,
    *,
    structures: str | list[str] | tuple[str, ...] = "all",
    coarse_segmentation: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    cortical_parcellation: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    wmparc: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    synthseg_weights: str | Path | None = None,
    synthseg_parc_weights: str | Path | None = None,
    device: str | torch.device = "cuda:0",
    threads: int = 4,
    optimization: str = "fast",
    parallel_regions: bool = False,
    max_parallel_regions: int = 2,
    output_dir: str | Path | None = None,
    save_highres: bool = True,
    save_posteriors: bool = False,
) -> SubregionResult:
    """Segment one raw T1 end to end; optionally save all outputs in one call.

    CPU calls temporarily constrain Torch intraop and the current Numba mask,
    restoring both on success or failure. CUDA keeps its existing thread path
    unless ``parallel_regions=True``; then independent recipes use separate
    CUDA streams and are merged in canonical structure order.
    """
    started = monotonic()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    if optimization not in ("fast", "balanced"):
        raise ValueError("optimization must be 'fast' or 'balanced'")
    if not isinstance(parallel_regions, bool):
        raise ValueError("parallel_regions must be a bool")
    if (isinstance(max_parallel_regions, bool)
            or not isinstance(max_parallel_regions, int)
            or max_parallel_regions < 1):
        raise ValueError("max_parallel_regions must be a positive integer")
    selected = _expand_structures(structures)
    root = Path(atlas_root) if atlas_root is not None else None
    if root is None:
        from .setup import configured_subregion_root
        from ..weights import cache_dir
        root = configured_subregion_root() or cache_dir() / "subregion_atlases"
        if not all((root / name / "AtlasMesh.gz").is_file()
                   for name in selected if name in _CANONICAL):
            from .setup import prepare_subregion_atlases
            prepare_subregion_atlases(root, device="cpu")
    missing = [name for name in selected if not (root / name / "AtlasMesh.gz").is_file()]
    if missing:
        raise FileNotFoundError(f"Subregion atlas packs not found: {missing}")
    from .context import SubregionContext
    from .recipes import make_recipe
    device = configure_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    use_parallel_regions = (parallel_regions and device.type == "cuda"
                            and len(selected) > 1 and max_parallel_regions > 1)
    parallel_workers = (min(max_parallel_regions, len(selected))
                        if use_parallel_regions else 1)
    preprocessing_started = monotonic()
    context = SubregionContext.prepare(
        t1, need_coarse=True, need_parc=any(name.startswith("hippo-amygdala") for name in selected),
        coarse_segmentation=coarse_segmentation, cortical_parcellation=cortical_parcellation,
        wmparc=wmparc, synthseg_weights=synthseg_weights,
        synthseg_parc_weights=synthseg_parc_weights, device=device)
    if coarse_segmentation is None:
        from .preprocessing import prepare_automatic_raw_input
        context = prepare_automatic_raw_input(context, device=device, threads=threads)
    native_image = getattr(context, "native_image", None)
    if native_image is None:
        native_image = context.image
    combined = np.zeros(native_image.shape, np.int32)
    best_conf = np.zeros(native_image.shape, np.float32)
    table: dict[int, str] = {0: "Unknown"}
    metadata: dict[int, SubregionLabel] = {}
    volumes: dict[int, dict[str, float]] = {}
    detailed: dict[str, TorchGEMSResult] = {}
    reports: dict[str, dict] = {"shared_preprocessing": {**context.metadata,
        "seconds": monotonic() - preprocessing_started,
        "coarse_source": context.metadata.get(
            "coarse_source", "provided" if coarse_segmentation is not None else None),
        "cortical_parcellation_source": context.metadata.get(
            "cortical_parcellation_source", "provided" if cortical_parcellation is not None else None),
        "model_calls": context.metadata.get("model_calls", {"SynthSeg": 0, "SynthSegPlus": 0}),
        "wmparc_source": "provided" if wmparc is not None else
                          "proxy" if context.wmparc_proxy is not None else None,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda" else None,
        "parallel_regions_requested": parallel_regions,
        "parallel_regions": use_parallel_regions,
        "max_parallel_regions_requested": max_parallel_regions,
        "max_parallel_regions": parallel_workers,
    }}
    voxel_volume = abs(np.linalg.det(native_image.affine[:3, :3]))
    if use_parallel_regions:
        # Preprocessing runs on the default stream. Establish one dependency
        # before workers start; recipe-local ticks then synchronize only their
        # own streams.
        torch.cuda.current_stream(device).synchronize()

    def run_recipe(name):
        recipe = make_recipe(name, root)
        if hasattr(recipe, "set_optimization_profile"):
            recipe.set_optimization_profile(optimization)
        logger.info("Starting %s", name)
        recipe_started = monotonic()
        if use_parallel_regions:
            # A separate stream lets independent region solvers overlap while
            # keeping all shared inputs read-only. The stream is synchronized
            # before the future completes so CPU materialization below never
            # races an outstanding device write.
            stream = torch.cuda.Stream(device=device)
            previous_precision = set_parallel_precision(True)
            try:
                with torch.cuda.stream(stream):
                    outcome = recipe.run(context, device)
            finally:
                restore_parallel_precision(previous_precision)
            stream.synchronize()
        else:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            outcome = recipe.run(context, device)
        if use_parallel_regions:
            outcome.report["parallel_started_seconds"] = recipe_started - parallel_started
            outcome.report["parallel_finished_seconds"] = monotonic() - parallel_started
        return name, recipe, outcome

    completed: dict[str, tuple[object, object]] = {}
    region_schedule: list[dict[str, float | str]] = []
    parallel_started = monotonic()
    if use_parallel_regions:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        # Stable GEMS geometry has an existing FP32 exception. Keep it fixed
        # for the whole overlap window; per-thread toggles would race across
        # streams and make the numerical policy order-dependent.
        torch.backends.cuda.matmul.allow_tf32 = False
        parallel_batches = _parallel_batches(selected, parallel_workers)
        reports["shared_preprocessing"]["parallel_batches"] = parallel_batches
        try:
            for parallel_order in parallel_batches:
                with ThreadPoolExecutor(max_workers=min(parallel_workers, len(parallel_order)),
                                        thread_name_prefix="fnit-gems") as executor:
                    futures = {executor.submit(run_recipe, name): name for name in parallel_order}
                    for future in as_completed(futures):
                        name, recipe, outcome = future.result()
                        completed[name] = (recipe, outcome)
                        region_schedule.append({
                            "name": name,
                            "started_seconds": float(outcome.report["parallel_started_seconds"]),
                            "finished_seconds": float(outcome.report["parallel_finished_seconds"]),
                            "recipe_seconds": float(outcome.report.get("seconds", 0.0)),
                        })
                        # Release this region's device tensors before the next
                        # batch. This keeps the measured peak bounded by one
                        # light pair or one high-resolution recipe.
                        outcome.fit.highres_labels = outcome.highres_labels
                        fit = outcome.fit
                        fit.labels = fit.labels.detach().cpu()
                        fit.posterior = fit.posterior.detach().cpu()
                        fit.priors = fit.priors.detach().cpu()
                        fit.vertices = fit.vertices.detach().cpu()
                        fit.gaussian_parameters = GaussianParameters(
                            fit.gaussian_parameters.means.detach().cpu(),
                            fit.gaussian_parameters.covariances.detach().cpu())
                torch.cuda.empty_cache()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.cuda.empty_cache()
        batch_peak = torch.cuda.max_memory_allocated(device) / 2**30
        for _, outcome in completed.values():
            outcome.report["peak_gpu_gib"] = batch_peak
            outcome.report["peak_gpu_scope"] = "all_parallel_regions"
        reports["shared_preprocessing"]["region_schedule"] = sorted(
            region_schedule, key=lambda row: row["finished_seconds"])
        reports["shared_preprocessing"]["parallel_precision_policy"] = "fp32_global_overlap"
    else:
        for name in selected:
            _, recipe, outcome = run_recipe(name)
            completed[name] = (recipe, outcome)

    # Merge in canonical order regardless of future completion order. This
    # preserves overlap resolution, label-table ordering and volume semantics.
    for name in selected:
        recipe, outcome = completed[name]
        outcome.report["optimization"] = optimization
        outcome.report["mesh_solver"] = getattr(outcome.fit, "optimization_stats", None)
        outcome.fit.highres_labels = outcome.highres_labels
        labels, confidence, support = _outcome_on_native_grid(outcome, context, native_image)
        _merge_native(combined, best_conf, labels, confidence, support)
        if device.type == "cuda":
            fit = outcome.fit
            fit.labels = fit.labels.detach().cpu()
            fit.posterior = fit.posterior.detach().cpu()
            fit.priors = fit.priors.detach().cpu()
            fit.vertices = fit.vertices.detach().cpu()
            fit.gaussian_parameters = GaussianParameters(
                fit.gaussian_parameters.means.detach().cpu(),
                fit.gaussian_parameters.covariances.detach().cpu())
            if not use_parallel_regions:
                torch.cuda.empty_cache()
        detailed[name] = outcome.fit
        reports[name] = outcome.report
        logger.info("Finished %s: %s", name, outcome.report)
        atlas = GEMSAtlas.from_freesurfer(recipe.directory / "AtlasMesh.gz",
                                          recipe.directory / "compressionLookupTable.txt")
        side = name.rsplit("-", 1)[-1] if name.startswith("hippo-amygdala") else None
        offset = 10000 if side == "right" else 0
        allowed = (set(int(v) for v in atlas.label_ids if 200 <= int(v) <= 246 and int(v) != 201
                       or 7000 <= int(v) < 8000) if side else
                   {int(v) for v in atlas.label_ids if int(v) in outcome.soft_volumes_mm3})
        for label, label_name in zip(atlas.label_ids, atlas.label_names):
            old_id = int(label)
            if old_id not in allowed:
                continue
            identifier = old_id + offset
            label_name = (side.title() + "-" + label_name if side else label_name)
            if identifier in table and table[identifier] != label_name:
                raise ValueError(f"Output label collision for {identifier}: {table[identifier]} vs {label_name}")
            table[identifier] = label_name
            family = ("HippoSF" if side else "ThalamicNuclei" if name == "thalamus" else "BrainstemSS")
            hemisphere = side if side else ("left" if label_name.startswith("Left-") else
                                            "right" if label_name.startswith("Right-") else None)
            parent = ("amygdala" if side and old_id >= 7000 else
                      "hippocampus" if side else name)
            metadata[identifier] = SubregionLabel(identifier, label_name, parent,
                                                   family, hemisphere, name)
            volumes[identifier] = {"soft_volume_mm3": outcome.soft_volumes_mm3.get(identifier, 0.0)}
    for identifier in volumes:
        volumes[identifier]["hard_volume_mm3"] = float(np.count_nonzero(combined == identifier) * voxel_volume)
    out = new_image(combined, native_image, affine=native_image.affine)
    out.set_qform(native_image.affine, code=0)
    out.set_sform(native_image.affine, code=2)
    result = SubregionResult(out, table, detailed, torch.as_tensor(best_conf), reports,
                            volumes, metadata,
                            input_source=str(t1) if isinstance(t1, (str, Path)) else None)
    result.timings["compute_seconds"] = monotonic() - started
    if output_dir is not None:
        result.save(output_dir, save_highres=save_highres, save_posteriors=save_posteriors)
    return result
