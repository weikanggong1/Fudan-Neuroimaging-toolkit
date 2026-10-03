"""Corrected DWI and official FreeSurfer anatomy to four region matrices."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
from nibabel.orientations import (
    apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform,
)
import numpy as np
import torch

from .anatomy import freesurfer_five_tissue, gmwmi_from_five_tissue, resample_labels_nearest
from .bet import bet_mask, mean_bzero, mrtrix_roundtrip_voxel_size
from .assignment import build_connectomes
from .atlas_builder import (
    combine_cortical_tian, glasser_to_t1, native_annotation_to_t1, schaefer_to_t1,
)
from .atlas_tian import fnirt_tian_to_t1, synthmorph_tian_to_t1
from .freesurfer_subject import (
    ConnectomeNode, FreeSurferSubject, fs_aparc_a2009s_atlas, fs_aparc_atlas,
)
from .fod import fit_mrtrix_msmt_csd
from .masks import dwi2mask_legacy, maskfilter_six_connected
from .mtnormalise import normalise_mrtrix_three_tissue
from .response import (
    _mrtrix_interpret_tensor_gradients,
    estimate_mrtrix_dhollander, fit_mrtrix_dhollander_tensor,
    mrtrix_shell_centres,
)
from .sift2 import estimate_sift2_weights
from .tcksample_precise import sample_streamline_mean_precise
from .tracking import Tractogram, probabilistic_tractography
from .checkpoints import (
    CheckpointStore, StreamlinePoints, fingerprint_paths, tensor_fingerprint,
)

# Saved BIDS matrices must be recalculated after numerical fixes even when
# the package version and original image paths remain unchanged.
CONNECTOME_NUMERICAL_REVISION = "accuracy-20261003-v1"

SCHAEFER_TIAN_ATLASES = {
    "schaefer200+tian-s1": (200, 1),
    "schaefer500+tian-s4": (500, 4),
    "schaefer1000+tian-s4": (1000, 4),
}
NATIVE_TIAN_ATLASES = {
    "aparc+tian-s1": "aparc",
    "aparc.a2009s+tian-s1": "aparc.a2009s",
}
GLASSER_TIAN_ATLASES = {
    "glasser+tian-s1": 1,
    "glasser+tian-s4": 4,
}


def _image(path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = nib.load(str(path))
    data = torch.from_numpy(np.asarray(image.get_fdata(dtype=np.float32))).to(device)
    affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
    return data, affine


def _scalar_on_grid(path: str | Path, reference: nib.spatialimages.SpatialImage,
                    device: torch.device, *, binary: bool = False) -> torch.Tensor:
    """Load scalar image on DWI voxel centers, allowing lossless axis flips.

    Input is a NIfTI scalar volume; output is float32 or bool ``[X,Y,Z]`` on
    ``device``. Equivalent original operation: ``mrconvert`` with stride
    changes and no resampling. Allow at most 0.001 mm NIfTI rounding at
    grid corners; reject shifted or resampled voxel grids.
    """
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path} must be a scalar 3D image")
    orientation = ornt_transform(io_orientation(image.affine),
                                 io_orientation(reference.affine))
    aligned_affine = image.affine @ inv_ornt_aff(orientation, data.shape)
    aligned = apply_orientation(data, orientation)
    corners = np.array(np.meshgrid(
        *((0, size - 1) for size in aligned.shape), indexing="ij",
    )).reshape(3, -1)
    homogeneous = np.vstack((corners, np.ones((1, corners.shape[1]))))
    grid_error_mm = np.linalg.norm(
        ((aligned_affine - reference.affine) @ homogeneous)[:3], axis=0,
    ).max()
    if aligned.shape != reference.shape[:3] or grid_error_mm > 1e-3:
        raise ValueError(f"{path} does not share DWI voxel centers")
    aligned = np.ascontiguousarray(aligned)
    if binary:
        if not np.all(np.isfinite(aligned)) or not np.all(
            (aligned == 0) | (aligned == 1)
        ):
            raise ValueError(f"{path} must contain only 0 and 1")
        return torch.as_tensor(aligned.astype(bool), device=device)
    return torch.as_tensor(aligned.astype(np.float32), device=device)


def _gradients(bvals_path: str | Path, bvecs_path: str | Path,
               n_volumes: int, affine: torch.Tensor,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert FSL eddy-rotated gradients to MRtrix RAS gradient frame.

    Input bvals is N values; bvecs is 3×N or N×3. Returns Float64 bvals [N]
    and unit world-frame bvecs [N,3]. NIfTI affine columns are normalised
    before Double polar decomposition, following MRtrix FSL import.
    The native default Auto policy scales b-values by original norm² only
    when max(abs(log(norm²))) > 0.01. Directions and b-values remain Double
    until the downstream solver; no DWI or tensor-image precision changes.
    The existing b<50 zero-vector rule and rejection of zero diffusion
    directions are retained; MRtrix can retain an arbitrary nonzero b0
    vector and warn about ambiguous zero diffusion directions.
    """
    bvals = torch.as_tensor(np.loadtxt(bvals_path).reshape(-1),
                            device=device, dtype=torch.float64)
    array = np.loadtxt(bvecs_path)
    if array.shape == (3, n_volumes):
        array = array.T
    if bvals.shape != (n_volumes,) or array.shape != (n_volumes, 3):
        raise ValueError("bvals/bvecs must match the DWI volume count")
    bvecs = torch.as_tensor(array, device=device, dtype=torch.float64).clone()
    nonzero = bvals >= 50
    bvecs[~nonzero] = 0
    interpreted = _mrtrix_interpret_tensor_gradients(
        torch.cat((bvecs, bvals[:, None]), dim=1),
    )
    bvecs, bvals = interpreted[:, :3], interpreted[:, 3]
    if bool(torch.linalg.det(affine[:3, :3]) > 0):
        bvecs[:, 0] = -bvecs[:, 0]
    linear = affine[:3, :3].double()
    cosine = linear / torch.linalg.vector_norm(linear, dim=0)[None, :]
    u, _, vh = torch.linalg.svd(cosine)
    bvecs = bvecs @ (u @ vh).T
    bvecs[nonzero] = torch.nn.functional.normalize(bvecs[nonzero], dim=-1)
    bvecs[~nonzero] = 0
    if bool((torch.linalg.vector_norm(bvecs[nonzero], dim=-1) < 0.9).any()):
        raise ValueError("non-b0 bvecs must be nonzero")
    return bvals, bvecs


def _registration(b0_brain: torch.Tensor, dwi_affine: torch.Tensor,
                  t1_brain: str | Path, device: torch.device) -> torch.Tensor:
    """Run Surfa-free TorchFLIRT 6-DOF/normmi; return DWI→T1 RAS-mm [4,4]."""
    from ..flirt import TorchFLIRT
    moving = nib.Nifti1Image(b0_brain.detach().cpu().numpy(), dwi_affine.cpu().numpy())
    result = TorchFLIRT(device=str(device), dof=6, cost="normmi")(
        moving, t1_brain,
    )
    return torch.as_tensor(result.moving_to_fixed_world, device=device,
                           dtype=torch.float64)


def _bet_on_dwi_grid(mean_b0: torch.Tensor,
                     reference: nib.spatialimages.SpatialImage,
                     device: torch.device) -> torch.Tensor:
    """Apply the established LAS BET implementation on any BIDS DWI orientation."""
    to_las = ornt_transform(io_orientation(reference.affine),
                            axcodes2ornt(("L", "A", "S")))
    las_b0 = torch.as_tensor(np.ascontiguousarray(apply_orientation(
        mean_b0.detach().cpu().numpy(), to_las)), device=device)
    las_spacing = tuple(reference.header.get_zooms()[int(axis)]
                        for axis in to_las[:, 0])
    las_mask = bet_mask(
        mean_b0=las_b0,
        voxel_size=mrtrix_roundtrip_voxel_size(las_spacing),
        fractional_threshold=0.2,
        vertical_gradient=-0.05,
        robust_center=True,
    )
    from_las = ornt_transform(axcodes2ornt(("L", "A", "S")),
                              io_orientation(reference.affine))
    return torch.as_tensor(np.ascontiguousarray(apply_orientation(
        las_mask.cpu().numpy(), from_las)), device=device)


@dataclass
class ConnectomeResult:
    """First atlas, all selected atlas results, and shared reconstruction data."""

    matrices: dict[str, torch.Tensor]
    region_labels: tuple[int, ...]
    atlas: torch.Tensor
    five_tissue: torch.Tensor
    five_tissue_affine: torch.Tensor
    gmwmi: torch.Tensor
    wm_sh: torch.Tensor
    fa: torch.Tensor
    brain_mask: torch.Tensor
    tractogram: Tractogram
    sift2_weights: torch.Tensor
    dwi_affine: torch.Tensor
    atlas_affine: torch.Tensor
    dwi_to_t1_world: torch.Tensor
    nodes: tuple[ConnectomeNode, ...] | None = None
    atlas_results: dict[str, "AtlasResult"] | None = None
    cache_status: dict | None = None
    preparation_stages: dict | None = None
    pair_results: dict | None = None


@dataclass
class AtlasResult:
    """One selected atlas, its row labels, and four connectivity matrices."""

    matrices: dict[str, torch.Tensor]
    region_labels: tuple[int, ...]
    atlas: torch.Tensor
    atlas_affine: torch.Tensor
    nodes: tuple[ConnectomeNode, ...] | None


def _checkpoint_policy(device: torch.device) -> dict:
    """Record the actual compute policy without changing process settings."""
    policy = dict(device=str(device), torch=str(torch.__version__),
                  numpy=str(np.__version__), nibabel=str(nib.__version__),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  matmul_precision=torch.get_float32_matmul_precision(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cpu_threads=torch.get_num_threads(),
                  cpu_interop_threads=torch.get_num_interop_threads(),
                  autocast_cuda=torch.is_autocast_enabled("cuda"),
                  autocast_cpu=torch.is_autocast_enabled("cpu"),
                  autocast_cuda_dtype=str(torch.get_autocast_dtype("cuda")),
                  autocast_cpu_dtype=str(torch.get_autocast_dtype("cpu")),
                  cuda_runtime=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version())
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        policy.update(gpu_name=properties.name,
                      gpu_capability=[properties.major, properties.minor],
                      gpu_uuid=str(getattr(properties, "uuid", "unavailable")))
    return policy


def _core_source_fingerprint(*, registration: bool) -> dict:
    """Hash core workers, packaged numerical assets and its orchestration.

    The atlas portion of this file is deliberately excluded. Its own keys
    are managed by the template stage, so edits there do not rerun tracking.
    """
    package = Path(__file__).parent
    names = ("anatomy", "bet", "response", "fod", "masks", "mtnormalise",
             "tracking", "sift2", "sift2_fixels", "sift2_mapping",
             "sift2_optimizer", "sift2_proc_mask", "tcksample_precise")
    paths = {name: package / (name + ".py") for name in names}
    paths.update(act_lut=package / "FreeSurfer2ACT_sgm_amyg_hipp_ids.tsv",
                 sift2_sphere=package / "data/mrtrix_sift2_1281.npz",
                 mask_components=package.parent / "synthseg_parc/postprocess.py")
    if registration:
        # Imported __pycache__ and other runtime files are not numerical source.
        paths.update({"flirt_" + str(path.relative_to(package.parent / "flirt")):
                      path for path in sorted((package.parent / "flirt").rglob("*.py"))})
        paths.update(shared_sampling=package.parent / "_sampling_plan.py",
                     shared_transforms=package.parent / "_transforms.py",
                     shared_nibabel=package.parent / "_nib.py",
                     shared_world_resampling=package.parent / "_world_resampling.py")
    # AST source segments are stable when unrelated atlas orchestration changes.
    text = Path(__file__).read_text()
    functions = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    helper_names = ("_image", "_scalar_on_grid", "_gradients", "_bet_on_dwi_grid",
                    "_compute_shared_core") + (("_registration",) if registration else ())
    helpers = {name: hashlib.sha256(functions[name].encode()).hexdigest()
               for name in helper_names}
    return dict(files=fingerprint_paths(paths), orchestration=helpers)


_CORE_ARRAYS = ("seg", "seg_affine", "five", "gmwmi", "transform", "five_affine",
                "wm_sh", "fa", "mask", "weights", "dwi_affine")


def _pack_core(core: dict) -> tuple[dict, dict]:
    tracks = core["tracks"]
    counts = np.asarray([len(path) for path in tracks.paths], dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    arrays = {name: core[name] for name in _CORE_ARRAYS}
    arrays.update(track_points=StreamlinePoints(tracks.paths), track_offsets=offsets,
                  track_endpoints=tracks.endpoints, track_lengths=tracks.lengths_mm,
                  track_mean_fa=tracks.mean_fa, track_seeds=tracks.accepted_seeds)
    metadata = dict(dwi_shape=list(core["dwi_shape"]), track_count=len(tracks.paths),
                    seeds_attempted=tracks.seeds_attempted, format_revision=1)
    return arrays, metadata


def _restore_core(arrays: dict, metadata: dict, device: torch.device) -> dict:
    """Validate stage structure before any transfer; retain all numeric values."""
    expected = {*_CORE_ARRAYS, "track_points", "track_offsets", "track_endpoints",
                "track_lengths", "track_mean_fa", "track_seeds"}
    if set(arrays) != expected or metadata["format_revision"] != 1:
        raise ValueError("incomplete shared-core checkpoint")
    shape = tuple(metadata["dwi_shape"])
    n = metadata["track_count"]
    if (len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
            or type(n) is not int or n <= 0 or type(metadata["seeds_attempted"]) is not int
            or metadata["seeds_attempted"] < n):
        raise ValueError("invalid shared-core metadata")
    offsets, points = arrays["track_offsets"], arrays["track_points"]
    if (offsets.dtype != torch.int64 or offsets.shape != (n + 1,)
            or points.ndim != 2 or points.shape[1] != 3
            or offsets[0] != 0 or offsets[-1] != len(points)
            or not bool(((offsets[1:] - offsets[:-1]) >= 2).all())):
        raise ValueError("invalid packed streamline offsets/points")
    for name, expected_shape in (("track_endpoints", (n, 2, 3)),
                                 ("track_seeds", (n, 3)), ("track_lengths", (n,)),
                                 ("track_mean_fa", (n,)), ("weights", (n,)),
                                 ("fa", shape), ("mask", shape), ("wm_sh", (*shape, 45))):
        if tuple(arrays[name].shape) != expected_shape:
            raise ValueError(f"invalid shared-core shape: {name}")
    seg_shape = tuple(arrays["seg"].shape)
    if (len(seg_shape) != 3 or tuple(arrays["five"].shape) != (*seg_shape, 5)
            or tuple(arrays["gmwmi"].shape) != seg_shape):
        raise ValueError("invalid shared-core anatomy grid")
    for name in ("seg_affine", "transform", "five_affine", "dwi_affine"):
        if arrays[name].shape != (4, 4) or arrays[name].dtype != torch.float64:
            raise ValueError(f"invalid shared-core affine: {name}")
    if arrays["weights"].dtype != torch.float64 or arrays["mask"].dtype != torch.bool:
        raise ValueError("invalid shared-core weight/mask dtype")
    for name in ("seg", "five", "gmwmi", "wm_sh", "fa", "track_points",
                 "track_endpoints", "track_lengths", "track_mean_fa", "track_seeds"):
        if arrays[name].dtype != torch.float32:
            raise ValueError(f"invalid shared-core field dtype: {name}")
    # No isfinite gate on FA/statistics: preserve legitimate NaN/Inf diagnostics.
    boundary = offsets.tolist()
    core = {name: arrays[name].to(device) for name in _CORE_ARRAYS}
    points = points.to(device)
    core["tracks"] = Tractogram(
        tuple(points[begin:end] for begin, end in zip(boundary[:-1], boundary[1:])),
        arrays["track_endpoints"].to(device), arrays["track_lengths"].to(device),
        arrays["track_mean_fa"].to(device), metadata["seeds_attempted"],
        arrays["track_seeds"].to(device),
    )
    core["dwi_shape"] = shape
    return core


def _shared_core_checkpoint(options: dict, *, checkpoint_dir, overwrite: bool):
    if checkpoint_dir is None:
        return _compute_shared_core(**options), {"status": "completed", "enabled": False}, None
    paths = {name: options[name] for name in (
        "dwi", "bvals", "bvecs", "t1_brain", "t1_segmentation", "brain_mask",
        "response_mask", "fod_mask", "normalise_mask", "fa_map")}
    inputs = fingerprint_paths(paths)
    parameters = dict(n_seeds=options["n_seeds"], seed=options["seed"],
                      compile_arc=options["compile_arc"],
                      shell_bvals=tensor_fingerprint(options["shell_bvals"]),
                      dwi_to_t1_world=tensor_fingerprint(options["dwi_to_t1_world"]))
    device = options["device"]
    policy = _checkpoint_policy(device)
    source = _core_source_fingerprint(registration=options["dwi_to_t1_world"] is None)
    store = CheckpointStore(Path(checkpoint_dir) / "shared", numerical_revision=CONNECTOME_NUMERICAL_REVISION,
                            device_policy=policy, source_fingerprint=source, overwrite=overwrite)
    key = store.make_key("core", inputs=inputs, parameters=parameters)
    restored = store.load("core", key, device="cpu")
    if restored is not None:
        try:
            core = _restore_core(*restored, device)
        except (ValueError, KeyError, TypeError) as exc:
            store.events.append(dict(stage="core", key=key, status="miss",
                                     reason=f"invalid_core_schema: {exc}"))
        else:
            # Inputs must still match after restoration as well as before it.
            if (fingerprint_paths(paths) != inputs or _checkpoint_policy(device) != policy
                    or _core_source_fingerprint(registration=options["dwi_to_t1_world"] is None) != source):
                raise RuntimeError("shared-core input/source/device policy changed during restoration")
            return core, dict(status="skipped", enabled=True, key=key), store
    core = _compute_shared_core(**options)
    def validate_publication():
        if (fingerprint_paths(paths) != inputs or _checkpoint_policy(device) != policy
                or _core_source_fingerprint(registration=options["dwi_to_t1_world"] is None) != source):
            raise RuntimeError("shared-core input/source/device policy changed during computation; not publishing")
    validate_publication()
    arrays, metadata = _pack_core(core)
    store.publish("core", key, arrays=arrays, metadata=metadata,
                  before_publish=validate_publication)
    return core, dict(status="completed", enabled=True, key=key), store


def _compute_shared_core(*, dwi, bvals, bvecs, t1_brain, t1_segmentation,
                         brain_mask, shell_bvals, response_mask, fod_mask,
                         normalise_mask, fa_map, dwi_to_t1_world, n_seeds,
                         seed, compile_arc, device):
    """Run the unchanged numerical chain through precise per-track FA."""
    reference = nib.load(str(dwi))
    dwi_data, dwi_affine = _image(dwi, device)
    if dwi_data.ndim != 4:
        raise ValueError("DWI must have shape [X,Y,Z,N]")
    bval, bvec = _gradients(bvals, bvecs, dwi_data.shape[-1],
                            dwi_affine, device)
    gradient = torch.cat((bvec.double(), bval.double()[:, None]), dim=1)
    if shell_bvals is None:
        _, _, shells, _ = mrtrix_shell_centres(gradient)
    else:
        shells = torch.as_tensor(shell_bvals, device=device, dtype=torch.float64)
    if shells.ndim != 1 or len(shells) < 2 or not bool((shells[1:] > shells[:-1]).all()):
        raise ValueError("shell_bvals must be ordered MRtrix shell centers")
    mean_b0 = mean_bzero(dwi=dwi_data, bvalues=bval)
    if brain_mask is None:
        mask = _bet_on_dwi_grid(mean_b0, reference, device)
    else:
        mask = _scalar_on_grid(brain_mask, reference, device, binary=True)
    response_selection = (dwi2mask_legacy(dwi_data, gradient[:, 3], shells) if response_mask is None else
                          _scalar_on_grid(response_mask, reference, device, binary=True))
    fod_selection = (maskfilter_six_connected(mask, "dilate") if fod_mask is None else
                     _scalar_on_grid(fod_mask, reference, device, binary=True))
    norm_selection = (maskfilter_six_connected(mask, "erode") if normalise_mask is None else
                      _scalar_on_grid(normalise_mask, reference, device, binary=True))
    if fa_map is None:
        fa, _ = fit_mrtrix_dhollander_tensor(dwi_data, gradient, mask)
    else:
        fa = _scalar_on_grid(fa_map, reference, device)
    seg, seg_affine = _image(t1_segmentation, device)
    if seg.ndim != 3:
        raise ValueError("FreeSurfer aparc+aseg must be a 3D label image")
    five = freesurfer_five_tissue(seg)
    gmwmi = gmwmi_from_five_tissue(five)
    if dwi_to_t1_world is None:
        b0_brain = mean_b0 * mask
        transform = _registration(b0_brain, dwi_affine, t1_brain, device)
    else:
        transform = torch.as_tensor(dwi_to_t1_world, device=device,
                                     dtype=torch.float64)
        if transform.shape != (4, 4):
            raise ValueError("dwi_to_t1_world must be 4x4")
    five_affine = torch.linalg.inv(transform) @ seg_affine
    shells, wm_response, gm_response, csf_response, _ = estimate_mrtrix_dhollander(
        dwi_data, gradient, shells, response_selection,
    )
    wm_raw, gm_raw, csf_raw = fit_mrtrix_msmt_csd(
        dwi_data, gradient, shells, wm_response, gm_response, csf_response,
        fod_selection,
    )
    wm_sh = normalise_mrtrix_three_tissue(
        wm_raw, gm_raw, csf_raw, norm_selection, dwi_affine,
    ).wm
    tracks = probabilistic_tractography(
        wm_sh, dwi_affine, five, five_affine, gmwmi,
        n_seeds=n_seeds, seed=seed,
        compile_arc=compile_arc,
        five_tissue_spacing_mm=nib.load(str(t1_segmentation)).header.get_zooms()[:3],
    )
    if not tracks.paths:
        raise RuntimeError("ACT tracking accepted no streamlines")
    step_size_mm = float(torch.linalg.vector_norm(dwi_affine[:3, :3], dim=0).prod().pow(1 / 3)) / 2
    weights = estimate_sift2_weights(
        tracks.paths, wm_sh, dwi_affine, five, five_affine,
        step_size_mm=step_size_mm,
    )
    tracks.mean_fa = sample_streamline_mean_precise(tracks.paths, fa, dwi_affine)
    return dict(seg=seg, seg_affine=seg_affine, five=five, gmwmi=gmwmi,
                transform=transform, five_affine=five_affine, wm_sh=wm_sh,
                fa=fa, mask=mask, tracks=tracks, weights=weights,
                dwi_affine=dwi_affine, dwi_shape=tuple(dwi_data.shape[:3]))


class UKBConnectome_pipeline:
    """Compute UKB-style connectomes from corrected DWI and FreeSurfer T1.

    The paired T1 segmentation uses completed FreeSurfer-format ``recon-all``
    aparc+aseg. ``run_bids`` selects or reuses its reconstruction backend. PyTorch
    computes response, FOD, mtnormalise, 5TT/GMWMI, ACT tracking, SIFT2,
    precise per-track FA and the four matrices. A completed FreeSurfer subject
    directory supplies T1 and one or more atlas choices; explicit T1/atlas inputs
    remain available for fixed-input comparisons. The caller may provide a
    fixed BET brain mask; otherwise the LAS DWI
    mean b0 is skull stripped with native PyTorch BET. CUDA uses TF32
    by default and no float16/bfloat16 conversion.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def run_bids(
        self,
        bids_root: str | Path,
        output_dir: str | Path,
        *,
        subject: str,
        n_seeds: int,
        atlas: str | Sequence[str] = "fs-aparc",
        session: str | None = None,
        run: str | None = None,
        acquisition: str | None = None,
        direction: str | None = None,
        t1: str | Path | None = None,
        freesurfer_subject_dir: str | Path | None = None,
        corrected_dwi: str | Path | None = None,
        rotated_bvecs: str | Path | None = None,
        eddy_gp_seed: int | None = None,
        recon_backend: str = "auto",
        recon_options: Mapping | None = None,
        checkpoint_dir: str | Path | None = None,
        overwrite: bool = False,
        **connectome_options,
    ) -> ConnectomeResult:
        """Run raw BIDS preparation then compute the selected atlas matrices.

        ``output_dir`` stores resumable DWI and recon-all intermediates;
        ``connectome_options`` are the optional arguments of ``__call__``.
        The Python result remains in memory; use the CLI to write matrices.
        ``eddy_gp_seed`` fixes EDDY's GP voxel sampling independently of the
        tractography ``seed``. ``None`` retains time-based EDDY sampling.
        ``recon_backend``/``recon_options`` are passed to BIDS preparation.
        Shared reconstruction checkpoints default to ``output_dir/checkpoints``;
        template changes reuse that core. ``overwrite=True`` recomputes stages
        into new generations and preserves earlier checkpoint generations.
        """
        from .bids import prepare_bids_connectome
        from .template_inputs import preflight_template_pairs, validate_readonly_subject_outputs
        from .input_protection import template_readonly_paths
        if connectome_options.get("template_pairs") is not None:
            connectome_options["template_pairs"] = preflight_template_pairs(
                connectome_options["template_pairs"], subject_dir=freesurfer_subject_dir)
        validate_readonly_subject_outputs(
            freesurfer_subject_dir, output_dir=output_dir,
            checkpoint_dir=checkpoint_dir or Path(output_dir) / "checkpoints")
        readonly = list(template_readonly_paths(connectome_options.get("template_pairs"),
                                                freesurfer_subject_dir))
        readonly.extend(value for name in ("brain_mask", "response_mask", "fod_mask", "normalise_mask",
                                            "fa_map", "mni_template", "mni_to_t1_transform", "synthmorph_weights")
                        if isinstance(value := connectome_options.get(name), (str, Path)))

        selected = prepare_bids_connectome(
            bids_root, output_dir, subject=subject, session=session, run=run,
            acquisition=acquisition, direction=direction, t1=t1,
            freesurfer_subject_dir=freesurfer_subject_dir,
            corrected_dwi=corrected_dwi, rotated_bvecs=rotated_bvecs,
            eddy_gp_seed=eddy_gp_seed,
            recon_backend=recon_backend, recon_options=recon_options,
            device=str(self.device), overwrite=overwrite,
            readonly_inputs=tuple(readonly),
        )
        result = self(
            selected.dwi, selected.bvals, selected.bvecs,
            freesurfer_subject_dir=selected.freesurfer_subject_dir,
            atlas=atlas, n_seeds=n_seeds,
            checkpoint_dir=(Path(output_dir) / "checkpoints" if checkpoint_dir is None else checkpoint_dir),
            overwrite=overwrite, **connectome_options,
        )
        if hasattr(selected, "stages"):
            result.preparation_stages = selected.stages
        return result

    @torch.inference_mode()
    def __call__(
        self,
        dwi: str | Path,
        bvals: str | Path,
        bvecs: str | Path,
        t1_brain: str | Path | None = None,
        *,
        t1_segmentation: str | Path | None = None,
        atlas_dwi: str | Path | None = None,
        freesurfer_subject_dir: str | Path | None = None,
        atlas: str | Sequence[str] = "fs-aparc",
        atlas_templates_dir: str | Path | None = None,
        fsaverage_dir: str | Path | None = None,
        mni_template: str | Path | None = None,
        synthmorph_weights: str | Path | None = None,
        tian_fnirt_coeff: str | Path | None = None,
        brain_mask: str | Path | None = None,
        n_seeds: int,
        shell_bvals: Sequence[float] | None = None,
        response_mask: str | Path | None = None,
        fod_mask: str | Path | None = None,
        normalise_mask: str | Path | None = None,
        fa_map: str | Path | None = None,
        dwi_to_t1_world: np.ndarray | torch.Tensor | None = None,
        seed: int = 0,
        compile_arc: bool = False,
        template_pairs: Sequence | None = None,
        assignment_radius: float = 4.0,
        mni_to_t1_transform=None,
        checkpoint_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> ConnectomeResult:
        """Run one subject; all images/gradients must describe the same scan.

        ``dwi`` is corrected float32 NIfTI [X,Y,Z,N]; ``bvals``/``bvecs``
        are FSL N and 3×N or N×3 eddy-rotated files.
        ``freesurfer_subject_dir`` is a completed recon-all subject directory;
        ``atlas="fs-aparc"`` builds a contiguous 84-node DWI-grid atlas
        from its ``mri/aparc+aseg.mgz``. ``aparc+tian-s1`` and
        ``aparc.a2009s+tian-s1`` instead use native cortical annotations
        plus Tian S1, yielding 84 and 164 different node definitions.
        A Schaefer+Tian atlas
        reads the original Schaefer and Tian template files from
        ``atlas_templates_dir``, maps Schaefer through ``fsaverage_dir``
        to the native ribbon, registers ``mni_template`` to T1 with FNIT
        SynthMorph joint using optional ``synthmorph_weights``, then builds
        a contiguous DWI atlas. Glasser+Tian additionally reads the original
        32k dlabel and sibling surface templates, uses Workbench for the
        fsLR-to-fsaverage label resample, then maps to the native ribbon with
        PyTorch; its labels follow right 1..180, left 181..360. For a fixed original UKB atlas, supply
        ``tian_fnirt_coeff`` instead of ``mni_template``; FNIT inverts the
        given FNIRT T1→MNI coefficient and samples Tian without calling FSL.
        Alternatively, ``t1_brain``,
        ``t1_segmentation`` and ``atlas_dwi`` must all be supplied. The
        explicit atlas is a nonnegative integer image in DWI RAS world space
        and may retain a separate voxel grid.
        ``brain_mask`` fixes a binary DWI BET mask; if omitted, native BET
        generates it from the MRtrix-style double-accumulated mean b0 on an
        LAS grid. MRtrix shell clustering runs from b-values by default; optional ``shell_bvals`` fixes the
        response-header shell labels for a same-input comparison. Optional response/FOD/normalise
        masks fix the exact reference masks; the response mask defaults to
        pinned MRtrix ``dwi2mask legacy`` from the DWI itself. FOD and normalise masks default
        to two six-neighbour dilation/erosion passes of ``brain_mask``.
        Optional ``fa_map`` is the precomputed UKB dti_FA on the DWI grid;
        without it, MRtrix-style DWI tensor fitting produces FA. Optional
        DWI→T1 RAS-mm transform freezes registration; otherwise TorchFLIRT
        runs 6-DOF/normmi. ``n_seeds`` counts tracking attempts and ``seed``
        seeds the PyTorch generator, whose sequence differs from MRtrix.
        ``compile_arc`` compiles the CUDA iFOD2 probability kernel on first
        use, with startup cost but lower steady propagation time.
        ``checkpoint_dir=None`` disables core checkpoints for this direct call;
        a directory enables SHA-verified, atomic stage reuse. ``overwrite=True``
        bypasses valid checkpoints while preserving their old generations.
        Template choices, MNI template mapping and ``assignment_radius`` (4 mm
        by default) do not enter the shared-core key. ``template_pairs`` adds
        paired matrices through the separate template builder; a supplied
        ``mni_to_t1_transform`` is consumed there, not by DWI registration.

        Output ``ConnectomeResult.matrices`` describes the first selected atlas;
        ``atlas_results[name]`` holds each atlas without rerunning tracking.
        The four matrices contain count int64 and SIFT2 FBC,
        mean length (mm), mean FA float32 K×K arrays. ``region_labels`` maps
        rows/columns to 1..K. ``nodes`` defines each row when the FreeSurfer
        subject input is used; ``atlas`` is then on the DWI voxel grid.
        ``five_tissue`` [A,B,C,5] and ``gmwmi`` [A,B,C] retain T1 grid but
        their affine maps into DWI RAS space. ``wm_sh`` [X,Y,Z,45] is the
        normalized FOD; ``fa`` [X,Y,Z] is the map sampled along tracks;
        ``brain_mask`` [X,Y,Z] is the DWI-grid bool BET mask.
        ``tractogram`` holds world-mm paths/endpoints/lengths/means and
        ``sift2_weights`` is float64 [T] in track order.
        """
        atlas_names = (atlas,) if isinstance(atlas, str) else tuple(atlas)
        if not atlas_names or len(set(atlas_names)) != len(atlas_names):
            raise ValueError("atlas must contain one or more distinct names")
        if n_seeds < 1:
            raise ValueError("n_seeds must be positive")
        if not math.isfinite(assignment_radius) or assignment_radius <= 0:
            raise ValueError("assignment_radius must be finite and positive")
        if freesurfer_subject_dir is not None:
            if any(value is not None for value in (t1_brain, t1_segmentation, atlas_dwi)):
                raise ValueError("freesurfer_subject_dir cannot be combined with explicit T1/atlas inputs")
            if any(name not in ("fs-aparc", "fs-aparc-a2009s", *SCHAEFER_TIAN_ATLASES,
                                *NATIVE_TIAN_ATLASES, *GLASSER_TIAN_ATLASES)
                   for name in atlas_names):
                raise ValueError("unsupported atlas from a subject directory")
            if any(name not in ("fs-aparc", "fs-aparc-a2009s")
                   for name in atlas_names) and (
                atlas_templates_dir is None or
                (any(name in (*SCHAEFER_TIAN_ATLASES, *GLASSER_TIAN_ATLASES)
                     for name in atlas_names)
                 and fsaverage_dir is None) or
                (mni_template is None) == (tian_fnirt_coeff is None)
            ):
                raise ValueError("cortical+Tian atlas requires atlas_templates_dir, surface-atlas fsaverage_dir and exactly one of mni_template or tian_fnirt_coeff")
            if tian_fnirt_coeff is not None and synthmorph_weights is not None:
                raise ValueError("synthmorph_weights cannot be used with tian_fnirt_coeff")
            subject = FreeSurferSubject(Path(freesurfer_subject_dir))
            t1_brain, t1_segmentation = subject.brain, subject.aparc_aseg
        elif any(value is None for value in (t1_brain, t1_segmentation, atlas_dwi)):
            raise ValueError("provide freesurfer_subject_dir or all of t1_brain, t1_segmentation, atlas_dwi")
        elif atlas_names != ("fs-aparc",):
            raise ValueError("a named atlas requires freesurfer_subject_dir")
        from .template_inputs import preflight_template_pairs, validate_readonly_subject_outputs
        if template_pairs is not None:
            template_pairs = preflight_template_pairs(template_pairs, subject_dir=freesurfer_subject_dir)
        validate_readonly_subject_outputs(freesurfer_subject_dir, checkpoint_dir=checkpoint_dir)
        core_options = dict(
            dwi=dwi, bvals=bvals, bvecs=bvecs, t1_brain=t1_brain,
            t1_segmentation=t1_segmentation, brain_mask=brain_mask,
            shell_bvals=shell_bvals, response_mask=response_mask,
            fod_mask=fod_mask, normalise_mask=normalise_mask, fa_map=fa_map,
            dwi_to_t1_world=dwi_to_t1_world, n_seeds=n_seeds,
            seed=seed, compile_arc=compile_arc, device=self.device,
        )
        core, core_status, store = _shared_core_checkpoint(
            core_options, checkpoint_dir=checkpoint_dir, overwrite=overwrite,
        )
        seg, seg_affine = core["seg"], core["seg_affine"]
        five, gmwmi = core["five"], core["gmwmi"]
        transform, five_affine = core["transform"], core["five_affine"]
        wm_sh, fa, mask = core["wm_sh"], core["fa"], core["mask"]
        tracks, weights = core["tracks"], core["weights"]
        dwi_affine, dwi_shape = core["dwi_affine"], core["dwi_shape"]
        atlas_results = {}
        tian_transform = None
        # These images depend on this subject and template choice, not on the
        # final cortical+Tian pairing. Keep reuse local to this invocation.
        cortical_cache = {}
        tian_cache = {}
        for atlas_name in atlas_names:
            nodes = None
            if freesurfer_subject_dir is not None:
                atlas_source_affine = seg_affine
                if atlas_name == "fs-aparc":
                    atlas_t1, nodes = fs_aparc_atlas(seg)
                elif atlas_name == "fs-aparc-a2009s":
                    atlas_t1, atlas_source_affine = _image(
                        subject.aparc_a2009s_aseg, self.device)
                    atlas_t1, nodes = fs_aparc_a2009s_atlas(
                        atlas_t1, subject.subject_dir)
                else:
                    templates = Path(atlas_templates_dir)
                    if atlas_name in NATIVE_TIAN_ATLASES:
                        tian_scale = 1
                        cortical_key = ("native", NATIVE_TIAN_ATLASES[atlas_name])
                    elif atlas_name in SCHAEFER_TIAN_ATLASES:
                        parcels, tian_scale = SCHAEFER_TIAN_ATLASES[atlas_name]
                        cortical_key = ("schaefer", parcels)
                    else:
                        tian_scale = GLASSER_TIAN_ATLASES[atlas_name]
                        cortical_key = ("glasser",)
                    if cortical_key not in cortical_cache:
                        if cortical_key[0] == "native":
                            cortical_cache[cortical_key] = native_annotation_to_t1(
                                subject_dir=subject.subject_dir,
                                annotation=NATIVE_TIAN_ATLASES[atlas_name],
                                device=str(self.device),
                            )
                        elif cortical_key[0] == "schaefer":
                            cortical_cache[cortical_key] = schaefer_to_t1(
                                subject_dir=subject.subject_dir,
                                fsaverage_dir=fsaverage_dir,
                                left_annot=templates / f"lh.Schaefer2018_{parcels}Parcels_7Networks_order.annot",
                                right_annot=templates / f"rh.Schaefer2018_{parcels}Parcels_7Networks_order.annot",
                                device=str(self.device),
                            )
                        else:
                            cortical_cache[cortical_key] = glasser_to_t1(
                                subject_dir=subject.subject_dir,
                                fsaverage_dir=fsaverage_dir,
                                atlas_templates_dir=templates,
                                device=str(self.device),
                            )
                    cortical, cortical_nodes = cortical_cache[cortical_key]
                    tian_name = f"Tian_Subcortex_S{tian_scale}_3T"
                    if tian_scale not in tian_cache:
                        if tian_fnirt_coeff is None:
                            tian_cache[tian_scale], tian_transform = synthmorph_tian_to_t1(
                                t1_brain=subject.brain,
                                mni_template=mni_template,
                                tian_mni=templates / f"{tian_name}.nii.gz",
                                device=str(self.device), weights=synthmorph_weights,
                                transform=tian_transform,
                            )
                        else:
                            tian_cache[tian_scale] = fnirt_tian_to_t1(
                                t1_brain=subject.brain,
                                tian_mni=templates / f"{tian_name}.nii.gz",
                                forward_coefficients=tian_fnirt_coeff,
                                device=str(self.device),
                            )
                    tian = tian_cache[tian_scale]
                    combined, nodes = combine_cortical_tian(
                        cortical_t1=cortical, cortical_nodes=cortical_nodes,
                        tian_t1=tian,
                        tian_names=tuple((templates / f"{tian_name}_label.txt")
                                         .read_text().splitlines()),
                    )
                    atlas_t1 = torch.as_tensor(np.asarray(combined.dataobj),
                                               device=self.device)
                    atlas_source_affine = torch.as_tensor(
                        combined.affine, device=self.device, dtype=torch.float64,
                    )
                atlas_data = resample_labels_nearest(
                    labels=atlas_t1, source_affine=atlas_source_affine,
                    target_shape=dwi_shape, target_affine=dwi_affine,
                    target_to_source_world=transform,
                )
                atlas_affine = dwi_affine
            else:
                atlas_data, atlas_affine = _image(atlas_dwi, self.device)
            if atlas_data.ndim != 3 or not bool(torch.isfinite(atlas_data).all()) or not bool(
                torch.equal(atlas_data, atlas_data.round())
            ) or bool((atlas_data < 0).any()):
                raise ValueError("atlas_dwi must contain finite nonnegative integer labels")
            atlas_data = atlas_data.to(torch.int32)
            if not bool((atlas_data > 0).any()):
                raise ValueError("atlas contains no region labels")
            region_labels = (tuple(node.index for node in nodes) if nodes is not None else
                             tuple(range(1, int(atlas_data.max()) + 1)))
            matrices = build_connectomes(
                tracks.endpoints, atlas_data, atlas_affine, weights=weights,
                lengths=tracks.lengths_mm, fa=tracks.mean_fa,
                node_count=len(nodes) if nodes is not None else None,
                radius=assignment_radius,
            )
            atlas_results[atlas_name] = AtlasResult(
                matrices, region_labels, atlas_data, atlas_affine, nodes,
            )
        first = atlas_results[atlas_names[0]]
        result = ConnectomeResult(
            first.matrices, first.region_labels, first.atlas, five, five_affine, gmwmi,
            wm_sh, fa, mask, tracks, weights, dwi_affine, first.atlas_affine,
            transform, first.nodes, atlas_results,
        )
        result.cache_status = dict(core=core_status["status"],
                                   core_key=core_status.get("key"),
                                   enabled=core_status["enabled"],
                                   atlas={name: "completed" for name in atlas_names},
                                   events=[] if store is None else store.events)
        if template_pairs is not None:
            from .paired_pipeline import build_template_pairs
            result.pair_results = build_template_pairs(
                result, template_pairs, subject_dir=freesurfer_subject_dir,
                t1_reference_path=t1_brain, mni_to_t1_transform=mni_to_t1_transform,
                mni_template=mni_template, synthmorph_weights=synthmorph_weights,
                checkpoint_dir=checkpoint_dir, overwrite=overwrite,
                assignment_radius=assignment_radius, device=str(self.device),
            )
        return result


UKBConnectome = UKBConnectome_pipeline
