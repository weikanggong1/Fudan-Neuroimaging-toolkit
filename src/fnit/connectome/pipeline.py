"""Corrected DWI and official FreeSurfer anatomy to four region matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    estimate_mrtrix_dhollander, fit_mrtrix_dhollander_tensor,
    mrtrix_shell_centres,
)
from .sift2 import estimate_sift2_weights
from .tcksample_precise import sample_streamline_mean_precise
from .tracking import Tractogram, probabilistic_tractography

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

    Input bvals is N values; bvecs is 3×N or N×3. Returns float32 bvals [N]
    and unit world-frame bvecs [N,3]. Matches ``mrconvert -fslgrad`` for
    non-b0 directions; MRtrix may retain a nonzero arbitrary b0 vector.
    """
    bvals = torch.as_tensor(np.loadtxt(bvals_path).reshape(-1),
                            device=device, dtype=torch.float32)
    array = np.loadtxt(bvecs_path)
    if array.shape == (3, n_volumes):
        array = array.T
    if bvals.shape != (n_volumes,) or array.shape != (n_volumes, 3):
        raise ValueError("bvals/bvecs must match the DWI volume count")
    bvecs = torch.as_tensor(array, device=device, dtype=torch.float32).clone()
    if bool(torch.linalg.det(affine[:3, :3]) > 0):
        bvecs[:, 0] = -bvecs[:, 0]
    u, _, vh = torch.linalg.svd(affine[:3, :3].float())
    bvecs = bvecs @ (u @ vh).T
    nonzero = bvals >= 50
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


@dataclass
class AtlasResult:
    """One selected atlas, its row labels, and four connectivity matrices."""

    matrices: dict[str, torch.Tensor]
    region_labels: tuple[int, ...]
    atlas: torch.Tensor
    atlas_affine: torch.Tensor
    nodes: tuple[ConnectomeNode, ...] | None


class UKBConnectome_pipeline:
    """Compute UKB-style connectomes from corrected DWI and FreeSurfer T1.

    The paired T1 segmentation must be official ``recon-all`` aparc+aseg;
    FreeSurfer itself remains the user's established external stage. PyTorch
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
        overwrite: bool = False,
        **connectome_options,
    ) -> ConnectomeResult:
        """Run raw BIDS preparation then compute the selected atlas matrices.

        ``output_dir`` stores resumable DWI and recon-all intermediates;
        ``connectome_options`` are the optional arguments of ``__call__``.
        The Python result remains in memory; use the CLI to write matrices.
        """
        from .bids import prepare_bids_connectome

        selected = prepare_bids_connectome(
            bids_root, output_dir, subject=subject, session=session, run=run,
            acquisition=acquisition, direction=direction, t1=t1,
            freesurfer_subject_dir=freesurfer_subject_dir,
            corrected_dwi=corrected_dwi, rotated_bvecs=rotated_bvecs,
            device=str(self.device), overwrite=overwrite,
        )
        return self(
            selected.dwi, selected.bvals, selected.bvecs,
            freesurfer_subject_dir=selected.freesurfer_subject_dir,
            atlas=atlas, n_seeds=n_seeds, **connectome_options,
        )

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
        reference = nib.load(str(dwi))
        dwi_data, dwi_affine = _image(dwi, self.device)
        if dwi_data.ndim != 4:
            raise ValueError("DWI must have shape [X,Y,Z,N]")
        bval, bvec = _gradients(bvals, bvecs, dwi_data.shape[-1],
                                dwi_affine, self.device)
        gradient = torch.cat((bvec.double(), bval.double()[:, None]), dim=1)
        if shell_bvals is None:
            _, _, shells, _ = mrtrix_shell_centres(gradient)
        else:
            shells = torch.as_tensor(shell_bvals, device=self.device, dtype=torch.float64)
        if shells.ndim != 1 or len(shells) < 2 or not bool((shells[1:] > shells[:-1]).all()):
            raise ValueError("shell_bvals must be ordered MRtrix shell centers")
        mean_b0 = mean_bzero(dwi=dwi_data, bvalues=bval)
        if brain_mask is None:
            mask = _bet_on_dwi_grid(mean_b0, reference, self.device)
        else:
            mask = _scalar_on_grid(brain_mask, reference, self.device, binary=True)
        response_selection = (dwi2mask_legacy(dwi_data, gradient[:, 3], shells) if response_mask is None else
                              _scalar_on_grid(response_mask, reference, self.device, binary=True))
        fod_selection = (maskfilter_six_connected(mask, "dilate") if fod_mask is None else
                         _scalar_on_grid(fod_mask, reference, self.device, binary=True))
        norm_selection = (maskfilter_six_connected(mask, "erode") if normalise_mask is None else
                          _scalar_on_grid(normalise_mask, reference, self.device, binary=True))
        if fa_map is None:
            fa, _ = fit_mrtrix_dhollander_tensor(dwi_data, gradient, mask)
        else:
            fa = _scalar_on_grid(fa_map, reference, self.device)
        seg, seg_affine = _image(t1_segmentation, self.device)
        if seg.ndim != 3:
            raise ValueError("FreeSurfer aparc+aseg must be a 3D label image")
        five = freesurfer_five_tissue(seg)
        gmwmi = gmwmi_from_five_tissue(five)
        if dwi_to_t1_world is None:
            b0_brain = mean_b0 * mask
            transform = _registration(b0_brain, dwi_affine, t1_brain, self.device)
        else:
            transform = torch.as_tensor(dwi_to_t1_world, device=self.device,
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
                    target_shape=tuple(dwi_data.shape[:3]), target_affine=dwi_affine,
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
            )
            atlas_results[atlas_name] = AtlasResult(
                matrices, region_labels, atlas_data, atlas_affine, nodes,
            )
        first = atlas_results[atlas_names[0]]
        return ConnectomeResult(
            first.matrices, first.region_labels, first.atlas, five, five_affine, gmwmi,
            wm_sh, fa, mask, tracks, weights, dwi_affine, first.atlas_affine,
            transform, first.nodes, atlas_results,
        )


UKBConnectome = UKBConnectome_pipeline
