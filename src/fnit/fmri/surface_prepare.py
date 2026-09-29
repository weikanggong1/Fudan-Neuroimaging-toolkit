"""Prepare native cortical meshes on the MNI grid used by fMRI projection.

FreeSurfer vertices are in the conformed ``mri/orig.mgz`` tkRAS frame. The
saved FNIT nonlinear pull is MNI scanner RAS -> T1 scanner RAS; moving a
surface to MNI therefore requires solving its inverse at every vertex.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MNISurfacePair:
    white: Path
    pial: Path
    midthickness: Path
    vertex_count: int
    max_inverse_residual_mm: float


@dataclass(frozen=True)
class MNISurfaceResult:
    left: MNISurfacePair
    right: MNISurfacePair


def _apply_affine(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def invert_mni_to_t1_pull(
    t1_world_points: np.ndarray,
    pull_ras: str | Path,
    initial_t1_to_mni_world: np.ndarray,
    *,
    device: str = "cpu",
    tolerance_mm: float = 0.05,
    max_iterations: int = 80,
    chunk_size: int = 32768,
) -> tuple[np.ndarray, float]:
    """Solve ``mni_world + pull(mni_world) == t1_world`` for each point.

    ``pull`` is the 3-component RAS displacement on the MNI grid produced by
    ``register_t1_to_mni``. The affine only initializes the fixed-point solve;
    success requires every final nonlinear residual to meet ``tolerance_mm``.
    Out-of-field and nonconvergent vertices fail rather than receiving an
    affine-only coordinate.
    """
    points = np.asarray(t1_world_points, dtype=np.float64)
    initial = np.asarray(initial_t1_to_mni_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("t1_world_points must be finite [N,3]")
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("initial_t1_to_mni_world must be a finite 4x4 affine")
    if tolerance_mm <= 0 or max_iterations < 1 or chunk_size < 1:
        raise ValueError("tolerance_mm, max_iterations and chunk_size must be positive")
    image = nib.load(str(pull_ras))
    if not isinstance(image, nib.Nifti1Image) or image.ndim != 4 or image.shape[3] != 3:
        raise ValueError("pull_ras must be a 4D MNI-grid NIfTI with 3 RAS components")
    field = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(field).all():
        raise ValueError("pull_ras contains nonfinite values")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    volume = torch.from_numpy(field.transpose(3, 2, 1, 0).copy())[None].to(selected)
    world_to_voxel = torch.as_tensor(np.linalg.inv(image.affine), dtype=torch.float64, device=selected)
    shape = image.shape[:3]
    output = np.empty_like(points)
    maximum = 0.0

    def sample(world):
        voxel = world @ world_to_voxel[:3, :3].T + world_to_voxel[:3, 3]
        valid = torch.ones(len(world), dtype=torch.bool, device=selected)
        for axis, length in enumerate(shape):
            valid &= (voxel[:, axis] >= 0) & (voxel[:, axis] <= length - 1)
        if not bool(valid.all()):
            raise ValueError("surface vertex lies outside the MNI pull-field grid")
        grid = torch.stack(tuple(
            2 * voxel[:, axis] / (shape[axis] - 1) - 1 for axis in (0, 1, 2)
        ), dim=-1).float().reshape(1, len(world), 1, 1, 3)
        # The tensor dimensions are [Z,Y,X]; grid_sample expects [X,Y,Z].
        return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0, :, :, 0, 0].T.to(torch.float64)

    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        target = torch.as_tensor(points[start:stop], dtype=torch.float64, device=selected)
        estimate = torch.as_tensor(_apply_affine(points[start:stop], initial),
                                   dtype=torch.float64, device=selected)
        for _ in range(max_iterations):
            residual = estimate + sample(estimate) - target
            if float(torch.linalg.vector_norm(residual, dim=1).max()) <= tolerance_mm:
                break
            estimate = estimate - 0.75 * residual
        residual = estimate + sample(estimate) - target
        largest = float(torch.linalg.vector_norm(residual, dim=1).max())
        if not np.isfinite(largest) or largest > tolerance_mm:
            raise RuntimeError(f"MNI pull inverse did not converge: max residual {largest:.4f} mm")
        output[start:stop] = estimate.cpu().numpy()
        maximum = max(maximum, largest)
    return output, maximum


def _write_gifti(path, points, faces, hemi):
    image = nib.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(np.asarray(points, dtype=np.float32),
                                     intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(np.asarray(faces, dtype=np.int32),
                                     intent="NIFTI_INTENT_TRIANGLE"),
        ],
        meta=nib.gifti.GiftiMetaData({
            "AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight"
        }),
    )
    nib.save(image, str(path))


def prepare_t1w_surface_geometry(subject_dir: str | Path, output_dir: str | Path,
                                 *, overwrite: bool = False) -> MNISurfaceResult:
    """Convert recon-all white/pial meshes from tkRAS to scanner T1w RAS.

    The returned pairs have native FreeSurfer vertex order. This only reads
    recon-all files with nibabel; no FreeSurfer executable is invoked.
    """
    subject = Path(subject_dir).expanduser().resolve()
    orig = nib.load(str(subject / "mri/orig.mgz"))
    if not isinstance(orig, nib.MGHImage):
        raise ValueError("mri/orig.mgz must be an MGH image")
    transform = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for hemi in ("lh", "rh"):
        paths = {name: output / f"{hemi}.{name}.T1w.native.surf.gii"
                 for name in ("white", "pial", "midthickness")}
        if any(path.exists() for path in paths.values()) and not overwrite:
            raise FileExistsError(paths["white"])
        white, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.white"))
        pial, pial_faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.pial"))
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"{hemi} white and pial topology differ")
        white = _apply_affine(white, transform)
        pial = _apply_affine(pial, transform)
        for name, vertices in (("white", white), ("pial", pial),
                               ("midthickness", (white + pial) / 2)):
            _write_gifti(paths[name], vertices, faces, hemi)
        result[hemi] = MNISurfacePair(paths["white"], paths["pial"],
                                      paths["midthickness"], len(white), 0.0)
    return MNISurfaceResult(result["lh"], result["rh"])


@dataclass(frozen=True)
class T1SurfacePreparation:
    geometry: MNISurfaceResult
    initial_spheres: tuple[Path, Path]
    individual_rois: tuple[Path, Path]


def prepare_fmriprep_surface_inputs(
    subject_dir: str | Path, hcp_assets_dir: str | Path, output_dir: str | Path,
    *, wb_command: str | Path = "wb_command", overwrite: bool = False,
) -> T1SurfacePreparation:
    """Prepare T1w native meshes, FS-to-fsLR spheres and cortex ROIs.

    Requires existing recon-all ``orig.mgz``, white/pial, sphere.reg and
    thickness files. Does not create MNI surfaces or resample wmparc.
    """
    import shutil
    import subprocess

    subject = Path(subject_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    mesh = Path(hcp_assets_dir).expanduser().resolve() / "global/templates/standard_mesh_atlases"
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(wb_command)
    geometry = prepare_t1w_surface_geometry(
        subject, output / "native", overwrite=overwrite,
    )

    def command(destination, *args):
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True)
        if not destination.is_file():
            raise RuntimeError(f"Workbench did not create {destination}")

    spheres = []
    rois = []
    for hemi, fs_hemi, pair in (("L", "lh", geometry.left),
                                ("R", "rh", geometry.right)):
        vertices, faces = fsio.read_geometry(str(subject / "surf" / f"{fs_hemi}.sphere.reg"))
        native_faces = np.asarray(nib.load(str(pair.white)).darrays[1].data)
        if len(vertices) != pair.vertex_count or not np.array_equal(faces, native_faces):
            raise ValueError(f"{fs_hemi}.sphere.reg topology differs from white/pial")
        sphere = output / f"{hemi}.sphere.FS.native.surf.gii"
        _write_gifti(sphere, vertices, faces, fs_hemi)
        average = mesh / f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii"
        transform = mesh / (
            f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR."
            f"spherical_std.164k_fs_{hemi}.surf.gii"
        )
        registered = output / f"{hemi}.sphere.FS_to_fsLR.native.surf.gii"
        command(registered, "-surface-sphere-project-unproject", sphere,
                average, transform, registered)
        thickness = np.asarray(fsio.read_morph_data(
            str(subject / "surf" / f"{fs_hemi}.thickness")), dtype=np.float32)
        if len(thickness) != pair.vertex_count or not np.isfinite(thickness).all():
            raise ValueError(f"{fs_hemi}.thickness is invalid")
        raw = output / f"{hemi}.roi.thickness.native.shape.gii"
        filled = output / f"{hemi}.roi.filled.native.shape.gii"
        individual = output / f"{hemi}.roi.individual.native.shape.gii"
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            (np.abs(thickness) > 0).astype(np.float32), intent="NIFTI_INTENT_SHAPE"
        )]), str(raw))
        command(filled, "-metric-fill-holes", pair.midthickness, raw, filled)
        command(individual, "-metric-remove-islands", pair.midthickness,
                filled, individual)
        spheres.append(registered)
        rois.append(individual)
    return T1SurfacePreparation(geometry, tuple(spheres), tuple(rois))


def prepare_mni_surface_geometry(
    subject_dir: str | Path,
    pull_ras: str | Path,
    initial_t1_to_mni_world: np.ndarray,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    tolerance_mm: float = 0.05,
    overwrite: bool = False,
) -> MNISurfaceResult:
    """Convert an existing subject's white/pial meshes to MNI scanner RAS.

    Requires ``mri/orig.mgz`` to interpret tkRAS vertices correctly; the
    scanner-original ``mri/orig/001.mgz`` alone is insufficient. Writes two
    native-topology GIFTI surfaces and their arithmetic midpoint per side.
    This function does not prepare sphere registration, cortical ROI, or
    subcortical labels for the downstream projection stage.
    """
    subject = Path(subject_dir).expanduser().resolve()
    orig_file = subject / "mri/orig.mgz"
    if not orig_file.is_file():
        raise FileNotFoundError(f"FreeSurfer conformed geometry is required: {orig_file}")
    orig = nib.load(str(orig_file))
    if not isinstance(orig, nib.MGHImage):
        raise ValueError("mri/orig.mgz must be an MGH image")
    tk_to_world = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    output = Path(output_dir).expanduser().resolve()
    for hemi in ("lh", "rh"):
        if (output / f"{hemi}.white.MNI.native.surf.gii").exists() and not overwrite:
            raise FileExistsError(output / f"{hemi}.white.MNI.native.surf.gii")
    output.mkdir(parents=True, exist_ok=True)
    pairs = {}
    for hemi in ("lh", "rh"):
        inputs = {}
        for name in ("white", "pial"):
            vertices, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.{name}"))
            inputs[name] = (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
        white, white_faces = inputs["white"]
        pial, pial_faces = inputs["pial"]
        if white.shape != pial.shape or not np.array_equal(white_faces, pial_faces):
            raise ValueError(f"{hemi} white and pial must share ordered topology")
        # Invert once for both surfaces, preserving one-to-one vertex pairs.
        t1_points = _apply_affine(np.concatenate((white, pial)), tk_to_world)
        mni_points, residual = invert_mni_to_t1_pull(
            t1_points, pull_ras, initial_t1_to_mni_world,
            device=device, tolerance_mm=tolerance_mm,
        )
        midpoint = (mni_points[:len(white)] + mni_points[len(white):]) / 2
        paths = {name: output / f"{hemi}.{name}.MNI.native.surf.gii"
                 for name in ("white", "pial", "midthickness")}
        for name, values in (("white", mni_points[:len(white)]),
                             ("pial", mni_points[len(white):]),
                             ("midthickness", midpoint)):
            _write_gifti(paths[name], values, white_faces, hemi)
        pairs[hemi] = MNISurfacePair(paths["white"], paths["pial"],
                                     paths["midthickness"], len(white), residual)
    return MNISurfaceResult(pairs["lh"], pairs["rh"])


@dataclass(frozen=True)
class SurfacePreparationResult:
    """Paths accepted by ``SurfacePipelineInputs`` after FS sphere projection."""

    left: object
    right: object
    subject_rois: Path
    atlas_rois: Path
    inverse_residual_mm: dict[str, float]


def _resample_wmparc_to_mni(wmparc_path, pull_path, reference_path, output_path, device):
    """Nearest-neighbour pull of integer segmentation into the MNI reference."""
    source = nib.load(str(wmparc_path))
    field_image = nib.load(str(pull_path))
    reference = nib.load(str(reference_path))
    if source.ndim != 3 or reference.ndim != 3 or field_image.shape != (*reference.shape, 3):
        raise ValueError("wmparc, reference, or nonlinear pull has invalid dimensions")
    if not np.allclose(field_image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("nonlinear pull and MNI reference grids differ")
    shape = reference.shape
    selected = torch.device(device)
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64, device=selected)
                            for n in shape), indexing="ij")
    voxels = torch.stack(axes).reshape(3, -1)
    affine = torch.as_tensor(reference.affine, dtype=torch.float64, device=selected)
    world = affine[:3, :3] @ voxels + affine[:3, 3:4]
    displacement = np.asarray(field_image.dataobj, dtype=np.float32)
    if not np.isfinite(displacement).all():
        raise ValueError("nonlinear pull contains nonfinite values")
    world += torch.as_tensor(displacement.reshape(-1, 3).T.copy(),
                             dtype=torch.float64, device=selected)
    to_source = torch.as_tensor(np.linalg.inv(source.affine), dtype=torch.float64,
                                device=selected)
    coords = to_source[:3, :3] @ world + to_source[:3, 3:4]
    grid = torch.stack(tuple(2 * coords[axis] / max(source.shape[axis] - 1, 1) - 1
                             for axis in (0, 1, 2)), dim=-1).float()
    grid = grid.reshape(1, *shape, 3)
    data = np.asarray(source.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("wmparc contains nonfinite labels")
    tensor = torch.as_tensor(data.transpose(2, 1, 0).copy(),
                             device=selected)[None, None]
    labels = F.grid_sample(tensor, grid.permute(0, 3, 2, 1, 4),
                           mode="nearest", padding_mode="zeros",
                           align_corners=True)[0, 0].permute(2, 1, 0)
    result = np.rint(labels.cpu().numpy()).astype(np.int32)
    if not np.any(result > 0):
        raise ValueError("MNI-resampled wmparc contains no labels")
    nib.save(nib.Nifti1Image(result, reference.affine), str(output_path))
    return output_path


def prepare_fs_sphere_projection_inputs(
    subject_dir: str | Path,
    pull_ras: str | Path,
    initial_t1_to_mni_world: np.ndarray,
    mni_reference: str | Path,
    hcp_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
    device: str = "cpu",
    overwrite: bool = False,
) -> SurfacePreparationResult:
    """Prepare explicit FS sphere projection assets from one FreeSurfer subject.

    ``hcp_assets_dir`` follows the pinned HCPpipelines directory layout.
    Requires the HCP FreeSurfer and subcortical label LUTs in ``global/config``.
    The resulting registered sphere is the FreeSurfer-to-fsLR projection,
    without MSMSulc/MSMAll estimation. The subject ROI follows the saved
    MNI-to-T1 nonlinear pull, not the initial affine.
    """
    import shutil
    import subprocess
    from .surface import SurfaceHemisphere

    subject = Path(subject_dir).expanduser().resolve()
    assets = Path(hcp_assets_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    mesh = assets / "global/templates/standard_mesh_atlases"
    atlas_rois = assets / "global/templates/91282_Greyordinates/Atlas_ROIs.2.nii.gz"
    all_lut = assets / "global/config/FreeSurferAllLut.txt"
    subcortical_lut = assets / "global/config/FreeSurferSubcorticalLabelTableLut.txt"
    for path in (atlas_rois, all_lut, subcortical_lut, subject / "mri/wmparc.mgz"):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = nib.load(str(mni_reference))
    pull_image = nib.load(str(pull_ras))
    atlas_image = nib.load(str(atlas_rois))
    if not isinstance(reference, nib.Nifti1Image) or reference.ndim != 3:
        raise ValueError("mni_reference must be a 3D NIfTI")
    for name, image in (("pull_ras", pull_image), ("Atlas_ROIs", atlas_image)):
        expected = (*reference.shape, 3) if name == "pull_ras" else reference.shape
        if image.shape != expected or not np.allclose(
            image.affine, reference.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"{name} does not match the MNI reference grid")
    if not np.allclose(reference.header.get_zooms()[:3], 2.0, atol=0.01, rtol=0):
        raise ValueError("mni_reference must have 2-mm voxels")
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
    output.mkdir(parents=True, exist_ok=True)
    native = prepare_mni_surface_geometry(
        subject, pull_ras, initial_t1_to_mni_world, output / "native",
        device=device, overwrite=overwrite,
    )

    def command(destination, *args):
        destination = Path(destination)
        destination.unlink(missing_ok=True)
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True)
        if not destination.is_file():
            raise RuntimeError(f"Workbench did not create {destination}")

    pairs = {}
    for hemi, pair in (("L", native.left), ("R", native.right)):
        fs_hemi = "lh" if hemi == "L" else "rh"
        sphere_vertices, sphere_faces = fsio.read_geometry(
            str(subject / "surf" / f"{fs_hemi}.sphere.reg")
        )
        native_faces = np.asarray(nib.load(str(pair.white)).darrays[1].data)
        if (len(sphere_vertices) != pair.vertex_count
                or not np.array_equal(sphere_faces, native_faces)):
            raise ValueError(f"{fs_hemi}.sphere.reg does not match white/pial topology")
        sphere_native = output / f"{hemi}.sphere.FS.native.surf.gii"
        _write_gifti(sphere_native, sphere_vertices, sphere_faces, fs_hemi)
        fsaverage_sphere = mesh / f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii"
        fs_to_fslr = mesh / (
            f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR."
            f"spherical_std.164k_fs_{hemi}.surf.gii"
        )
        atlas_sphere_164 = mesh / f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"
        atlas_roi_164 = mesh / f"{hemi}.atlasroi.164k_fs_LR.shape.gii"
        atlas_sphere_32 = mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii"
        atlas_roi_32 = mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii"
        for path in (fsaverage_sphere, fs_to_fslr, atlas_sphere_164,
                     atlas_roi_164, atlas_sphere_32, atlas_roi_32):
            if not path.is_file():
                raise FileNotFoundError(path)
        registered = output / f"{hemi}.sphere.FS_to_fsLR.native.surf.gii"
        command(registered, "-surface-sphere-project-unproject", sphere_native,
                fsaverage_sphere, fs_to_fslr, registered)
        thickness = np.asarray(fsio.read_morph_data(
            str(subject / "surf" / f"{fs_hemi}.thickness")
        ))
        if len(thickness) != pair.vertex_count or not np.isfinite(thickness).all():
            raise ValueError(f"{fs_hemi} thickness vertex count or values are invalid")
        raw_roi = output / f"{hemi}.roi.thickness.native.shape.gii"
        filled_roi = output / f"{hemi}.roi.filled.native.shape.gii"
        individual_roi = output / f"{hemi}.roi.individual.native.shape.gii"
        projected_roi = output / f"{hemi}.atlasroi.native_projected.shape.gii"
        native_roi = output / f"{hemi}.roi.native.shape.gii"
        nib.save(nib.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(
                (np.abs(thickness) > 0).astype(np.float32),
                intent="NIFTI_INTENT_SHAPE"
            )],
            meta=nib.gifti.GiftiMetaData({
                "AnatomicalStructurePrimary": "CortexLeft" if hemi == "L" else "CortexRight"
            }),
        ), str(raw_roi))
        command(filled_roi, "-metric-fill-holes", pair.midthickness,
                raw_roi, filled_roi)
        command(individual_roi, "-metric-remove-islands", pair.midthickness,
                filled_roi, individual_roi)
        command(projected_roi, "-metric-resample", atlas_roi_164,
                atlas_sphere_164, registered, "BARYCENTRIC", projected_roi,
                "-largest")
        command(native_roi, "-metric-math", "(atlas + individual) > 0",
                native_roi, "-var", "atlas", projected_roi,
                "-var", "individual", individual_roi)
        final_roi = np.asarray(nib.load(str(native_roi)).darrays[0].data)
        if len(final_roi) != pair.vertex_count or not np.any(final_roi > 0):
            raise ValueError(f"{fs_hemi} cleaned native cortex ROI is empty or mismatched")
        atlas_mid = output / f"{hemi}.midthickness.32k_fsLR.surf.gii"
        command(atlas_mid, "-surface-resample", pair.midthickness,
                registered, atlas_sphere_32, "BARYCENTRIC", atlas_mid)
        pairs[hemi] = SurfaceHemisphere(
            white=pair.white, pial=pair.pial, midthickness=pair.midthickness,
            registered_sphere=registered, native_roi=native_roi,
            atlas_sphere=atlas_sphere_32, atlas_midthickness=atlas_mid,
            atlas_roi=atlas_roi_32,
        )

    wmparc_mni = output / "wmparc.MNI152_2mm.nii.gz"
    _resample_wmparc_to_mni(subject / "mri/wmparc.mgz", pull_ras,
                            mni_reference, wmparc_mni, device)
    wmparc_label = output / "wmparc.MNI152_2mm.label.nii.gz"
    subject_rois = output / "ROIs.2.nii.gz"
    command(wmparc_label, "-volume-label-import", wmparc_mni, all_lut,
            wmparc_label, "-drop-unused-labels")
    command(subject_rois, "-volume-label-import", wmparc_label,
            subcortical_lut, subject_rois, "-discard-others")
    return SurfacePreparationResult(
        left=pairs["L"], right=pairs["R"],
        subject_rois=subject_rois, atlas_rois=atlas_rois,
        inverse_residual_mm={"L": native.left.max_inverse_residual_mm,
                             "R": native.right.max_inverse_residual_mm},
    )
