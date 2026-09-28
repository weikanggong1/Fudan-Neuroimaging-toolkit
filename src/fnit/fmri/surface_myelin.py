"""Create native and fsLR32k T1w/FLAIR myelin maps with Workbench."""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .surface import SurfaceHemisphere
from .surface_prepare import _apply_affine, _write_gifti


@dataclass(frozen=True)
class MyelinHemisphere:
    native_map: Path
    native_map_corrected: Path
    atlas_map_corrected: Path
    native_ribbon: Path


def create_surface_myelin_maps(
    t1_restored: str | Path,
    t2_restored: str | Path,
    subject_dir: str | Path,
    left: SurfaceHemisphere,
    right: SurfaceHemisphere,
    hcp_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
) -> dict[str, MyelinHemisphere]:
    """Map bias-corrected T1w/FLAIR ratio to native cortex and fsLR32k.

    ``left`` and ``right`` supply native registration spheres and ROI metrics;
    their MNI-space white/pial surfaces are not used. Scanner-space surfaces
    are reconstructed from the matching FreeSurfer directory. The Workbench
    myelin sampling and low-frequency reference correction follow UKB v1.5.
    """
    t1_image = nib.load(str(Path(t1_restored).expanduser().resolve()))
    t2_image = nib.load(str(Path(t2_restored).expanduser().resolve()))
    subject = Path(subject_dir).expanduser().resolve()
    assets = Path(hcp_assets_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
    if (t1_image.ndim != 3 or t2_image.shape != t1_image.shape
            or not np.allclose(t1_image.affine, t2_image.affine, atol=1e-4, rtol=0)):
        raise ValueError("restored T1w and FLAIR must occupy the same 3D grid")
    orig = nib.load(str(subject / "mri/orig.mgz"))
    scanner = nib.load(str(subject / "mri/orig/001.mgz"))
    if (scanner.shape != t1_image.shape or not np.allclose(
            scanner.affine, t1_image.affine, atol=1e-4, rtol=0)):
        raise ValueError("recon-all surfaces and restored T1w use different scanner grids")
    t1 = np.asarray(t1_image.dataobj, dtype=np.float32)
    t2 = np.asarray(t2_image.dataobj, dtype=np.float32)
    if not np.isfinite(t1).all() or not np.isfinite(t2).all():
        raise ValueError("restored structural images contain nonfinite values")
    mesh = assets / "global/templates/standard_mesh_atlases"
    reference_map = mesh / "Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii"
    for path in (reference_map, left.registered_sphere, right.registered_sphere,
                 left.native_roi, right.native_roi):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    wb_env = os.environ.copy()
    wb_env.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 8, 8)))

    def command(*args):
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True, env=wb_env)

    ratio = np.divide(t1, t2, out=np.zeros_like(t1), where=t2 != 0)
    ratio[(t2 == 0) & (t1 > 0)] = 100
    np.clip(ratio, 0, 100, out=ratio)
    ratio_file = output / "T1wDividedByT2w.nii.gz"
    nib.save(nib.Nifti1Image(ratio, t1_image.affine, t1_image.header), str(ratio_file))
    left_reference = output / "L.RefMyelinMap.164k.func.gii"
    right_reference = output / "R.RefMyelinMap.164k.func.gii"
    command("-cifti-separate-all", reference_map,
            "-left", left_reference, "-right", right_reference)
    tk_to_world = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    result = {}
    myelin_sigma = 5 / (2 * np.sqrt(2 * np.log(2)))
    surface_sigma = 4 / (2 * np.sqrt(2 * np.log(2)))
    correction_sigma = np.sqrt(200)
    for hemi, fs_hemi, prepared, reference_metric in (
        ("L", "lh", left, left_reference), ("R", "rh", right, right_reference)
    ):
        folder = output / hemi
        folder.mkdir(exist_ok=True)
        white, faces = fsio.read_geometry(str(subject / "surf" / f"{fs_hemi}.white"))
        pial, pial_faces = fsio.read_geometry(str(subject / "surf" / f"{fs_hemi}.pial"))
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"{fs_hemi} white and pial topology differ")
        white = _apply_affine(white, tk_to_world)
        pial = _apply_affine(pial, tk_to_world)
        mid = (white + pial) / 2
        white_file, pial_file, mid_file = (
            folder / f"{hemi}.{name}.T1.native.surf.gii"
            for name in ("white", "pial", "midthickness")
        )
        for path, points in ((white_file, white), (pial_file, pial), (mid_file, mid)):
            _write_gifti(path, points, faces, fs_hemi)
        thickness = np.asarray(fsio.read_morph_data(
            str(subject / "surf" / f"{fs_hemi}.thickness")
        ), dtype=np.float32)
        if thickness.shape != (len(white),):
            raise ValueError(f"{fs_hemi} thickness does not match surface topology")
        thickness_file = folder / f"{hemi}.thickness.native.shape.gii"
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            thickness, intent="NIFTI_INTENT_SHAPE")]), str(thickness_file))
        white_distance = folder / "white_distance.nii.gz"
        pial_distance = folder / "pial_distance.nii.gz"
        command("-create-signed-distance-volume", white_file, t1_restored, white_distance)
        command("-create-signed-distance-volume", pial_file, t1_restored, pial_distance)
        white_outside = np.asarray(nib.load(str(white_distance)).dataobj) > 0
        pial_inside = np.asarray(nib.load(str(pial_distance)).dataobj) < 0
        grey = white_outside & pial_inside
        ribbon = folder / f"{hemi}.ribbon.nii.gz"
        nib.save(nib.Nifti1Image(grey.astype(np.uint8), t1_image.affine,
                                 t1_image.header), str(ribbon))
        native_map = folder / f"{hemi}.MyelinMap.native.func.gii"
        smoothed_map = folder / f"{hemi}.SmoothedMyelinMap.native.func.gii"
        command("-volume-to-surface-mapping", ratio_file, mid_file, native_map,
                "-myelin-style", ribbon, thickness_file, myelin_sigma)
        command("-metric-smoothing", mid_file, native_map, surface_sigma,
                smoothed_map, "-roi", prepared.native_roi)

        sphere_164 = mesh / f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"
        sphere_32 = mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii"
        roi_164 = mesh / f"{hemi}.atlasroi.164k_fs_LR.shape.gii"
        roi_32 = mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii"
        mid_164, mid_32 = (folder / f"{hemi}.midthickness.{size}k.surf.gii"
                           for size in (164, 32))
        for sphere, destination in ((sphere_164, mid_164), (sphere_32, mid_32)):
            command("-surface-resample", mid_file, prepared.registered_sphere,
                    sphere, "BARYCENTRIC", destination)
        reference_native = folder / f"{hemi}.RefMyelinMap.native.func.gii"
        command("-metric-resample", reference_metric, sphere_164,
                prepared.registered_sphere, "ADAP_BARY_AREA", reference_native,
                "-area-surfs", mid_164, mid_file, "-current-roi", roi_164)
        command("-metric-dilate", reference_native, mid_file, 30,
                reference_native, "-nearest")
        command("-metric-mask", reference_native, prepared.native_roi, reference_native)
        correction_native = []
        for name, metric in (("MyelinMap", native_map),
                             ("RefMyelinMap", reference_native)):
            at_32 = folder / f"{hemi}.{name}.32k.func.gii"
            smooth_32 = folder / f"{hemi}.{name}.smooth.32k.func.gii"
            smooth_native = folder / f"{hemi}.{name}.smooth.native.func.gii"
            command("-metric-resample", metric, prepared.registered_sphere,
                    sphere_32, "ADAP_BARY_AREA", at_32, "-area-surfs", mid_file,
                    mid_32, "-current-roi", prepared.native_roi)
            command("-metric-smoothing", mid_32, at_32, correction_sigma,
                    smooth_32, "-roi", roi_32)
            command("-metric-resample", smooth_32, sphere_32,
                    prepared.registered_sphere, "ADAP_BARY_AREA", smooth_native,
                    "-area-surfs", mid_32, mid_file, "-current-roi", roi_32)
            command("-metric-dilate", smooth_native, mid_file, 30,
                    smooth_native, "-nearest")
            command("-metric-mask", smooth_native, prepared.native_roi, smooth_native)
            correction_native.append(smooth_native)
        bias = folder / f"{hemi}.myelin_bias.native.func.gii"
        corrected = folder / f"{hemi}.MyelinMap_BC.native.func.gii"
        corrected_32 = folder / f"{hemi}.MyelinMap_BC.32k.func.gii"
        command("-metric-math", "(Individual - Reference) * Mask", bias,
                "-var", "Individual", correction_native[0],
                "-var", "Reference", correction_native[1],
                "-var", "Mask", prepared.native_roi)
        command("-metric-math", "(Individual - Bias) * Mask", corrected,
                "-var", "Individual", native_map, "-var", "Bias", bias,
                "-var", "Mask", prepared.native_roi)
        command("-metric-resample", corrected, prepared.registered_sphere,
                sphere_32, "ADAP_BARY_AREA", corrected_32,
                "-area-surfs", mid_file, mid_32,
                "-current-roi", prepared.native_roi)
        command("-metric-mask", corrected_32, roi_32, corrected_32)
        result[hemi] = MyelinHemisphere(native_map, corrected, corrected_32, ribbon)
    return result
