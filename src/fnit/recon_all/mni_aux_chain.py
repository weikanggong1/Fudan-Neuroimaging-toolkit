"""Subject-specific MNI152 affine and MCA/dura plus venous-sinus labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.synthmorph import SynthMorph

from .aux_seg import MCA_MODEL, VSINUS_MODEL, mri_mcadura_seg, mri_vsinus_seg


TEMPLATE_DIR = Path("average/mni_icbm152_nlin_asym_09c/reg-targets")


def write_mni_voxel_lta(output_file: str | Path, matrix: np.ndarray,
                        source_file: str | Path, target_file: str | Path) -> None:
    """Write the full-MNI-to-native voxel transform with nibabel geometry."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("expected a 4x4 voxel transform")

    def volume_info(path: str | Path) -> list[str]:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError("LTA volume geometry must be 3D")
        sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        if isinstance(image, nib.MGHImage):
            directions = np.asarray(image.header["Mdc"], dtype=np.float64)
            center = np.asarray(image.header["Pxyz_c"], dtype=np.float64)
        else:
            directions = (image.affine[:3, :3] / sizes).T + 0.0
            center = (image.affine @ np.array([*(d / 2 for d in image.shape), 1.0]))[:3]
        lines = ["valid = 1", "filename = none",
                 "volume = " + " ".join(str(d) for d in image.shape),
                 "voxelsize = " + " ".join(f"{v:.15e}" for v in sizes)]
        for name, row in zip(("xras", "yras", "zras"), directions):
            lines.append(name + "   = " + " ".join(f"{v:.15e}" for v in row))
        lines.append("cras   = " + " ".join(f"{v:.15e}" for v in center))
        return lines

    lines = ["type      = 0 # LINEAR_VOX_TO_VOX", "nxforms   = 1",
             "mean      = 0.0000 0.0000 0.0000", "sigma     = 1.0000",
             "1 4 4"]
    lines.extend(" ".join(f"{v:.15e}" for v in row) for row in transform)
    lines.extend(["src volume info", *volume_info(source_file),
                  "dst volume info", *volume_info(target_file)])
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def validate_mni_aux_assets(weights_dir: str | Path,
                            assets_dir: str | Path) -> None:
    """Require the three model weights, two MNI images and three priors."""
    weights, assets = Path(weights_dir), Path(assets_dir)
    required = (
        weights / "synthmorph.affine.2.h5", weights / MCA_MODEL,
        weights / VSINUS_MODEL,
        *(assets / TEMPLATE_DIR / f"mni152.1.0mm{suffix}.nii.gz"
          for suffix in (".cropped", "")),
        *(assets / "average" /
          f"mca-dura.prior.warp.mni152.1.0mm.{hemi}.nii.gz"
          for hemi in ("lh", "rh")),
        assets / "average/vsinus.no-sp.prior.mni152.1.0mm.mgz",
    )
    for file in required:
        if not file.is_file():
            raise FileNotFoundError(file)


def _crop_nonzero(image: Path, output: Path) -> Path:
    """Match the conformed-T1 bounding crop from mri_mask -bb 3."""
    source = nib.load(str(image))
    values = np.asarray(source.dataobj)
    bounds = []
    for axis in range(3):
        nonzero = np.flatnonzero(np.any(
            values != 0, axis=tuple(i for i in range(3) if i != axis)))
        if not len(nonzero):
            raise ValueError("orig.mgz has no nonzero voxels")
        bounds.append((max(0, int(nonzero[0]) - 3),
                       min(values.shape[axis], int(nonzero[-1]) + 3)))
    crop = np.ascontiguousarray(values[tuple(slice(lo, hi) for lo, hi in bounds)])
    translation = np.eye(4)
    translation[:3, 3] = [lo for lo, _ in bounds]
    affine = source.affine @ translation
    result = nib.Nifti1Image(crop, affine)
    result.set_sform(affine, code=1)
    result.set_qform(affine, code=1)
    result.header.set_xyzt_units("mm")
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))
    return output


def register_mni152_affine(subject_dir: str | Path, weights_dir: str | Path,
                           assets_dir: str | Path, *, device: str = "cpu",
                           threads: int = 4, precision_report: list | None = None) -> Path:
    """Write the full-MNI152-to-native voxel LTA used by auxiliary priors.

    Input: subject/mri/orig.mgz, external affine weight and cropped/full MNI152
    templates. Output: invol.crop.nii.gz, aff.lta and reg.targ_to_invol.lta in
    subject/mri/transforms/synthmorph.1.0mm.1.0mm. Returns the final LTA path.
    """
    subject = Path(subject_dir)
    native = subject / "mri/orig.mgz"
    target_dir = Path(assets_dir) / TEMPLATE_DIR
    cropped_target = target_dir / "mni152.1.0mm.cropped.nii.gz"
    full_target = target_dir / "mni152.1.0mm.nii.gz"
    weight = Path(weights_dir) / "synthmorph.affine.2.h5"
    for file in (native, cropped_target, full_target, weight):
        if not file.is_file():
            raise FileNotFoundError(file)
    transform_dir = subject / "mri/transforms/synthmorph.1.0mm.1.0mm"
    crop = _crop_nonzero(native, transform_dir / "invol.crop.nii.gz")
    torch.set_num_threads(threads)
    model = SynthMorph(weights=weights_dir, device=device, model="affine", extent=256, configure_precision=False)
    world_affine = model(crop, cropped_target, header_only=True, precision_report=precision_report).transform
    world_affine.save(str(transform_dir / "aff.lta"))
    native_image = nib.load(str(native))
    full_image = nib.load(str(full_target))
    world = world_affine.matrix
    target_to_native = np.linalg.inv(native_image.affine) @ np.linalg.inv(world) @ full_image.affine
    target_to_native[3] = (0, 0, 0, 1)
    output = transform_dir / "reg.targ_to_invol.lta"
    write_mni_voxel_lta(output_file=output, matrix=target_to_native,
                        source_file=full_target, target_file=native)
    return output


def run_mni_aux_chain(subject_dir: str | Path, weights_dir: str | Path,
                      assets_dir: str | Path, *, device: str = "cpu",
                      threads: int = 4) -> dict:
    """Generate the affine LTA and both conformed auxiliary label volumes.

    Requires subject/mri/orig.mgz, nu.mgz, synthseg.rca.mgz; external affine,
    MCA/dura, venous-sinus weights; cropped/full MNI152 targets and three
    priors. Returns paths keyed lta, mca_dura, vsinus. Also writes
    subject/stats/vsinus.stats when segmentation completes.
    """
    subject = Path(subject_dir)
    mri = subject / "mri"
    weights = Path(weights_dir)
    assets = Path(assets_dir)
    validate_mni_aux_assets(weights, assets)
    for file in (mri / "nu.mgz", mri / "synthseg.rca.mgz"):
        if not file.is_file():
            raise FileNotFoundError(file)
    forwards = []
    lta = register_mni152_affine(subject, weights, assets,
                                 device=device, threads=threads, precision_report=forwards)
    torch.set_num_threads(threads)
    directory = lta.parent
    mca_dura = mri_mcadura_seg(mri / "nu.mgz", mri / "mca-dura.mgz",
                               directory, assets, device=device, weights_dir=weights, precision_report=forwards)
    talairach = mri / "transforms/talairach.xfm.lta"
    vsinus = mri_vsinus_seg(mri / "nu.mgz", mri / "vsinus.mgz",
                            directory, assets, ctxseg_path=mri / "synthseg.rca.mgz",
                            stats_path=subject / "stats/vsinus.stats",
                            talairach_lta=talairach if talairach.is_file() else None,
                            device=device, weights_dir=weights, precision_report=forwards)
    return {"lta": lta, "mca_dura": mca_dura, "vsinus": vsinus,
            "runtime": {"device": str(device), "auxiliary_forwards": forwards}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(run_mni_aux_chain(args.subject_dir, args.weights, args.assets,
                            device=args.device, threads=args.threads))


if __name__ == "__main__":
    main()
