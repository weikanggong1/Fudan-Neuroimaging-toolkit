"""Move MNI volume masks onto the BEDPOSTX diffusion grid before tracking."""

from pathlib import Path

import nibabel as nib
import numpy as np

from ..applywarp import TorchApplyWarp
from ..convertwarp import TorchConvertWarp
from ..invwarp import TorchInvWarp


def _same_grid(first, second):
    return first.shape == second.shape and np.allclose(
        first.affine, second.affine, atol=1e-4, rtol=0)


def _mask_list(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        listing = Path(value)
        if listing.name.endswith((".nii", ".nii.gz")):
            return [listing]
        return [path if path.is_absolute() else listing.parent / path
                for line in listing.read_text().splitlines() if line.strip()
                for path in [Path(line.strip())]]
    return [Path(path) for path in value]


def prepare_mni_masks(samples_dir, output_dir, *, mni_reference, diff2struct_mat,
                      struct2mni_warp, diff2mni_warp, dmri_pipeline_dir,
                      registration_backend, device, overwrite, **masks):
    """Return diffusion-grid mask paths; save the composite and inverse fields."""
    diffusion = nib.load(str(Path(samples_dir) / "nodif_brain_mask.nii.gz"))
    singles = ("seed", "mask", "avoid", "stop", "target2", "target3", "lrtarget3")
    sequences = ("regions", "waypoints", "wtstop", "targetmasks")
    values = {name: ([Path(masks[name])] if masks.get(name) is not None else None)
              for name in singles}
    values.update({name: _mask_list(masks.get(name)) for name in sequences})
    all_paths = [path for paths in values.values() if paths for path in paths]
    images = {path: nib.load(str(path)) for path in all_paths}
    mni_paths = [path for path in all_paths if not _same_grid(images[path], diffusion)]
    if not mni_paths:
        return masks, None
    modes = sum((dmri_pipeline_dir is not None, diff2mni_warp is not None,
                 diff2struct_mat is not None or struct2mni_warp is not None))
    if modes != 1:
        raise ValueError("mask geometry differs from diffusion; give one of "
                         "dmri_pipeline_dir, diff2mni_warp, or "
                         "diff2struct_mat plus struct2mni_warp")
    if dmri_pipeline_dir is not None:
        root = Path(dmri_pipeline_dir)
        registration = root / "registration"
        tbss_warp = registration / "dti_FA_to_MNI_warp.nii.gz"
        mmorf_warp = registration / "mmorf_warp.nii.gz"
        if registration_backend == "auto":
            available = [name for name, path in (("tbss", tbss_warp),
                                                  ("mmorf", mmorf_warp)) if path.is_file()]
            if len(available) != 1:
                raise ValueError("pipeline registration must contain exactly one "
                                 "TBSS or MMORF warp; specify registration_backend")
            registration_backend = available[0]
        if registration_backend not in ("tbss", "mmorf"):
            raise ValueError("registration_backend must be auto, tbss, or mmorf")
        native = nib.load(str(root / "native" / "dti_FA.nii.gz"))
        if not _same_grid(native, diffusion):
            raise ValueError("BEDPOSTX mask geometry must match pipeline native/dti_FA.nii.gz")
        if mni_reference is None:
            mni_reference = registration / "standard" / "FA.nii.gz"
    elif registration_backend != "auto":
        raise ValueError("registration_backend requires dmri_pipeline_dir")
    if diff2struct_mat is not None and struct2mni_warp is None:
        raise ValueError("diff2struct_mat requires struct2mni_warp")
    if struct2mni_warp is not None and diff2struct_mat is None:
        raise ValueError("struct2mni_warp requires diff2struct_mat")
    mni = nib.load(str(mni_reference)) if mni_reference is not None else images[mni_paths[0]]
    directory = Path(output_dir) / "mni_to_diffusion"
    forward_path = directory / "diff2mni_warp.nii.gz"
    inverse_path = directory / "mni2diff_warp.nii.gz"
    outputs = [forward_path, inverse_path]
    mapped = {}
    for index, path in enumerate(dict.fromkeys(mni_paths)):
        target = directory / "masks" / f"{index:03d}" / path.name
        mapped[path] = target
        outputs.append(target)
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    model = TorchConvertWarp(device)
    if dmri_pipeline_dir is not None and registration_backend == "mmorf":
        forward = model.run_mmorf(
            reference=mni, source=native, mmorf_warp=mmorf_warp,
            affine=registration / "dti_FA_to_MNI_affine.mat", output=forward_path)
    else:
        warp = (tbss_warp if dmri_pipeline_dir is not None
                else diff2mni_warp if diff2mni_warp is not None else struct2mni_warp)
        forward = model.run(reference=mni, warp1=warp,
                            premat=diff2struct_mat, output=forward_path)
    inverse = TorchInvWarp(device).run(
        reference=diffusion, warp=forward.image, output=inverse_path,
        warp_convention="relative")
    warper = TorchApplyWarp(device)
    for source, target in mapped.items():
        binary = nib.Nifti1Image((np.asarray(images[source].dataobj) > 0).astype(np.uint8),
                                 images[source].affine, images[source].header)
        warper.run(input=binary, reference=diffusion, output=target,
                   warp=inverse.image, warp_convention="relative",
                   interpolation="nearest", output_dtype="char")

    converted = dict(masks)
    for name, paths in values.items():
        if paths is not None:
            results = [mapped.get(path, path) for path in paths]
            converted[name] = results if name in sequences else results[0]
    return converted, directory
