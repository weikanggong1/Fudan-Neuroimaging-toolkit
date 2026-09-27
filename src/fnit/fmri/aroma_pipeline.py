"""ICA-AROMA replacement after pre-FIX FEAT preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .aroma import classify_aroma, denoise_aroma
from .confounds import clean_confounds
from .ica import ICAResult, decompose_spatial_ica


@dataclass(frozen=True)
class AromaResult:
    ica: ICAResult
    features: Path
    noise_components: Path
    denoised_bold: Path
    confounds_cleaned_bold: Path | None


def run_aroma_pipeline(
    filtered_func_data,
    brain_mask,
    motion_parameters,
    csf_mask,
    edge_mask,
    outside_mask,
    output_dir,
    *,
    n_components=None,
    tr=None,
    mode="nonaggr",
    device=None,
    n_splits=1000,
    random_state=0,
    ica_max_iter=500,
    wm_mask=None,
    regression_csf_mask=None,
    regress_csf=False,
    regress_motion=False,
    motion_model=24,
    bandpass=None,
    global_signal=False,
    mni_template=None,
    mni_pull_ras=None,
    epi_to_t1_world=None,
):
    """Decompose, classify, denoise, then optionally regress other confounds.

    Classification masks must match the thresholded IC maps. Supply all three
    MNI arguments to classify in MNI space; BOLD and optional regression masks
    remain on the native EPI grid. With
    ``n_components=None``, PICA selects the component count by a Laplace PPCA
    approximation. Thresholded maps use a positive/negative Gamma-Gaussian
    mixture posterior; component identity still requires real-data comparison.
    ``regression_csf_mask`` supplies a native EPI tissue mask when MNI-space
    classification and CSF regression are both requested.
    """
    transforms = (mni_template, mni_pull_ras, epi_to_t1_world)
    if any(value is not None for value in transforms) and not all(
        value is not None for value in transforms
    ):
        raise ValueError("mni_template, mni_pull_ras and epi_to_t1_world must be supplied together")
    if mni_template is not None:
        if regress_csf and regression_csf_mask is None:
            raise ValueError("regression_csf_mask on the native EPI grid is required for MNI classification with regress_csf=True")
        target = nib.load(str(mni_template))
        for mask in (csf_mask, edge_mask, outside_mask):
            mask_image = nib.load(str(mask))
            if mask_image.shape != target.shape or not np.allclose(
                mask_image.affine, target.affine, atol=1e-4
            ):
                raise ValueError("classification masks must match mni_template")
    source = Path(filtered_func_data)
    image = nib.load(str(source))
    if image.ndim != 4:
        raise ValueError("filtered_func_data must be a 4D NIfTI image")
    unit = image.header.get_xyzt_units()[1]
    unit_scale = {"msec": 0.001, "usec": 0.000001}.get(unit, 1.0)
    seconds = float(tr if tr is not None else image.header.get_zooms()[3] * unit_scale)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("tr must be positive and finite")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ica = decompose_spatial_ica(
        source, brain_mask, output / "ica", n_components=n_components,
        device=device, random_state=random_state, max_iter=ica_max_iter,
    )
    if not ica.converged:
        raise RuntimeError("spatial ICA did not converge; increase max_iter in decompose_spatial_ica")
    classification_maps = ica.thresholded_maps
    if mni_template is not None:
        from .normalization import resample_world

        classification_maps = resample_world(
            ica.thresholded_maps, mni_template,
            np.linalg.inv(np.asarray(epi_to_t1_world, dtype=np.float64)),
            output / "ica_thresholded_MNI152_2mm.nii.gz",
            pre_affine_pull_ras=mni_pull_ras, device=device,
        )
    features = classify_aroma(
        classification_maps, ica.mixing, ica.frequency_power,
        motion_parameters, csf_mask, edge_mask, outside_mask, seconds,
        n_splits=n_splits, random_state=random_state,
    )
    feature_path = output / "aroma_features.tsv"
    values = np.column_stack((
        np.arange(1, ica.n_components + 1), features["max_rp_corr"],
        features["edge_fraction"], features["high_freq_content"],
        features["csf_fraction"],
    ))
    np.savetxt(feature_path, values, fmt=["%d", "%.9g", "%.9g", "%.9g", "%.9g"],
               delimiter="\t", header="component\tmax_rp_corr\tedge_fraction\thigh_freq_content\tcsf_fraction", comments="")
    noise_path = output / "aroma_noise_components.txt"
    noise_path.write_text("\n".join(str(index + 1) for index in features["noise_indices"]) + "\n")
    denoised = denoise_aroma(
        source, ica.mixing, features["noise_indices"],
        output / "filtered_func_data_aroma.nii.gz", mode=mode, device=device,
    )
    clean_path = None
    if wm_mask is not None or regress_csf or regress_motion or bandpass is not None or global_signal:
        clean_path = clean_confounds(
            denoised, output / "filtered_func_data_aroma_confounds.nii.gz",
            wm_mask=wm_mask,
            csf_mask=(regression_csf_mask or csf_mask) if regress_csf else None,
            brain_mask=brain_mask, motion=motion_parameters if regress_motion else None,
            motion_model=motion_model, bandpass=bandpass, tr=seconds,
            global_signal=global_signal, device=device,
        )
    return AromaResult(ica, feature_path, noise_path, denoised, clean_path)
