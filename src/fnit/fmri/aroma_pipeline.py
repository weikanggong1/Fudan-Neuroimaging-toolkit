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
    n_components,
    tr=None,
    mode="nonaggr",
    device=None,
    n_splits=1000,
    random_state=0,
    ica_max_iter=500,
    wm_mask=None,
    regress_csf=False,
    regress_motion=False,
    motion_model=24,
    bandpass=None,
    global_signal=False,
):
    """Decompose, classify, denoise, then optionally regress other confounds.

    Every mask must already match the BOLD voxel grid. The ICA maps use a
    simple |Z| threshold; they are not MELODIC mixture-model threshold maps.
    Thus the classifier rule is reproduced but complete official ICA-AROMA
    component classification equivalence is not claimed.
    """
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
    features = classify_aroma(
        ica.thresholded_maps, ica.mixing, ica.frequency_power,
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
            wm_mask=wm_mask, csf_mask=csf_mask if regress_csf else None,
            brain_mask=brain_mask, motion=motion_parameters if regress_motion else None,
            motion_model=motion_model, bandpass=bandpass, tr=seconds,
            global_signal=global_signal, device=device,
        )
    return AromaResult(ica, feature_path, noise_path, denoised, clean_path)
