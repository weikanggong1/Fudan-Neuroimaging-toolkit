"""Read-only FastVBM stage isolation from already saved real-data arrays.

No model fitting, original software calls, benchmark timings or image outputs.
The original coefficients are diagnostic inputs, never production inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from fnit.applywarp import TorchApplyWarp
from fnit._transforms import DenseWarp
from fnit.fast_vbm.registration import _common_applywarp, _pull_ras_to_fsl_fields


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaled_mm(image):
    matrix = np.diag((*image.header.get_zooms()[:3], 1.)).astype(np.float64)
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = (image.shape[0] - 1) * image.header.get_zooms()[0]
    return matrix


def metrics(a, b, selection):
    a, b = a[selection].astype(np.float64), b[selection].astype(np.float64)
    error = a - b
    scale = float(np.percentile(b, 99) - np.percentile(b, 1))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"values": int(a.size), "different_values": int(np.count_nonzero(error)),
            "mae": float(np.mean(np.abs(error))), "rmse": rmse,
            "p99_absolute_error": float(np.percentile(np.abs(error), 99)),
            "maximum_absolute_error": float(np.abs(error).max()),
            "reference_p99_minus_p1": scale, "nrmse": rmse / scale if scale else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    template = nib.load(args.template)
    native = nib.load(args.reference / "T1_brain_pve_1.nii.gz")
    candidate_native = nib.load(args.candidate / "T1_brain_pve_1.nii.gz")
    native_data = np.asanyarray(native.dataobj)
    candidate_native_data = np.asanyarray(candidate_native.dataobj)
    brain = ((np.asanyarray(nib.load(args.reference_mask).dataobj) > 0) &
             (np.asanyarray(template.dataobj) > 0))
    regions = {"whole_grid": np.ones(template.shape, dtype=bool), "template_brain": brain}
    report = {"scope": "saved real GM, affine and original warp posthoc isolation; no optimization or benchmark",
              "image_shape": native.shape, "template_shape": template.shape,
              "source_identity": {}, "comparisons": {}}
    names = ["T1_brain_pve_1.nii.gz", "gm_affine.mat", "gm_coeff.nii.gz", "gm_dense.nii.gz",
             "T1_GM_to_template_GM.nii.gz", "T1_GM_JAC_nl.nii.gz", "T1_GM_to_template_GM_mod.nii.gz"]
    for group, directory in [("reference", args.reference), ("candidate", args.candidate)]:
        for name in names + ["fast_vbm_report.json"]:
            path = directory / name
            if path.exists():
                report["source_identity"][group + "/" + name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for name, path in [("template", args.template), ("reference_mask", args.reference_mask)]:
        report["source_identity"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    def compare(name, a, b):
        report["comparisons"][name] = {key: metrics(a, b, selection) for key, selection in regions.items()}

    original_warped = np.asanyarray(nib.load(args.reference / "T1_GM_to_template_GM.nii.gz").dataobj)
    candidate_warped = np.asanyarray(nib.load(args.candidate / "T1_GM_to_template_GM.nii.gz").dataobj)
    compare("frozen_pipeline_warped_gm", candidate_warped, original_warped)
    apply = TorchApplyWarp(device="cpu")
    coeff_result = apply(native, template, warp=nib.load(args.reference / "gm_coeff.nii.gz"),
                         interpolation="trilinear", output_dtype="float")
    coeff_warped = np.asanyarray(coeff_result.image.dataobj)
    compare("same_original_coefficients_fnit_applywarp_vs_original", coeff_warped, original_warped)
    dense_result = apply(native, template, warp=nib.load(args.reference / "gm_dense.nii.gz"),
                         interpolation="trilinear", warp_convention="relative", output_dtype="float")
    dense_warped = np.asanyarray(dense_result.image.dataobj)
    compare("same_original_dense_fnit_applywarp_vs_original", dense_warped, original_warped)
    compare("same_original_coefficients_vs_original_dense_fnit_applywarp", coeff_warped, dense_warped)
    upstream_result = apply(candidate_native, template, warp=nib.load(args.reference / "gm_coeff.nii.gz"),
                            interpolation="trilinear", output_dtype="float")
    compare("only_frozen_fast_gm_change_same_original_coefficients",
            np.asanyarray(upstream_result.image.dataobj), coeff_warped)
    report["original_coefficients_applywarp_qc"] = coeff_result.qc
    report["original_dense_applywarp_qc"] = dense_result.qc

    # Independent NumPy/SciPy linear sampling isolates the recorded affine
    # delta. It is not an assertion of FSL applywarp border equivalence.
    candidate_report = json.loads((args.candidate / "fast_vbm_report.json").read_text())
    original_forward = np.loadtxt(args.reference / "gm_affine.mat")
    candidate_pull_world = np.array(candidate_report["registration"]["pull_world_affine"])
    candidate_forward = (scaled_mm(template) @ np.linalg.inv(template.affine) @
                         np.linalg.inv(candidate_pull_world) @ candidate_native.affine @
                         np.linalg.inv(scaled_mm(candidate_native)))
    grid = np.indices(template.shape, dtype=np.float64).reshape(3, -1)
    target_fsl = scaled_mm(template) @ np.vstack((grid, np.ones((1, grid.shape[1]))))

    # Reconstruct the original dense field's RAS pull independently, then
    # exercise the common production conversion on the actual input grids.
    original_dense = np.asanyarray(nib.load(args.reference / "gm_dense.nii.gz").dataobj)
    source_fsl = target_fsl[:3] + original_dense.reshape(-1, 3).T
    source_voxels = np.linalg.inv(scaled_mm(native)) @ np.vstack(
        (source_fsl, np.ones((1, source_fsl.shape[1]))))
    source_world = native.affine @ source_voxels
    target_world = template.affine @ np.vstack((grid, np.ones((1, grid.shape[1]))))
    original_ras = (source_world - target_world)[:3].T.reshape((*template.shape, 3)).astype(np.float32)
    pull = DenseWarp(original_ras, source=native, target=template)
    _, converted_dense, _ = _pull_ras_to_fsl_fields(
        pull, native, template, original_forward, device=torch.device("cpu"))
    converted_array = converted_dense.numpy()
    report["same_original_field_ras_to_fsl_roundtrip"] = {
        "maximum_component_error_mm": float(np.abs(converted_array - original_dense).max()),
        "component_rmse_mm": float(np.sqrt(np.mean(
            (converted_array.astype(np.float64) - original_dense) ** 2))),
        "moving_header_voxel_sizes": [float(v) for v in native.header.get_zooms()[:3]],
        "moving_affine_column_norms": np.linalg.norm(native.affine[:3, :3], axis=0).tolist()}
    converted_image = nib.Nifti1Image(converted_array, template.affine)
    converted_image.header["intent_code"] = 2006
    converted_result = apply(native, template, warp=converted_image,
                             interpolation="trilinear", warp_convention="relative", output_dtype="float")
    compare("same_original_dense_ras_roundtrip_original_header_applywarp_vs_original",
            np.asanyarray(converted_result.image.dataobj), original_warped)
    # The actual common chain reconstructs a NIfTI from data and affine,
    # which regenerates pixdim from affine column norms. Keep that pairing
    # when isolating its conversion/resampling, especially on oblique data.
    common_warped, common_qc = _common_applywarp(
        native_data, np.asanyarray(template.dataobj), native.affine,
        template.affine, converted_dense, device=torch.device("cpu"))
    compare("same_original_dense_ras_roundtrip_actual_common_chain_vs_original",
            common_warped, original_warped)
    report["original_dense_actual_common_chain_qc"] = common_qc

    def affine_sample(data, matrix):
        source_voxels = np.linalg.inv(scaled_mm(native)) @ np.linalg.inv(matrix) @ target_fsl
        return map_coordinates(data, source_voxels[:3], order=1, prefilter=False,
                               mode="constant", cval=0.).reshape(template.shape)

    affine_original = affine_sample(native_data, original_forward)
    affine_candidate = affine_sample(native_data, candidate_forward)
    compare("only_recorded_flirt_change_same_original_gm_affine_only",
            affine_candidate, affine_original)
    compare("only_frozen_fast_gm_change_same_original_affine",
            affine_sample(candidate_native_data, original_forward), affine_original)
    report["affine_sampling_scope"] = "independent full-template affine-only trilinear constant-zero diagnostic, not nonlinear sensitivity bound"
    report["original_flirt_matrix"] = original_forward.tolist()
    report["candidate_flirt_matrix_from_recorded_world_pull"] = candidate_forward.tolist()
    for group, directory in [("reference", args.reference), ("candidate", args.candidate)]:
        warped = np.asanyarray(nib.load(directory / "T1_GM_to_template_GM.nii.gz").dataobj)
        jacobian = np.asanyarray(nib.load(directory / "T1_GM_JAC_nl.nii.gz").dataobj)
        modulated = np.asanyarray(nib.load(directory / "T1_GM_to_template_GM_mod.nii.gz").dataobj)
        compare(group + "_saved_modulation_vs_float32_product", modulated, warped * jacobian)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value["template_brain"] for key, value in report["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
