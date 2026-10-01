"""Separate PVE estimation, BBR estimation and FLIRT prefiltering on real maps.

--case-json contains private paths to an EPI reference, both EPI brain masks,
native T1-to-EPI FSL matrix, candidate T1-to-EPI world matrix, and WM/CSF maps.
It can provide already validated native/native sampling outputs for reuse.
The four PVE-source x BBR-source combinations use the same common EPI mask.
Neither FAST nor affine/nonlinear optimization is called. Images and matrices
remain in --private-output; --report-out contains anonymous metrics and hashes.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, directory):
    path = Path(value)
    return path if path.is_absolute() else directory / path


def agreement(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    delta = first - second
    if first.size == 0:
        raise ValueError("Empty numerical comparison region")
    centred_first, centred_second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(centred_first) * np.linalg.norm(centred_second)
    return {
        "values": int(first.size),
        "pearson_r": float(np.dot(centred_first, centred_second) / denominator) if denominator else None,
        "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta**2))),
        "maximum_absolute_difference": float(np.abs(delta).max()),
        "mean_signed_difference": float(delta.mean()),
    }


def mask_agreement(candidate, reference):
    intersection = int((candidate & reference).sum())
    candidate_count, reference_count = int(candidate.sum()), int(reference.sum())
    total = candidate_count + reference_count
    return {
        "candidate_voxels": candidate_count, "reference_voxels": reference_count,
        "dice": 2 * intersection / total if total else 1.,
        "gained_voxels": int((candidate & ~reference).sum()),
        "lost_voxels": int((reference & ~candidate).sum()),
    }


def load_on_grid(path, reference):
    image = nib.load(path)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("A probability/mask image does not match the expected grid")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite probability or mask values")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from fnit.flirt import TorchFLIRT, flirt_to_world_affine, world_to_flirt_affine
    from fnit.flirt import core as flirt_core
    from fnit.fmri import normalization

    if "_coordinates_from_fsl_coefficients" not in inspect.getsource(flirt_core._resample_output):
        raise ValueError("Use the corrected FLIRT output sampler that excludes TF32 coordinate matrix products")
    case = json.loads(args.case_json.read_text())
    directory = args.case_json.parent
    paths = {key: resolve_path(case[key], directory) for key in (
        "reference", "native_epi_mask", "candidate_epi_mask",
        "native_t1_to_epi_fsl", "candidate_t1_to_epi_world",
    )}
    args.private_output.mkdir(parents=True, exist_ok=True)
    reference = nib.load(paths["reference"])
    if reference.ndim != 3:
        raise ValueError("The EPI reference must be 3D")
    native_epi_mask = load_on_grid(paths["native_epi_mask"], reference) > 0
    candidate_epi_mask = load_on_grid(paths["candidate_epi_mask"], reference) > 0
    common = native_epi_mask & candidate_epi_mask
    if not common.any():
        raise ValueError("The common EPI brain mask is empty")
    common_path = args.private_output / "common_epi_mask.nii.gz"
    nib.save(nib.Nifti1Image(common.astype(np.uint8), reference.affine), common_path)
    native_fsl = np.loadtxt(paths["native_t1_to_epi_fsl"])
    candidate_world = np.loadtxt(paths["candidate_t1_to_epi_world"])
    sampler = TorchFLIRT(device=args.device)
    tissues = {}
    hashes = {key: sha256(value) for key, value in paths.items()}
    private_outputs = {}
    for tissue_name, specification in case["tissues"].items():
        pve_paths = {origin: resolve_path(specification[origin + "_pve"], directory)
                     for origin in ("native", "candidate")}
        pve_images = {origin: nib.load(path) for origin, path in pve_paths.items()}
        source = pve_images["native"]
        if source.ndim != 3:
            raise ValueError("PVE inputs must be 3D")
        source_pve = {origin: load_on_grid(path, source) for origin, path in pve_paths.items()}
        source_mask = np.ones(source.shape, dtype=bool)
        for key in ("native_t1_mask", "candidate_t1_mask"):
            if key in case:
                path = resolve_path(case[key], directory)
                source_mask &= load_on_grid(path, source) > 0
                hashes[key] = sha256(path)
        native_world = flirt_to_world_affine(
            native_fsl, source.affine, reference.affine, source.shape, reference.shape,
            source.header.get_zooms()[:3], reference.header.get_zooms()[:3],
        )
        world_matrices = {"native": native_world, "candidate": candidate_world}
        original = load_on_grid(resolve_path(specification["native_resampled"], directory), reference)
        original_mask = (original >= .8) & common
        results, arrays = {}, {}
        for method in ("plain_trilinear", "flirt_applyxfm"):
            arrays[method], results[method] = {}, {}
            for pve_origin in ("native", "candidate"):
                for bbr_origin in ("native", "candidate"):
                    label = pve_origin + "_pve__" + bbr_origin + "_bbr"
                    private = args.private_output / (tissue_name + "_" + method + "_" + label + ".nii.gz")
                    reuse_name = "reuse_" + method + "_native_native"
                    if pve_origin == bbr_origin == "native" and reuse_name in specification:
                        reused = resolve_path(specification[reuse_name], directory)
                        values = load_on_grid(reused, reference)
                        nib.save(nib.Nifti1Image(values, reference.affine, reference.header), private)
                        hashes[tissue_name + "_reused_" + method] = sha256(reused)
                    elif method == "plain_trilinear":
                        normalization.resample_world(
                            pve_paths[pve_origin], paths["reference"],
                            np.linalg.inv(world_matrices[bbr_origin]), private,
                            interpolation="linear", device=args.device,
                        )
                        values = load_on_grid(private, reference)
                    else:
                        moving = pve_images[pve_origin]
                        matrix = native_fsl if bbr_origin == "native" else world_to_flirt_affine(
                            world_matrices[bbr_origin], moving.affine, reference.affine,
                            moving.shape, reference.shape, moving.header.get_zooms()[:3],
                            reference.header.get_zooms()[:3],
                        )
                        result = sampler.applyxfm(moving, reference, init=matrix)
                        nib.save(result.moved, private)
                        values = load_on_grid(private, reference)
                    arrays[method][label] = values
                    binary = (values >= .8) & common
                    results[method][label] = {
                        "threshold_voxels_common": int(binary.sum()),
                        "threshold_voxels_native_brain": int(((values >= .8) & native_epi_mask).sum()),
                        "threshold_voxels_candidate_brain": int(((values >= .8) & candidate_epi_mask).sum()),
                        "mask_vs_existing_fsl": mask_agreement(binary, original_mask),
                        "pve_vs_existing_fsl": agreement(values[common], original[common]),
                        "voxels_near_threshold_0p78_to_0p82": int(((values >= .78) & (values <= .82) & common).sum()),
                        "sha256": sha256(private),
                    }
                    private_outputs[tissue_name + "_" + method + "_" + label] = str(private)
            baseline = arrays[method]["native_pve__native_bbr"]
            baseline_mask = (baseline >= .8) & common
            for label, values in arrays[method].items():
                results[method][label]["pve_vs_same_method_native_native"] = agreement(values[common], baseline[common])
                results[method][label]["mask_vs_same_method_native_native"] = mask_agreement((values >= .8) & common, baseline_mask)
        blur_effect = {}
        for label in arrays["plain_trilinear"]:
            plain, blurred = arrays["plain_trilinear"][label], arrays["flirt_applyxfm"][label]
            blur_effect[label] = {
                "pve_difference": agreement(blurred[common], plain[common]),
                "mask_difference": mask_agreement((blurred >= .8) & common, (plain >= .8) & common),
            }
        stored_masks = {}
        for origin, key in (("native", "native_mask"), ("candidate", "candidate_mask")):
            if key in specification:
                path = resolve_path(specification[key], directory)
                stored_masks[origin] = load_on_grid(path, reference) > 0
                hashes[tissue_name + "_" + key] = sha256(path)
        tissues[tissue_name] = {
            "input_pve_common_t1": agreement(source_pve["candidate"][source_mask], source_pve["native"][source_mask]),
            "existing_fsl_mask_voxels_common": int(original_mask.sum()),
            "variants": results, "blur_effect": blur_effect,
            "stored_native_mask_matches_existing_fsl_threshold": bool(np.array_equal(stored_masks["native"] & common, original_mask)) if "native" in stored_masks else None,
            "stored_candidate_mask_vs_original_plain_sampling": mask_agreement(
                (arrays["plain_trilinear"]["candidate_pve__candidate_bbr"] >= .8) & common,
                stored_masks["candidate"] & common,
            ) if "candidate" in stored_masks else None,
        }
        for key in ("native_pve", "candidate_pve", "native_resampled"):
            hashes[tissue_name + "_" + key] = sha256(resolve_path(specification[key], directory))
    report = {
        "schema_version": 1, "source_revision": args.source_revision,
        "estimation_source_revisions": case.get("estimation_source_revisions", {}),
        "scope": "Real saved WM/CSF PVE x BBR 2x2 cross-control; no FAST, FLIRT optimization or complete pipeline rerun.",
        "common_epi_voxels": int(common.sum()), "native_epi_voxels": int(native_epi_mask.sum()),
        "candidate_epi_voxels": int(candidate_epi_mask.sum()), "threshold": .8,
        "sampling_methods": {
            "plain_trilinear": "FNIT resample_world linear with no prefilter",
            "flirt_applyxfm": "FNIT TorchFLIRT.applyxfm with official default downsampling prefilter and ordered FP32 geometry",
        },
        "tissues": tissues, "sha256": {"inputs": hashes, "common_epi_mask": sha256(common_path),
            "script": sha256(Path(__file__)), "implementation": {
                "src/fnit/flirt/core.py": sha256(Path(flirt_core.__file__)),
                "src/fnit/fmri/normalization.py": sha256(Path(normalization.__file__)),
            }},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "nibabel": nib.__version__, "torch": torch.__version__, "device": args.device},
        "limits": ["Crosses change one estimator at a time on a common grid/mask; threshold counts are nonlinear and effects are not additive.",
                   "The original-input numerical FSL oracle is verified separately; reused outputs retain their file hashes."],
        "privacy": "Anonymous scalar metrics and hashes only; source paths, arrays and commands remain private.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.private_output / "manifest.private.json").write_text(json.dumps(private_outputs, indent=2) + "\n")
    print(json.dumps({"completed": "tissue_cross_control", "common_epi_voxels": int(common.sum())}), flush=True)


if __name__ == "__main__":
    main()
