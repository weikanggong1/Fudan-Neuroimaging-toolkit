"""Compare SynthStrip's fixed-input geometry and sampling to official surfa.

The private manifest contains ``weights``, ``original_program`` and ``cases``.
Each case supplies ``input`` and ``original_mask`` paths. Image arrays, individual
masks and distances remain in ``--private-output``; the report contains hashes
and anonymous summary metrics. This benchmark alone imports surfa, which is
provided by the original FreeSurfer environment and is not a FNIT dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthstrip import pipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_comparison(candidate, reference):
    left = np.asarray(candidate)
    right = np.asarray(reference)
    if left.shape != right.shape:
        return {"shape_equal": False}
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {
        "shape_equal": True,
        "arrays_equal": bool(np.array_equal(left, right)),
        "changed_voxels": int(np.count_nonzero(left != right)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def run(manifest, private_output, device):
    import surfa as sf

    private_output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Fixed real inputs; official geometry, unchanged checkpoint, independent sampling",
        "driver_sha256": sha256(__file__),
        "source_sha256": sha256(pipeline.__file__),
        "original_program_sha256": sha256(manifest["original_program"]),
        "weights_sha256": sha256(manifest["weights"]),
        "environment": {"torch": torch.__version__, "nibabel": nib.__version__,
                        "surfa": sf.__version__, "device": str(device)},
        "cases": {},
        "notes": [
            "The network model and checkpoint are identical; official geometry uses validation-only surfa.",
            "Saved original masks come from a separate original-program run; near-threshold GPU roundoff can change binary labels.",
            "Shared GPU timing includes control comparisons and is not a SynthStrip speed benchmark.",
        ],
    }
    for label, case in manifest["cases"].items():
        image = nib.load(case["input"])
        original = sf.load_volume(case["input"])
        reference = original.conform(voxsize=1., dtype="float32", method="nearest", orientation="LIA")
        candidate = pipeline._conform_lia_1mm(image, device=device)
        result = {
            "input_sha256": sha256(case["input"]),
            "original_mask_sha256": sha256(case["original_mask"]),
            "input_shape": list(image.shape),
            "conform": array_comparison(candidate.dataobj, reference.data),
            "conform_affine_max_abs_mm": float(np.max(np.abs(candidate.affine - reference.geom.vox2world.matrix))),
        }
        reference = reference.crop_to_bbox()
        candidate = pipeline._crop_nonzero(candidate)
        target_shape = np.clip(np.ceil(np.asarray(reference.shape[:3]) / 64).astype(int) * 64, 192, 320)
        reference = reference.reshape(target_shape)
        candidate = pipeline._reshape_center(candidate, target_shape)
        data = np.asarray(candidate.dataobj)
        data = data - data.min()
        data = (data / np.percentile(data, 99)).clip(0, 1)
        reference -= reference.min()
        reference = (reference / reference.percentile(99)).clip(0, 1)
        result["model_input"] = array_comparison(data, reference.data)
        result["model_shape"] = list(data.shape)
        result["model_affine_max_abs_mm"] = float(np.max(np.abs(candidate.affine - reference.geom.vox2world.matrix)))
        start = time.perf_counter()
        model = pipeline.SynthStrip(manifest["weights"], device=device, threads=8)
        with torch.no_grad():
            tensor = torch.as_tensor(np.ascontiguousarray(data[None, None]), device=device)
            distance = model.model(tensor).squeeze().cpu().numpy()
        original_sdt = reference.new(distance).resample_like(original, fill=100)
        candidate_sdt = pipeline._resample_affine(distance, candidate.affine, image.shape, image.affine,
                                                 device=device, fill=100)
        result["same_prediction_distance"] = array_comparison(candidate_sdt, original_sdt.data)
        original_mask = (original_sdt < 1).connected_component_mask(k=1, fill=True).data
        candidate_mask = pipeline._largest_filled_component(candidate_sdt < 1)
        result["same_prediction_mask_equal"] = bool(np.array_equal(candidate_mask, original_mask))
        saved_mask = nib.load(case["original_mask"]).get_fdata() > 0
        denominator = candidate_mask.sum() + saved_mask.sum()
        result["saved_original_mask"] = {
            "dice": float(2 * np.count_nonzero(candidate_mask & saved_mask) / denominator),
            "changed_voxels": int(np.count_nonzero((candidate_mask > 0) != saved_mask)),
            "candidate_voxels": int(candidate_mask.sum()), "reference_voxels": int(saved_mask.sum()),
        }
        result["control_wall_seconds"] = time.perf_counter() - start
        nib.save(nib.Nifti1Image(candidate_mask, image.affine, image.header),
                 private_output / f"{label}_mask.nii.gz")
        report["cases"][label] = result
        del model, tensor, distance
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    report["environment"]["cuda_matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    report["environment"]["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = run(manifest, args.private_output, args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
