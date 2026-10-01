#!/usr/bin/env python3
"""Check SynthStrip repeatability on a real 3D image in fresh processes.

Masks and signed distances stay in a private directory. The public report
contains scalar comparisons and hashes, without image paths or voxel arrays.
Current constructor policy is tested by default; --compare-autotune adds the
previous benchmark=True policy as a diagnostic intervention.
"""

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def flags(torch):
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def worker(args):
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.synthstrip import SynthStrip

    torch.set_num_threads(args.threads)
    selected = torch.device(args.device)
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
    image = nib.load(str(args.image))
    if image.ndim != 3:
        raise ValueError("--image must be a real 3D NIfTI image")
    report = {
        "policy": args.policy, "run": args.run,
        "input_file_sha256": sha256_file(args.image),
        "input_decoded_sha256": array_hash(np.asanyarray(image.dataobj)),
        "input_shape": list(image.shape),
        "input_voxel_sizes_mm": [float(v) for v in image.header.get_zooms()],
        "border_mm": args.border, "no_csf": args.no_csf,
    }
    started = time.perf_counter()
    extractor = SynthStrip(weights=str(args.weights), device=selected,
                           no_csf=args.no_csf, threads=args.threads)
    report["initialization_seconds"] = time.perf_counter() - started
    report["constructor_flags"] = flags(torch)
    if args.policy == "benchmark_on":
        torch.backends.cudnn.benchmark = True
    report["inference_flags"] = flags(torch)
    report["checkpoint_sha256"] = sha256_file(extractor.model_path)
    report["checkpoint_bytes"] = int(extractor.model_path.stat().st_size)
    digest = hashlib.sha256()
    for name, tensor in sorted(extractor.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(array_hash(tensor.detach().cpu().numpy()).encode())
    report["model_state_sha256"] = digest.hexdigest()
    if args.preceding_image is not None:
        # The fMRI volume pipeline uses one extractor for EPI followed by T1.
        report["preceding_image_sha256"] = sha256_file(args.preceding_image)
        extractor(nib.load(str(args.preceding_image)), border=args.border)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
    report["model_inputs"] = []
    report["model_predictions"] = []

    def record_input(_module, values):
        value = values[0].detach().cpu().numpy()
        report["model_inputs"].append({"sha256": array_hash(value),
                                       "shape": list(value.shape), "dtype": str(value.dtype)})

    def record_prediction(_module, _values, prediction):
        value = prediction.detach().cpu().numpy()
        report["model_predictions"].append({"sha256": array_hash(value),
                                            "shape": list(value.shape), "dtype": str(value.dtype)})

    extractor.model.register_forward_pre_hook(record_input)
    extractor.model.register_forward_hook(record_prediction)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
        torch.cuda.reset_peak_memory_stats(selected)
    started = time.perf_counter()
    result = extractor(image, border=args.border)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    report["call_seconds_including_hash_hooks"] = time.perf_counter() - started
    report["final_flags"] = flags(torch)
    args.worker_output.mkdir(parents=True, exist_ok=False, mode=0o700)
    mask_path = args.worker_output / "mask.private.nii.gz"
    distance_path = args.worker_output / "distance.private.nii.gz"
    nib.save(result.mask, str(mask_path))
    nib.save(result.distance, str(distance_path))
    mask = np.asarray(result.mask.dataobj, dtype=np.uint8)
    distance = np.asarray(result.distance.dataobj, dtype=np.float32)
    report["mask"] = {"brain_voxels": int(np.count_nonzero(mask)),
                      "decoded_sha256": array_hash(mask), "file_sha256": sha256_file(mask_path)}
    report["distance"] = {"decoded_sha256": array_hash(distance),
                          "file_sha256": sha256_file(distance_path)}
    report["sources_sha256"] = {
        name: sha256_file(importlib.import_module(name).__file__)
        for name in ("fnit.synthstrip.pipeline", "fnit.synthstrip.model",
                     "fnit.synthstrip.geometry", "fnit._nib", "fnit.weights")
    }
    report["software"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                          "numpy": np.__version__, "nibabel": nib.__version__,
                          "cudnn": torch.backends.cudnn.version(), "device": args.device}
    if selected.type == "cuda":
        report["software"]["gpu"] = torch.cuda.get_device_name(selected)
        report["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(selected))
        report["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(selected))
    write_json(args.worker_output / "run.private.json", report)


def compare_masks(left_path, right_path, distance_path, border):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion

    left_image, right_image = nib.load(str(left_path)), nib.load(str(right_path))
    if left_image.shape != right_image.shape or not np.array_equal(
            left_image.affine, right_image.affine):
        raise ValueError("compared masks must have identical shape and affine")
    left, right = np.asarray(left_image.dataobj) != 0, np.asarray(right_image.dataobj) != 0
    change = left != right
    count = int(change.sum())
    a, b = int(left.sum()), int(right.sum())
    boundary = (left & ~binary_erosion(left)) | (right & ~binary_erosion(right))
    nearby = binary_dilation(boundary)
    result = {
        "left_brain_voxels": a, "right_brain_voxels": b,
        "dice": float(2 * (left & right).sum() / (a + b)) if a + b else 1.0,
        "exact_equal": not count, "changed_voxels": count,
        "left_only_voxels": int((left & ~right).sum()),
        "right_only_voxels": int((right & ~left).sum()),
        "changed_within_one_voxel_of_either_boundary": int((change & nearby).sum()),
    }
    if count:
        distance = np.asarray(nib.load(str(distance_path)).dataobj, dtype=np.float32)
        gap = np.abs(distance[change].astype(np.float64) - border)
        result["distance_to_border_on_changed_voxels_mm"] = {
            "min": float(gap.min()), "median": float(np.median(gap)),
            "p95": float(np.percentile(gap, 95)), "max": float(gap.max()),
        }
    return result


def compare_distances(left_path, right_path):
    import nibabel as nib
    import numpy as np
    left_image, right_image = nib.load(str(left_path)), nib.load(str(right_path))
    if left_image.shape != right_image.shape or not np.array_equal(
            left_image.affine, right_image.affine):
        raise ValueError("compared signed distances must have identical shape and affine")
    left, right = np.asarray(left_image.dataobj, dtype=np.float32), np.asarray(right_image.dataobj, dtype=np.float32)
    error = left.astype(np.float64) - right.astype(np.float64)
    return {"exact_equal": bool(np.array_equal(left, right)),
            "changed_voxels": int(np.count_nonzero(left != right)),
            "rmse_mm": float(np.sqrt(np.mean(error * error))),
            "max_abs_mm": float(np.abs(error).max())}


def main(args):
    if args.private_output.exists() or args.report.exists():
        raise FileExistsError("use a fresh private-output directory and report file")
    args.private_output.mkdir(parents=True, mode=0o700)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = {"image": sha256_file(args.image)}
    if args.preceding_image is not None:
        input_hashes["preceding_image"] = sha256_file(args.preceding_image)
    if args.reference_mask is not None:
        input_hashes["reference_mask"] = sha256_file(args.reference_mask)
    policies = ("default", "benchmark_on") if args.compare_autotune else ("default",)
    records = {}
    for policy in policies:
        for run in range(1, args.repeats + 1):
            key = f"{policy}_{run}"
            destination = args.private_output / key
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--policy", policy, "--run", str(run),
                       "--worker-output", str(destination), "--image", str(args.image),
                       "--weights", str(args.weights), "--device", args.device,
                       "--threads", str(args.threads), "--border", str(args.border)]
            if args.no_csf:
                command.append("--no-csf")
            if args.preceding_image is not None:
                command.extend(["--preceding-image", str(args.preceding_image)])
            started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True)
            (args.private_output / f"{key}.stdout.private.log").write_text(completed.stdout)
            (args.private_output / f"{key}.stderr.private.log").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{key} failed; details remain in its private stderr log")
            records[key] = json.loads((destination / "run.private.json").read_text())
            records[key]["fresh_process_seconds"] = time.perf_counter() - started
    comparisons = {}
    for left, right in itertools.combinations(records, 2):
        a, b = args.private_output / left, args.private_output / right
        comparisons[f"{left}_vs_{right}"] = {
            "mask": compare_masks(a / "mask.private.nii.gz", b / "mask.private.nii.gz",
                                  a / "distance.private.nii.gz", args.border),
            "distance": compare_distances(a / "distance.private.nii.gz", b / "distance.private.nii.gz"),
            "model_state_hash_equal": records[left]["model_state_sha256"] == records[right]["model_state_sha256"],
            "model_input_hashes_equal": records[left]["model_inputs"] == records[right]["model_inputs"],
            "checkpoint_hash_equal": records[left]["checkpoint_sha256"] == records[right]["checkpoint_sha256"],
        }
    reference_comparisons = {}
    if args.reference_mask is not None:
        for key in records:
            directory = args.private_output / key
            reference_comparisons[key] = compare_masks(
                directory / "mask.private.nii.gz", args.reference_mask,
                directory / "distance.private.nii.gz", args.border)
    inputs_unchanged = sha256_file(args.image) == input_hashes["image"]
    if args.preceding_image is not None:
        inputs_unchanged &= sha256_file(args.preceding_image) == input_hashes["preceding_image"]
    if args.reference_mask is not None:
        inputs_unchanged &= sha256_file(args.reference_mask) == input_hashes["reference_mask"]
    report = {
        "scope": "One real 3D image, fresh-process repeatability; no simulated benchmark",
        "driver_sha256": sha256_file(__file__), "inputs_sha256": input_hashes,
        "inputs_unchanged": bool(inputs_unchanged), "policies": list(policies),
        "repeats_per_policy": args.repeats, "runs": records,
        "pair_comparisons": comparisons, "reference_mask_comparisons": reference_comparisons,
        "timing_scope": "Fresh-process time includes startup/load/inference/writes; call time includes tensor hashing hooks. These are diagnostic observations on a shared GPU, not model throughput claims.",
        "privacy": "Public report contains scalars and hashes only. Images, signed distances and logs remain in the private output directory.",
    }
    write_json(args.report, report)
    default_pairs = [value for key, value in comparisons.items()
                     if key.startswith("default_") and "_vs_default_" in key]
    print(json.dumps({"completed_processes": len(records), "inputs_unchanged": bool(inputs_unchanged),
                      "default_mask_exact_across_processes": all(value["mask"]["exact_equal"] for value in default_pairs),
                      "default_distance_exact_across_processes": all(value["distance"]["exact_equal"] for value in default_pairs)}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Real 3D NIfTI input")
    p.add_argument("--weights", type=Path, required=True, help="Local official SynthStrip checkpoint or directory")
    p.add_argument("--preceding-image", type=Path, help="Optional EPI/SBRef extracted first with the same model")
    p.add_argument("--reference-mask", type=Path, help="Optional saved same-grid mask for comparison")
    p.add_argument("--private-output", type=Path, help="Fresh directory for images and logs")
    p.add_argument("--report", type=Path, help="Fresh public JSON containing scalars and hashes")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--border", type=float, default=1.0, help="Mask border threshold, mm")
    p.add_argument("--no-csf", action="store_true", help="Use the official no-CSF model")
    p.add_argument("--repeats", type=int, default=2, help="Fresh processes per policy, minimum 2")
    p.add_argument("--compare-autotune", action="store_true", help="Also test the former benchmark=True policy")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--policy", choices=("default", "benchmark_on"), help=argparse.SUPPRESS)
    p.add_argument("--run", type=int, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.threads < 1 or args.repeats < 2:
        p.error("threads must be positive and repeats must be at least two")
    if args.worker:
        if args.worker_output is None or args.policy is None:
            p.error("worker arguments missing")
        worker(args)
    else:
        if args.private_output is None or args.report is None:
            p.error("--private-output and --report are required")
        main(args)
