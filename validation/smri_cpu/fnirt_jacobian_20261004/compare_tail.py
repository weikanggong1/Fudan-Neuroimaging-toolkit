"""Replay only old/new FastVBM tails with one captured complete real FNIRT state."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess

import nibabel as nib
import numpy as np
import torch

from fnit._nib import new_image
from fnit._transforms import DenseWarp
from fnit.fast_vbm import registration as current
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a, b, selected):
    a, b = a[selected].astype(np.float64), b[selected].astype(np.float64)
    error = a - b
    scale = float(np.percentile(b, 99) - np.percentile(b, 1))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"values": int(a.size), "different_values": int(np.count_nonzero(error)),
            "mae": float(np.abs(error).mean()), "rmse": rmse,
            "nrmse": rmse / scale if scale else None,
            "p99_absolute_error": float(np.percentile(np.abs(error), 99)),
            "maximum_absolute_error": float(np.abs(error).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["candidate", "reference", "template", "reference-mask", "capture",
                 "legacy-registration", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    gpu = None
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        torch.cuda.set_per_process_memory_fraction(min(1., 20e9 / properties.total_memory), args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        gpu = {"device_uuid": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"], text=True).strip(),
               "name": properties.name, "total_memory": properties.total_memory,
               "allocator_budget_bytes": 20000000000}
    spec = importlib.util.spec_from_file_location(
        "fnit.fast_vbm._legacy_jacobian_registration", args.legacy_registration)
    legacy = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    moving = nib.load(args.candidate / "T1_brain_pve_1.nii.gz")
    fixed = nib.load(args.template)
    reference_mask = nib.load(args.reference_mask)
    capture = json.loads((args.capture / "capture.public.json").read_text())
    pull = DenseWarp(np.asanyarray(nib.load(args.capture / "pull.nii.gz").dataobj),
                     source=moving, target=fixed)
    analytic = np.asanyarray(nib.load(args.capture / "analytic.nii.gz").dataobj)
    initial = np.array(capture["initial_forward_world"])
    flirt = np.array(capture["initial_flirt"])
    model = TorchFNIRT(device=args.device, config=GMFNIRTConfig())
    results = {}
    for name, module in [("baseline", legacy), ("candidate", current)]:
        def prepare(*values, **kwargs):
            moving_data, fixed_data, moving_affine, fixed_affine = values[2:6]
            mask_image, mask_data, mask_source = module._reference_mask(reference_mask, fixed, fixed_data)
            return module._PreparedRegistration(
                moving_data, fixed_data, moving_affine, fixed_affine, flirt,
                initial, np.linalg.inv(initial), initial, mask_image, mask_source,
                {"scope": "recorded affine replay; no FLIRT fit"},
                module._pre_nonlinear_signature(moving_data, fixed_data, moving_affine,
                                               fixed_affine, flirt, mask_data))
        module._prepare_registration = prepare
        module._estimate_nonlinear = lambda *unused: (pull, capture["fit_qc"], analytic)
        results[name] = module._register_gm(moving, fixed, device=args.device,
                                          reference_mask=reference_mask,
                                          registration_backend="fnirt", deform_model=model)
        out = args.output / name
        out.mkdir()
        for field in ["warped_gm", "jacobian", "modulated_gm"]:
            nib.save(getattr(results[name], field), out / (field + ".nii.gz"))
        (out / "qc.public.json").write_text(json.dumps(results[name].qc, indent=2) + "\n")
    regions = {"whole_grid": np.ones(fixed.shape, dtype=bool), "template_brain":
               (np.asanyarray(reference_mask.dataobj) > 0) & (np.asanyarray(fixed.dataobj) > 0)}
    names = {"warped_gm": "T1_GM_to_template_GM.nii.gz", "jacobian": "T1_GM_JAC_nl.nii.gz",
             "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz"}
    comparisons = {}
    old, new = results["baseline"], results["candidate"]
    for field, official_name in names.items():
        old_image, new_image_value = getattr(old, field), getattr(new, field)
        a, b = np.asanyarray(new_image_value.dataobj), np.asanyarray(old_image.dataobj)
        official = np.asanyarray(nib.load(args.reference / official_name).dataobj)
        comparisons[field] = {
            "arrays_exact": bool(np.array_equal(a, b)),
            "header_exact": bool(np.array_equal(old_image.header.binaryblock, new_image_value.header.binaryblock)),
            "affine_exact": bool(np.array_equal(old_image.affine, new_image_value.affine)),
            "old_vs_official": {name: metrics(b, official, mask) for name, mask in regions.items()},
            "new_vs_official": {name: metrics(a, official, mask) for name, mask in regions.items()},
            "new_vs_old": {name: metrics(a, b, mask) for name, mask in regions.items()}}
    old_qc, new_qc = old.qc, new.qc
    changed = [key for key in set(old_qc) | set(new_qc) if old_qc.get(key) != new_qc.get(key)]
    report = {"status": "complete", "scope": "old/new whole real GM tail replay; one captured CPU FNIRT fit; no pipeline benchmark",
              "host": socket.gethostname(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(), "device": args.device, "gpu": gpu,
              "source_identity": {"current_registration_sha256": digest(current.__file__),
                                  "legacy_registration_sha256": digest(args.legacy_registration),
                                  "capture_report_sha256": digest(args.capture / "capture.public.json"),
                                  "analytic_sha256": digest(args.capture / "analytic.nii.gz"),
                                  "pull_sha256": digest(args.capture / "pull.nii.gz"),
                                  "coefficients_sha256": digest(args.capture / "coefficients.nii.gz")},
              "comparisons": comparisons, "all_qc_exact": old_qc == new_qc,
              "changed_qc_keys": sorted(changed),
              "complete_estimator_qc_exact": old_qc["nonlinear_estimator_qc"] == new_qc["nonlinear_estimator_qc"] == capture["fit_qc"],
              "new_cpu_jacobian_equals_captured_analytic": bool(np.array_equal(new.jacobian.dataobj, analytic)),
              "old_cpu_tail_matches_capture": {field: bool(np.array_equal(
                  getattr(old, field).dataobj, nib.load(args.capture / filename).dataobj))
                  for field, filename in [("warped_gm", "old_warped.nii.gz"),
                                          ("jacobian", "old_dense_jacobian.nii.gz"),
                                          ("modulated_gm", "old_modulated.nii.gz")]} if args.device == "cpu" else None}
    if gpu:
        torch.cuda.synchronize(args.device)
        gpu.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)})
    (args.output / "comparison.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"device": args.device, "qc_exact": report["all_qc_exact"],
                      "complete_estimator_qc_exact": report["complete_estimator_qc_exact"],
                      "arrays_exact": {k: v["arrays_exact"] for k, v in comparisons.items()},
                      "official_brain_rmse": {k: [v["old_vs_official"]["template_brain"]["rmse"],
                                                 v["new_vs_official"]["template_brain"]["rmse"]]
                                              for k, v in comparisons.items()}}))


if __name__ == "__main__":
    main()
