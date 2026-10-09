"""两例真实MRI的固定源码亮区准备回归；outvol-only不执行表面放置。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(candidate, reference):
    delta = np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
    return {"different_voxels": int(np.count_nonzero(delta)),
            "maximum_absolute_error": float(delta.max()), "p99_absolute_error": float(np.percentile(delta, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-module", type=Path, required=True)
    parser.add_argument("--legacy-module", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.output_directory.exists():
        raise ValueError("positive threads and a new output directory required")
    args.output_directory.mkdir(parents=True)
    subject = args.subject.resolve()
    paths = [subject / "mri/brain.finalsurfs.mgz", subject / "mri/wm.mgz",
             subject / "mri/aseg.presurf.mgz", subject / "surf/lh.orig", subject / "surf/autodet.gw.stats.lh.dat"]
    brain_image, wm_image = (nib.load(str(p)) for p in paths[:2])
    if brain_image.shape != wm_image.shape or not np.array_equal(brain_image.affine, wm_image.affine):
        raise ValueError("MRI grids differ")
    brain, wm = np.asarray(brain_image.dataobj), np.asarray(wm_image.dataobj)
    stats = dict(line.split()[:2] for line in paths[-1].read_text().splitlines() if len(line.split()) >= 2)
    mid_gray = float(stats["MID_GRAY"])
    legacy = load(args.legacy_module, "legacy_placement_volume")
    candidate = load(args.candidate_module, "candidate_placement_volume")
    report = {"scope": "frozen_same_input_MRI_preparation_only_not_surface_placement_or_recon_all",
        "hostname": platform.node(), "code_commit": args.code_commit,
        "script_sha256": sha(__file__), "source_sha256": {"legacy": sha(args.legacy_module), "candidate": sha(args.candidate_module)},
        "native_sha256": sha(args.native_binary), "threads": args.threads,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "input_sha256": {str(p.relative_to(subject)): sha(p) for p in paths},
        "shape": list(brain.shape), "brain_dtype": str(brain.dtype), "wm_dtype": str(wm.dtype),
        "same_input_grid": True, "gpu": "not_used; existing_CPU_volume_prepare", "execution_status": "running", "modes": {}}

    def save():
        p = args.output_directory / "report.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(p)

    save()
    for surface in ("white", "pial"):
        row = {}
        report["modes"][surface] = row
        outputs = {}
        for name, module in (("legacy", legacy), ("candidate", candidate)):
            tick = time.perf_counter()
            prepared, labels = module.prepare_placement_volume(brain=brain, wm=wm,
                surface=surface, mid_gray=mid_gray, restore_255=True)
            row[name] = {"seconds_compute_preloaded_arrays": time.perf_counter()-tick,
                "label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in (0, 100, 130)}}
            # pial --outvol-only保存invol；CBV/PS是之后另建的输入，不能混称。
            if surface == "pial":
                expected = np.asarray(brain, np.uint8).copy()
                expected[(wm >= 5) & (expected > 110)] = 110
                expected[labels == 100] = np.uint8(np.floor(mid_gray+.5))
                row["reference_stage"] = "pial_invol_after_border_replacement_not_CBV_or_PS"
            else:
                expected = prepared
                row["reference_stage"] = "white_invol_equals_CBV_and_PS"
            outputs[name] = expected
        native_volume = args.output_directory / f"native-{surface}-invol.mgz"
        command = [str(args.native_binary), "--adgws-in", str(paths[-1]), "--wm", str(paths[1]),
            "--threads", str(args.threads), "--invol", str(paths[0]), "--lh", "--i", str(paths[3]),
            "--o", str(args.output_directory / f"unused-{surface}.diagnostic-surface"), "--"+surface,
            "--seg", str(paths[2]), "--restore-255", "--nsmooth", "0", "--no-rip",
            "--no-pin-medial-wall", "--outvol-only", str(native_volume)]
        if surface == "pial":
            # 原生参数校验要求 repulse-surf，MRI-only 在放置之前退出，
            # 因此此处同拓扑 orig 仅满足读取契约，不能视为 white 或完整 pial 输入。
            command.extend(("--repulse-surf", str(paths[3])))
            row["MRI_only_auxiliary_surface"] = {
                "role": "original_surface_loaded_as_unused_repulsion_coordinates",
                "sha256": sha(paths[3]),
                "not_a_white_or_pial_placement_reference": True,
            }
        tick = time.perf_counter()
        with (args.output_directory / f"native-{surface}.log").open("w") as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row["native"] = {"command": command, "returncode": run.returncode,
                         "seconds_including_process_read_write": time.perf_counter()-tick}
        save()
        if run.returncode != 0:
            report["execution_status"] = "failed_native"
            save()
            raise RuntimeError("native MRI-only reference failed; log preserved")
        ni = nib.load(str(native_volume))
        nv = np.asarray(ni.dataobj)
        same_grid = nv.shape == brain.shape and np.array_equal(ni.affine, brain_image.affine)
        row["native"].update(shape=list(nv.shape), dtype=str(nv.dtype), same_shape_affine=same_grid,
                              volume_sha256=sha(native_volume))
        if not same_grid:
            raise RuntimeError("native grid differs; no same-index comparison")
        for name, expected in outputs.items():
            row[name].update(same_dtype=expected.dtype == nv.dtype, error_against_native=error(expected, nv))
        row["changed_output_voxels"] = int(np.count_nonzero(outputs["legacy"] != outputs["candidate"]))
        save()
    report["input_sha256_after"] = {str(p.relative_to(subject)): sha(p) for p in paths}
    report["execution_status"] = "complete" if report["input_sha256_after"] == report["input_sha256"] else "failed_input_changed"
    save()


if __name__ == "__main__":
    main()
