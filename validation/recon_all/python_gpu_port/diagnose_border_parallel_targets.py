"""录取后的真实四轮状态：复用同一 FNIT 边界函数体测串行/四线程 prange。

仅生成私有诊断副本，fastmath=False；目标有序平均与表面更新不并行。
此处计时包括函数包装与数据转换，但不代表完整放置或 recon-all 提速。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import nibabel as nib
import numba
import numpy as np

from diagnose_pial_first_difference import compare
from diagnose_preaparc_target_passes import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-directory", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--border-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    admission_path = args.checkpoints_directory / "report.json"
    admission = json.loads(admission_path.read_text())
    if admission["status"] != "complete" or admission["probe_admission"]["coordinates"]["different_elements"]:
        raise RuntimeError("full native probe admission not complete")
    args.output_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    numba.set_num_threads(args.threads)
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_volume import prepare_placement_volume
    h = args.hemisphere
    brain = nib.load(str(args.subject / "mri/brain.finalsurfs.mgz"))
    seg = np.asarray(nib.load(str(args.subject / "mri/aseg.presurf.mgz")).dataobj)
    wm = np.asarray(nib.load(str(args.subject / "mri/wm.mgz")).dataobj)
    stats_path = args.checkpoints_directory / f"native_probe/surf/autodet.gw.stats.{h}.dat"
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines() if len(line.split()) >= 2)
    volume, _ = prepare_placement_volume(np.asarray(brain.dataobj), wm,
                                         surface="white", mid_gray=float(stats["MID_GRAY"]))
    vertices, _, meta = nib.freesurfer.read_geometry(str(args.subject / f"surf/{h}.orig"), read_metadata=True)
    affine = surface_ras_to_voxel(brain.header, meta)
    thresholds = np.array([float(stats[f"white_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    original = state(args.checkpoints_directory / "native_target.pass00.entry", len(vertices))["floats"][:, :3].copy()
    source = args.border_source.read_text()
    decorator = "@njit(cache=True)\ndef _search("
    vertex_loop = "for v in range(nvertices):"
    if source.count(decorator) != 1 or source.count(vertex_loop) != 1:
        raise RuntimeError("canonical border function body changed")
    parallel_source = source.replace("from numba import njit", "from numba import njit, prange")
    parallel_source = parallel_source.replace(decorator, "@njit(cache=True, parallel=True, fastmath=False)\ndef _search(")
    parallel_source = parallel_source.replace(vertex_loop, "for v in prange(nvertices):")
    modules = {}
    for name, text in (("serial", source), ("parallel", parallel_source)):
        path = args.output_directory / f"border_{name}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(f"border_parallel_diagnostic_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    report = dict(scope="frozen_real_four_pass_border_targets_only_not_full_surface_speed",
        canonical_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        diagnostic_parallel_source_sha256=hashlib.sha256(parallel_source.encode()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        admitted_native_receipt_sha256=hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        hostname=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads=args.threads, numba_threads=numba.get_num_threads(), fastmath=False, passes=[], status="running")
    for i in range(4):
        entry = state(args.checkpoints_directory / f"native_target.pass{i:02d}.entry", len(vertices))
        before = state(args.checkpoints_directory / f"native_target.pass{i:02d}.before_average", len(vertices))
        active = before["flags"][:, 0] == 0
        function_args = (volume, seg, before["floats"][:, :3], before["floats"][:, 3:6],
                         original, ~active, entry["floats"][:, 6], affine, thresholds)
        options = dict(hemisphere=h, surface="white", sigma=2.0 / (1 << i))
        row = dict(pass_index=i, warmup={}, ABBA=[])
        if i == 0:
            for name in ("serial", "parallel"):
                tick = perf_counter()
                result = modules[name].compute_border_values_first_pass(*function_args, **options)
                row["warmup"][name] = dict(seconds_including_possible_JIT=perf_counter()-tick)
        serial_result = None
        for name in ("serial", "parallel", "parallel", "serial"):
            tick = perf_counter()
            result = modules[name].compute_border_values_first_pass(*function_args, **options)
            item = dict(backend=name, seconds=perf_counter()-tick)
            if serial_result is None:
                serial_result = tuple(value.copy() for value in result)
            item["vs_serial_all_vertices"] = {key: compare(a, b) for key, a, b in zip(
                ("values", "distance", "mean", "target_coordinates", "marked", "sigma"), serial_result, result)}
            row["ABBA"].append(item)
        report["passes"].append(row)
        (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    report["all_fields_exact"] = all(field["different_elements"] == 0 for p in report["passes"]
        for run in p["ABBA"] for field in run["vs_serial_all_vertices"].values())
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
