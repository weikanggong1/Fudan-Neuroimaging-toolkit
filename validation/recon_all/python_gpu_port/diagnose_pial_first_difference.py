"""冻结同输入 pial 的透明逐项诊断；仅 benchmark，不修改生产算法。

原生诊断副本必须与未插桩程序的完整最终几何一致。Python 原函数照常
执行，局部包装器只保存返回值；各步坐标、目标和梯度保留在私有目录。
诊断墙钟包含额外写出，不用于声称算法提速。全部路径均为具名参数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    if a.shape != b.shape:
        return {"same_shape": False}
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"same_shape": True, "different_elements": int(np.count_nonzero(a != b)),
            "max_absolute": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta * delta)))}


def native_state(path, count):
    dtype = np.dtype([("floats", "<f4", (9,)), ("flags", "<i4", (3,))])
    result = np.fromfile(path, dtype=dtype)
    if len(result) != count:
        raise ValueError("native state length does not match frozen mesh")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--native-reference-surface", type=Path, required=True)
    parser.add_argument("--assets-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--existing-native-diagnostic-directory", type=Path,
                        help="仅复用带相同输入/程序哈希且完整原生几何通过的诊断检查点")
    args = parser.parse_args()
    if not (args.candidate_directory / "place_pial_python.py").is_file():
        raise FileNotFoundError("candidate-directory must directly contain place_pial_python.py")
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    os.chmod(args.output_directory, 0o700)
    hemi, out = args.hemisphere, args.output_directory
    inputs = [Path(f"surf/{hemi}.white"), Path(f"surf/autodet.gw.stats.{hemi}.dat"),
              *[Path(f"label/{hemi}.{name}") for name in
                ("cortex.label", "cortex+hipamyg.label", "aparc.annot")],
              *[Path(f"mri/{name}.mgz") for name in ("brain.finalsurfs", "wm", "aseg.presurf")]]
    report = {"scope": "same_input_native_probe_and_full_python_pial_diagnostic",
              "hostname": platform.node(), "threads": args.threads,
              "cpu_affinity_count": len(os.sched_getaffinity(0)), "device": args.device,
              "code_base_commit": args.code_base_commit,
              "input_sha256": {str(p): sha(args.subject / p) for p in inputs},
              "probe_sha256": sha(args.probe), "script_sha256": sha(__file__),
              "native_reference_surface_sha256": sha(args.native_reference_surface),
              "timing_scope": "diagnostic including private per-step IO; not a performance gate",
              "status": "started"}

    def save():
        temporary = out / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(out / "report.json")

    save()
    native_root = args.existing_native_diagnostic_directory or out
    native = native_root / "native_subject"
    native_prefix = native_root / "native_gradient"
    values_prefix = native_root / "native_targets"
    if args.existing_native_diagnostic_directory is None:
        for relative in inputs:
            destination = native / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.subject / relative, destination)
        env = dict(os.environ, PLACE_GRAD_PREFIX=str(native_prefix),
                   PLACE_VALUES_PREFIX=str(values_prefix), FREESURFER_HOME=str(args.assets_directory))
        with (out / "native.private.log").open("w") as log:
            command = [sys.executable, "-m", "fnit.recon_all.pial_t1_conda", str(native), hemi,
                       "--binary", str(args.probe), "--assets-dir", str(args.assets_directory),
                       "--threads", str(args.threads)]
            started = time.perf_counter()
            process = subprocess.run(command, env=env, stdout=log, stderr=log)
        report["native_probe_wall_seconds"] = time.perf_counter() - started
        report["native_probe_exit_code"] = process.returncode
        if process.returncode:
            report["status"] = "failed_native_probe"
            save()
            raise RuntimeError("private native probe failed; inspect private log")
    else:
        receipt_path = native_root / "report.json"
        receipt = json.loads(receipt_path.read_text())
        for key in ("input_sha256", "probe_sha256", "native_reference_surface_sha256"):
            if report[key] != receipt[key]:
                raise ValueError(f"native checkpoint does not match {key}")
        if receipt.get("native_probe_exit_code") != 0:
            raise ValueError("native checkpoint did not complete")
        report["native_checkpoint_report_sha256"] = sha(receipt_path)
        report["native_probe_exit_code"] = 0
        report["native_probe_wall_seconds"] = None
    nx, nf = fs.read_geometry(str(native / f"surf/{hemi}.pial.T1"))
    reference, reference_faces = fs.read_geometry(str(args.native_reference_surface))
    report["probe_admission"] = {"ordered_faces_exact": bool(np.array_equal(nf, reference_faces)),
                                  "coordinates": compare(nx, reference)}
    if not np.array_equal(nx, reference) or not np.array_equal(nf, reference_faces):
        report["status"] = "failed_probe_geometry_admission"
        save()
        raise RuntimeError("instrumented native final geometry differs; cannot use it as native reference")
    save()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_pial_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    report["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    report["half_precision"] = False
    report["allocator_environment"] = {key: os.environ.get(key) for key in
        ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")}
    records, trace = {}, []
    state = {"step": 0, "part": {}}
    originals = {}

    def wrap(name, recorder):
        original = getattr(stage, name)
        originals[name] = original
        def call(*a, **k):
            result = original(*a, **k)
            recorder(a, k, result)
            return result
        setattr(stage, name, call)

    def intensity(a, k, result):
        state["step"] += 1
        state["part"] = {"current": a[1].copy(), "normals": a[2].copy(),
                         "ripped": a[3].copy(), "targets": a[4].copy(),
                         "sigmas": a[5].copy(), "intensity": result.copy(), "trials": []}
        if state["step"] == 1:
            np.savez(out / "python_static.npz", placement=a[0], affine=a[6], zooms=a[7])

    wrap("intensity_gradient", intensity)
    wrap("surface_repulsion_gradient", lambda a, k, r: state["part"].update(repulsion=r.copy()))
    wrap("average_signed_gradients", lambda a, k, r: state["part"].update(averaged=r.copy()))
    wrap("spring_gradient", lambda a, k, r: state["part"].update({k["direction"]: r.copy()}))
    wrap("quadratic_curvature", lambda a, k, r: state["part"].update(curvature=r.copy()))
    def collision(a, k, result):
        state["part"]["trials"].append({"proposed": a[2].copy(), "candidate": result[0].copy()})
        # 接受位移缓冲区由有序碰撞原地更新，不能把它标为原始梯度。
        state["part"]["accepted_offsets_final"] = k["accepted_offsets"].copy()
    wrap("asynchronous_first_step", collision)
    def callback(step, outer_pass, coordinates, diagnostics):
        part = state["part"]
        arrays = {key: value for key, value in part.items() if key != "trials"}
        arrays["accepted"] = coordinates
        for index, trial in enumerate(part["trials"]):
            arrays[f"proposal_{index}"] = trial["proposed"]
            arrays[f"candidate_{index}"] = trial["candidate"]
        path = out / f"python_step{step:02d}.npz"
        np.savez(path, **arrays)
        row = {"step": step, "pass": outer_pass, "diagnostics": diagnostics,
               "state_sha256": sha(path)}
        trace.append(row)
        report["python_trace"] = trace
        save()
    torch.cuda.synchronize(torch.device(args.device))
    started = time.perf_counter()
    try:
        result = stage.place_pial_t1(subject=args.subject, hemisphere=hemi,
            output=out / f"{hemi}.pial.python", max_steps=200,
            sampling_backend="cpu", regularization_backend="cpu",
            candidate_backend="torch_snapshot", candidate_grid_cells_per_axis=3,
            retained_mht_backend="compiled", cleanup_marking_backend="source_torch",
            cleanup_candidate_grid_cells_per_axis=3, device=args.device,
            trace_callback=callback, profile=True)
        torch.cuda.synchronize(torch.device(args.device))
    finally:
        for name, original in originals.items():
            setattr(stage, name, original)
    report["python_diagnostic_wall_seconds"] = time.perf_counter() - started
    report["python_result"] = result
    report["python_source_sha256"] = {Path(m.__file__).name: sha(m.__file__)
        for name, m in list(sys.modules.items()) if name.startswith("fnit.recon_all.place_")
        and getattr(m, "__file__", None)}
    comparisons = []
    for row in trace:
        step = row["step"]
        with np.load(out / f"python_step{step:02d}.npz") as py:
            clear = native_state(Path(f"{native_prefix}.step{step:02d}.clear"), len(nx))
            after = native_state(Path(f"{native_prefix}.step{step:02d}.after_collision"), len(nx))
            item = {"step": step, "pass": row["pass"],
                    "entry_coordinates": compare(clear["floats"][:, :3], py["current"]),
                    "accepted_coordinates": compare(after["floats"][:, :3], py["accepted"]),
                    "entry_normals": compare(clear["floats"][:, 3:6], py["normals"]),
                    "rip_flags": compare(clear["flags"][:, 0], py["ripped"])}
            targets = np.fromfile(f"{native_prefix}.step{step:02d}.intensity_input", dtype="<f4").reshape(-1, 2)
            item["targets"] = compare(targets[:, 0], py["targets"])
            item["sigmas"] = compare(targets[:, 1], py["sigmas"])
            for native_name, py_name in (("intensity", "intensity"), ("surface_repulsion", None),
                                         ("normal_spring", None), ("curvature", None),
                                         ("tangential_spring", "gradient")):
                term = native_state(Path(f"{native_prefix}.step{step:02d}.{native_name}"), len(nx))["floats"][:, 6:9]
                if py_name == "gradient":
                    candidate = np.float32(np.float32(np.float32(py["averaged"] + py["normal"]) +
                        np.float32(py["curvature"][:, None] * py["normals"])) + py["tangent"])
                elif py_name:
                    candidate = py[py_name]
                elif native_name == "surface_repulsion":
                    candidate = np.float32(py["intensity"] + py["repulsion"])
                elif native_name == "normal_spring":
                    candidate = np.float32(py["averaged"] + py["normal"])
                else:
                    candidate = np.float32(np.float32(py["averaged"] + py["normal"]) +
                        np.float32(py["curvature"][:, None] * py["normals"]))
                item[native_name] = compare(term, candidate)
            comparisons.append(item)
    report["per_step_comparison"] = comparisons
    report["first_different_accepted_coordinate_step"] = next(
        (r["step"] for r in comparisons if r["accepted_coordinates"]["different_elements"]), None)
    report["input_sha256_after"] = {str(p): sha(args.subject / p) for p in inputs}
    report["status"] = "complete" if report["input_sha256"] == report["input_sha256_after"] else "input_changed"
    save()
    print(json.dumps({"status": report["status"], "probe_admission": report["probe_admission"],
        "first_coordinate_difference_step": report["first_different_accepted_coordinate_step"]}))


if __name__ == "__main__":
    main()
