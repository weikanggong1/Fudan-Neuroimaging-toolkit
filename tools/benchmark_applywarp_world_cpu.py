#!/usr/bin/env python3
"""Isolated real-data CPU verification of ApplyWarp's explicit RAS world API.

Official fMRIPrep runs only in ``official-worker`` inside its frozen container.
``run`` uses complete images, one declared observation and a shared timing lock.
Private manifests/output images stay on the server; ``export`` writes aggregates.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def environment(threads):
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(threads)
        env["SINGULARITYENV_" + name] = str(threads)
    env["OMP_DYNAMIC"] = env["MKL_DYNAMIC"] = "FALSE"
    return env


def worker(request_path, result_path, official=False):
    request = json.loads(Path(request_path).read_text())
    threads = request["threads"]
    os.sched_setaffinity(0, set(request["affinity"]))
    if not official:
        source_root = Path(request["source_root"]).resolve()
        sys.path.insert(0, str(source_root / "src"))
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        from fnit.applywarp import TorchApplyWarp, WorldTransformChain
        import fnit.applywarp.core as actual_core
        if not Path(actual_core.__file__).resolve().is_relative_to(source_root / "src"):
            raise RuntimeError("Wrong frozen FNIT imported")
    import nibabel as nib
    import numpy as np
    if official:
        import nitransforms as nt
        from fmriprep.interfaces.resampling import resample_image
    case = request["case"]
    official_concurrency = int(case.get("official_concurrency", threads))
    if not 1 <= official_concurrency <= threads:
        raise ValueError("Official frame concurrency must fit the declared CPU budget")
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source, target = nib.load(case["input"]), nib.load(case["reference"])
    matrix = np.linalg.inv(np.loadtxt(case["bbr_forward_world"]))
    field = nib.load(case["pull_ras"]) if case.get("pull_ras") else None
    motion = np.load(case["motion"], allow_pickle=False) if case.get("motion") else None
    mask = nib.load(case["output_mask"]) if case.get("output_mask") else None
    if case.get("complete_490") and (source.ndim != 4 or source.shape[3] != 490
                                     or motion is None or motion.shape != (490, 4, 4)):
        raise ValueError("The complete main case requires all 490 frames and matrices")
    if field is not None and (field.shape != (*target.shape[:3], 3)
                             or not np.allclose(field.affine, target.affine, atol=1e-4)):
        raise ValueError("Saved RAS pull field must use the target grid")
    if official:
        if case["coordinate_precision"] != "fmriprep" or case["boundary"] != "grid-constant":
            raise ValueError("An FNIT extension is not a matching fMRIPrep benchmark")
        transforms = [] if field is None else [nt.nonlinear.DenseFieldTransform(field, is_deltas=True)]
        transforms.append(nt.Affine(matrix))
        if motion is not None:
            transforms.append(nt.linear.LinearTransformsMapping(motion))
        image = resample_image(
            source, target, nt.TransformChain(transforms), fieldmap=None,
            pe_info=None, jacobian=False, nthreads=official_concurrency, output_dtype="f4",
            order={"nearest": 0, "linear": 1, "spline": 3}[case["interpolation"]],
            mode="grid-constant", cval=0, prefilter=True,
        )
        if mask is not None:
            values = np.asanyarray(image.dataobj).astype(np.float32, copy=True)
            valid = np.asanyarray(mask.dataobj) > .5
            values *= valid if values.ndim == 3 else valid[..., None]
            image = nib.Nifti1Image(values, image.affine, image.header)
        nib.save(image, output)
    else:
        transformation = WorldTransformChain(target, matrix, field, motion, case["coordinate_precision"])
        warper = TorchApplyWarp(device="cpu")
        arguments = dict(interpolation=case["interpolation"], boundary=case["boundary"],
                         output_mask=mask, batch_size=case["batch_size"],
                         spatial_chunk_size=case["spatial_chunk_size"])
        if case.get("entry", "run_world") == "apply_world":
            image = warper.apply_world(source, transformation, **arguments)
            nib.save(image, output)
        else:
            warper.run_world(source, transformation, output, **arguments)
    elapsed = time.perf_counter() - start
    save_json(result_path, {
        "api_load_compute_save_seconds": elapsed,
        "scope": "entry framework imports/setup excluded; function-internal lazy imports and complete source/target/field/matrices/mask loading, computation and NIfTI save included",
        "source_shape": list(source.shape), "output_shape": list(nib.load(output).shape),
        "affinity": sorted(os.sched_getaffinity(0)), "thread_budget": threads,
        "official_frame_concurrency": official_concurrency if official else None,
        "openblas_thread_limit": os.environ.get("OPENBLAS_NUM_THREADS"),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": str(output), "pid": os.getpid(),
        "versions": {"numpy": np.__version__, "nibabel": nib.__version__,
                     "scipy": importlib.metadata.version("scipy"),
                     **({"nitransforms": nt.__version__} if official else {"torch": torch.__version__})},
    })


def metadata(image):
    import numpy as np
    return {"shape": list(image.shape), "dtype": np.dtype(image.get_data_dtype()).name,
            "zooms": list(map(float, image.header.get_zooms())),
            "xyzt_units": list(image.header.get_xyzt_units()),
            "qform_code": int(image.header["qform_code"]), "sform_code": int(image.header["sform_code"]),
            "slope": float(image.dataobj.slope), "intercept": float(image.dataobj.inter)}


def compare(left_path, right_path, mask_path):
    import numpy as np
    import nibabel as nib
    left, right = nib.load(left_path), nib.load(right_path)
    if left.shape != right.shape:
        return {"status": "shape_mismatch", "left_shape": left.shape, "right_shape": right.shape}
    mask = np.asanyarray(nib.load(mask_path).dataobj) > .5
    total = {name: dict(count=0, different=0, bit_different=0, absolute=0., squared=0., reference_squared=0., maximum=0., nonfinite=0)
             for name in ("whole", "brain", "outside_brain")}
    frames = 1 if left.ndim == 3 else left.shape[-1]
    for frame in range(frames):
        a = np.asarray(left.dataobj if left.ndim == 3 else left.dataobj[..., frame], dtype=np.float32).reshape(-1)
        b = np.asarray(right.dataobj if right.ndim == 3 else right.dataobj[..., frame], dtype=np.float32).reshape(-1)
        for start in range(0, a.size, 262144):
            end = min(start + 262144, a.size)
            brain = mask.reshape(-1)[start:end]
            for name, selection in (("whole", slice(None)), ("brain", brain), ("outside_brain", ~brain)):
                aa, bb = a[start:end][selection], b[start:end][selection]
                finite = np.isfinite(aa) & np.isfinite(bb)
                state = total[name]
                state["count"] += aa.size
                state["nonfinite"] += int((~finite).sum())
                state["bit_different"] += int(np.count_nonzero(aa.view(np.uint32) != bb.view(np.uint32)))
                delta = aa[finite].astype(np.float64) - bb[finite].astype(np.float64)
                state["different"] += int(np.count_nonzero(delta))
                state["absolute"] += float(np.abs(delta).sum())
                state["squared"] += float((delta * delta).sum())
                state["reference_squared"] += float((bb[finite].astype(np.float64) ** 2).sum())
                if delta.size: state["maximum"] = max(state["maximum"], float(np.abs(delta).max()))
    regions = {}
    for name, state in total.items():
        count = state["count"]
        regions[name] = {
            "values": count, "nonfinite": state["nonfinite"], "different_values": state["different"],
            "different_fraction": state["different"] / count if count else None,
            "bit_different_values": state["bit_different"],
            "max_absolute_error": state["maximum"], "mae": state["absolute"] / count if count else None,
            "rmse": (state["squared"] / count) ** .5 if count else None,
            "relative_l2": (state["squared"] / max(state["reference_squared"], 1e-60)) ** .5,
        }
    return {"status": "compared", "regions": regions, "left_metadata": metadata(left),
            "right_metadata": metadata(right), "affine_max_error": float(np.abs(left.affine - right.affine).max()),
            "header_binary_identical": left.header.binaryblock == right.header.binaryblock,
            "extensions_identical": left.header.extensions == right.header.extensions}


def run(args):
    if not args.single_observation:
        raise ValueError("This protocol requires explicit --single-observation")
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpus = [int(value) for value in args.cpuset.split(",")]
    if len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("Provide distinct accessible CPUs")
    budgets = [int(value) for value in args.threads.split(",")]
    if min(budgets) < 1 or max(budgets) > len(cpus): raise ValueError("Invalid thread budgets")
    tool = Path(__file__).resolve()
    report = {"schema_version": 1, "host": socket.gethostname(), "status": "waiting_for_lock",
              "timing_protocol": "single complete observation, no warmup/repeated stable speed estimate",
              "manifest_sha256": sha256(args.manifest), "tool_sha256": sha256(tool), "records": []}
    report["source_hashes"] = {}
    for name, root in (("candidate", args.candidate_root), ("baseline", args.baseline_root)):
        report["source_hashes"][name] = {relative: sha256(Path(root) / relative) for relative in (
            "src/fnit/applywarp/core.py", "src/fnit/_world_resampling.py",
            "src/fnit/eddy/fsl2111_strict/spline.py", "src/fnit/flirt/coordinates.py",
        )}
    report["official_program"] = {"container_sha256": sha256(manifest["fmriprep_image"]),
                                  "singularity_sha256": sha256(manifest["singularity"])}
    save_json(output / "suite.private.json", report)
    with Path(args.lock_file).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        report["status"] = "running"
        for threads in budgets:
            affinity = cpus[:threads]
            for case in manifest["cases"]:
                if args.case_id and case["id"] not in args.case_id: continue
                location = output / f"threads_{threads}" / case["id"]
                location.mkdir(parents=True, exist_ok=True)
                official = case.get("official_match", False)
                reference_backend = "official" if official else "baseline"
                backends = [reference_backend, "candidate"] if threads == budgets[0] else ["candidate", reference_backend]
                record = {"case_id": case["id"], "case_semantics": {k: case[k] for k in (
                    "interpolation", "boundary", "coordinate_precision", "entry", "batch_size", "spatial_chunk_size", "complete_490")},
                    "masked": bool(case.get("output_mask")), "threads": threads,
                    "affinity": affinity, "reference_backend": reference_backend, "load_before": os.getloadavg(),
                    "observations": {}, "input_hashes": {}}
                for key in ("input", "reference", "pull_ras", "bbr_forward_world", "motion", "output_mask"):
                    if case.get(key): record["input_hashes"][key] = sha256(case[key])
                for backend in backends:
                    destination = location / backend
                    destination.mkdir(parents=True, exist_ok=True)
                    root = args.candidate_root if backend == "candidate" else args.baseline_root
                    request = {"case": case, "threads": threads, "affinity": affinity,
                               "source_root": root, "output": str(destination / "warped.nii")}
                    request_path, result_path = destination / "request.private.json", destination / "worker.private.json"
                    save_json(request_path, request)
                    command = [args.python, str(tool), "worker", "--request", str(request_path), "--result", str(result_path)]
                    if backend == "official":
                        command = [manifest["singularity"], "exec", "--cleanenv", "--bind", "/cwStorage:/cwStorage",
                                   "--bind", "/public:/public", manifest["fmriprep_image"], "python", str(tool),
                                   "official-worker", "--request", str(request_path), "--result", str(result_path)]
                    env = environment(threads)
                    openblas_threads = int(case.get("openblas_threads", threads))
                    if not 1 <= openblas_threads <= threads:
                        raise ValueError("OpenBLAS thread limit must fit the declared CPU budget")
                    env["OPENBLAS_NUM_THREADS"] = env["SINGULARITYENV_OPENBLAS_NUM_THREADS"] = str(openblas_threads)
                    env["PYTHONPATH"] = str(Path(root) / "src")
                    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    before = time.perf_counter()
                    with (destination / "stdout.private.log").open("wb") as out, (destination / "stderr.private.log").open("wb") as err:
                        process = subprocess.run(["/usr/bin/taskset", "-c", ",".join(map(str, affinity)), *command],
                                                 env=env, stdout=out, stderr=err)
                    elapsed = time.perf_counter() - before
                    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    if process.returncode:
                        record["failure"] = {"backend": backend, "exit_code": process.returncode}
                        save_json(location / "timing.private.json", record)
                        raise RuntimeError("World worker failed; see private logs")
                    details = json.loads(result_path.read_text())
                    details.update(full_process_seconds=elapsed,
                                   user_cpu_seconds=after_usage.ru_utime-before_usage.ru_utime,
                                   system_cpu_seconds=after_usage.ru_stime-before_usage.ru_stime,
                                   output_sha256=sha256(details["output"]))
                    record["observations"][backend] = details
                    save_json(location / "timing.private.json", record)
                    print(json.dumps({"case": case["id"], "threads": threads, "backend": backend, "status": "executed"}), flush=True)
                record["precision"] = compare(record["observations"]["candidate"]["output"],
                    record["observations"][reference_backend]["output"], manifest["brain_mask"])
                record["load_after"] = os.getloadavg()
                report["records"].append(record)
                save_json(location / "timing.private.json", record)
                save_json(output / "suite.private.json", report)
        report["status"] = "executed_with_full_precision_comparisons"
        save_json(output / "suite.private.json", report)


def export(source, destination):
    private = json.loads(Path(source).read_text())
    public = {key: private[key] for key in ("schema_version", "host", "status", "timing_protocol", "tool_sha256", "source_hashes", "official_program")}
    public["privacy"] = "Private real fMRI chain: aggregate results only; no participant identifiers, paths or images"
    public["records"] = []
    for record in private["records"]:
        value = {key: record[key] for key in ("case_id", "case_semantics", "masked", "threads", "affinity", "reference_backend", "precision")}
        value["observations"] = {}
        for backend, item in record["observations"].items():
            value["observations"][backend] = {key: item[key] for key in (
                "api_load_compute_save_seconds", "scope", "source_shape", "output_shape", "affinity",
                "thread_budget", "maximum_rss_kib", "versions", "full_process_seconds", "user_cpu_seconds", "system_cpu_seconds")}
            if "official_frame_concurrency" in item:
                value["observations"][backend]["official_frame_concurrency"] = item["official_frame_concurrency"]
            if "openblas_thread_limit" in item:
                value["observations"][backend]["openblas_thread_limit"] = item["openblas_thread_limit"]
        public["records"].append(value)
    save_json(destination, public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("worker", "official-worker"):
        child = sub.add_parser(name)
        child.add_argument("--request", required=True); child.add_argument("--result", required=True)
    child = sub.add_parser("run")
    for name in ("manifest", "candidate-root", "baseline-root", "output-dir", "cpuset", "lock-file"):
        child.add_argument("--"+name, required=True)
    child.add_argument("--python", default=sys.executable)
    child.add_argument("--threads", default="1,8")
    child.add_argument("--single-observation", action="store_true")
    child.add_argument("--case-id", action="append")
    child = sub.add_parser("export"); child.add_argument("--source", required=True); child.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in ("worker", "official-worker"):
        worker(args.request, args.result, args.command == "official-worker")
    elif args.command == "run": run(args)
    else: export(args.source, args.output)


if __name__ == "__main__": main()
