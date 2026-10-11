"""同一冻结原网格做完整N4文件API线程ABBA；参考只在调用后诊断读取。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_one(args):
    process_tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    import_tick = time.perf_counter()
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all import n4_simpleitk as implementation
    from fnit.recon_all import n4_itk, mgh_compat
    import_seconds = time.perf_counter() - import_tick
    root = args.fnit_root
    input_path = root / "runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1" / (args.case + "-candidate") / "subject/mri/orig.mgz"
    baseline = root / "runs/recon_n4_python_adapter_20261010_v1" / args.case
    reference = json.loads((baseline / "report.json").read_text())
    input_before = sha(input_path)
    if input_before != reference["source_sha256"]:
        raise ValueError("declared same input differs from frozen native reference")
    global_before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    capture = {}
    original_factory = sitk.N4BiasFieldCorrectionImageFilter
    original_array = implementation.correct_array

    class FieldObserver:
        def __init__(self):
            self.filter = original_factory()

        def __getattr__(self, name):
            return getattr(self.filter, name)

        def GetLogBiasFieldAsImage(self, image):
            field = self.filter.GetLogBiasFieldAsImage(image)
            capture["log_field"] = field  # 不在计时的API内部额外下载/保存数组。
            return field

    def array_observer(*positional, **keywords):
        output = original_array(*positional, **keywords)
        capture["corrected"] = output[0]  # 只保留原返回对象，不重算N4。
        return output

    sitk.N4BiasFieldCorrectionImageFilter = FieldObserver
    implementation.correct_array = array_observer
    api_tick = time.perf_counter()
    try:
        details = implementation.correct_volume(input_file=input_path,
            output_file=args.output / "nu0.mgz", fitting_threads=args.fitting_threads,
            reconstruction_threads=4, profile_path=args.output / "profile.json")
    finally:
        implementation.correct_array = original_array
        sitk.N4BiasFieldCorrectionImageFilter = original_factory
    api_seconds = time.perf_counter() - api_tick
    global_after = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    log_field = np.ascontiguousarray(sitk.GetArrayFromImage(capture["log_field"]).transpose(2, 1, 0))
    corrected = capture["corrected"]
    np.save(args.output / "log_bias_field.npy", log_field)
    np.save(args.output / "corrected_float.npy", corrected)
    source, actual = (nib.load(str(p)) for p in (input_path, args.output / "nu0.mgz"))
    report = {"status": "complete", "scope": "same-input complete file N4; no raw-T1 whole",
        "case": args.case, "fitting_threads": args.fitting_threads, "reconstruction_threads": 4,
        "source_module_sha256": sha(implementation.__file__), "script_sha256": sha(__file__),
        "n4_conversion_sha256": sha(n4_itk.__file__), "mgh_io_sha256": sha(mgh_compat.__file__),
        "source_binding_sha256": sha(args.module_directory / "CANDIDATE_BINDING.json"),
        "input_sha256_before": input_before, "input_sha256_after": sha(input_path),
        "reference_report_sha256": sha(baseline / "report.json"),
        "reference_program_sha256": reference["reference_program_sha256"],
        "reference_source_sha256": reference["reference_source_sha256"],
        "reference_scope": "existing two repeat original program runs, same benchmark host; not rerun now",
        "CPU_affinity": sorted(os.sched_getaffinity(0)), "hostname": platform.node(),
        "CPU_model": next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")),
        "load_average_before_record": list(os.getloadavg()),
        "imports_seconds": import_seconds, "complete_API_seconds_including_read_write_profile_hash": api_seconds,
        "details": details, "global_threads_before": global_before, "global_threads_after": global_after,
        "global_threads_restored": global_before == global_after,
        "input_shape": [int(x) for x in source.shape], "output_shape": [int(x) for x in actual.shape],
        "input_dtype": str(source.get_data_dtype()), "output_dtype": str(actual.get_data_dtype()),
        "geometry_exact": bool(np.array_equal(source.affine, actual.affine)),
        "input_affine_mm": source.affine.tolist(), "output_affine_mm": actual.affine.tolist(),
        "log_field_dtype": str(log_field.dtype), "corrected_dtype": str(corrected.dtype),
        "arrays_finite": bool(np.isfinite(log_field).all() and np.isfinite(corrected).all()),
        "log_field_sha256": sha(args.output / "log_bias_field.npy"),
        "corrected_float_sha256": sha(args.output / "corrected_float.npy"),
        "nu0_sha256": sha(args.output / "nu0.mgz"),
        "child_wall_seconds_before_final_report": time.perf_counter() - process_tick,
        "observation": "only retained library outputs; diagnostics array saving outside complete API clock",
        "GPU": "not used; no CUDA operation", "production_default_changed": False}
    write(args.output / "run.json", report)
    if not report["global_threads_restored"] or not report["arrays_finite"] or not report["geometry_exact"] or input_before != report["input_sha256_after"]:
        raise RuntimeError("N4 correctness/context gate failed")
    print(json.dumps({"case": args.case, "fitting_threads": args.fitting_threads,
                      "API_seconds": api_seconds, "fitting_seconds": details["timing_parts_seconds"]["fitting"]}), flush=True)


def compare_case(case, paths, root, output):
    import numpy as np
    import nibabel as nib
    from fnit.recon_all.n4_itk import _to_uchar
    records = [json.loads((path / "run.json").read_text()) for path in paths]
    if any(r["status"] != "complete" for r in records) or len({r["input_sha256_before"] for r in records}) != 1:
        raise ValueError("ABBA producer status/input identity changed")

    def difference(first, second):
        error = np.abs(first.astype(np.float64) - second.astype(np.float64))
        return {"different_voxels": int(np.count_nonzero(error)), "maximum_absolute_error": float(error.max()),
                "p99_absolute_error": float(np.percentile(error, 99)), "mean_absolute_error": float(error.mean()),
                "first_dtype": str(first.dtype), "second_dtype": str(second.dtype)}

    names = {"log_bias_field": "log_bias_field.npy", "corrected_float": "corrected_float.npy", "nu0": "nu0.mgz"}
    comparisons = {}
    for name, filename in names.items():
        arrays = [(np.asarray(nib.load(str(p / filename)).dataobj) if name == "nu0" else np.load(p / filename)) for p in paths]
        comparisons[name] = {label: difference(arrays[a], arrays[b]) for label, a, b in
            (("A1_vs_B1", 0, 1), ("A1_vs_B2", 0, 2), ("A1_vs_A2", 0, 3), ("B1_vs_B2", 1, 2))}
    reference_directory = root / "runs/recon_n4_python_adapter_20261010_v1" / case
    reference = json.loads((reference_directory / "report.json").read_text())
    shape = tuple(records[0]["input_shape"])
    native_arrays = [np.fromfile(reference_directory / f"native-{index}.f32", dtype=np.float32).reshape(shape, order="F") for index in (0, 1)]
    if sha(reference_directory / "native-0.f32") != reference["native_float_sha256"]:
        raise ValueError("native frozen checkpoint SHA changed")
    native_repeats = difference(native_arrays[0], native_arrays[1])
    native_comparisons = []
    for path in paths:
        native_comparisons.append({"float_corrected": difference(np.load(path / "corrected_float.npy"), native_arrays[0]),
            "uint8": difference(np.asarray(nib.load(str(path / "nu0.mgz")).dataobj), _to_uchar(native_arrays[0]))})
    observed_a = [records[i]["complete_API_seconds_including_read_write_profile_hash"] for i in (0, 3)]
    observed_b = [records[i]["complete_API_seconds_including_read_write_profile_hash"] for i in (1, 2)]
    report = {"status": "complete", "case": case, "order": [1, 4, 4, 1], "same_input": True,
        "run_json_sha256": [sha(path / "run.json") for path in paths], "comparisons": comparisons,
        "native_repeat_float": native_repeats, "native_comparisons_each_ABBA_run": native_comparisons,
        "native_float_sha256_each_repeat": [sha(reference_directory / f"native-{i}.f32") for i in (0, 1)],
        "native_reference_report_sha256": sha(reference_directory / "report.json"),
        "native_reference_same_host": records[0]["hostname"] == reference["host"],
        "median_A_API_seconds": float(np.median(observed_a)), "median_B_API_seconds": float(np.median(observed_b)),
        "observed_API_ratio_A_over_B": float(np.median(observed_a) / np.median(observed_b)),
        "all_geometry_exact": all(r["geometry_exact"] for r in records),
        "all_thread_context_restored": all(r["global_threads_restored"] for r in records),
        "overall_equivalence": "not_assessed", "whole_pipeline_speedup": "not_measured", "production_default_changed": False}
    write(output / f"{case}_comparisons.json", report)


def run_group(args):
    args.output.mkdir(parents=True, exist_ok=False)
    for case in ("sub07", "sub06"):
        runs = []
        for index, fit in enumerate((1, 4, 4, 1)):
            directory = args.output / (case + f"-{index + 1}-fit{fit}")
            runs.append(directory)
            command = [sys.executable, __file__, "--mode", "one", "--fnit-root", str(args.fnit_root),
                "--module-directory", str(args.module_directory), "--output", str(directory),
                "--case", case, "--fitting-threads", str(fit)]
            started = time.perf_counter()
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (args.output / (directory.name + ".log")).write_text(process.stdout)
            write(args.output / (directory.name + "_process.json"), {"exit_code": process.returncode,
                "full_process_wall_seconds": time.perf_counter() - started, "command_scope": "own Python N4 file API; no FS stage program"})
            print(process.stdout, flush=True)
            if process.returncode:
                raise RuntimeError(f"N4 run failed; log retained: {directory.name}")
        compare_case(case, runs, args.fnit_root, args.output)
    write(args.output / "COMPLETE.json", {"status": "complete", "cases": 2, "runs": 8,
        "order": "sub07 ABBA then sub06 ABBA", "script_sha256": sha(__file__)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("one", "group"), required=True)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--module-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("sub07", "sub06"))
    parser.add_argument("--fitting-threads", type=int, choices=(1, 4))
    args = parser.parse_args()
    if args.mode == "one" and (args.case is None or args.fitting_threads is None):
        parser.error("mode=one requires case/fitting-threads")
    (run_one if args.mode == "one" else run_group)(args)


if __name__ == "__main__":
    main()
