"""新依赖target/预检的完整N4同输入回归；参考仅在候选写出后读取。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module-directory", "input", "reference", "reference-report", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("input-sha256", "reference-sha256", "reference-report-sha256", "code-version"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all import input_n4_chain, n4_simpleitk, n4_itk, mgh_compat
    import nibabel as nib
    import numpy as np
    import torch
    if sha(args.input) != args.input_sha256:
        raise ValueError("frozen input SHA changed")
    if sha(args.reference) != args.reference_sha256 or sha(args.reference_report) != args.reference_report_sha256:
        raise ValueError("frozen reference SHA changed")
    before_cuda = torch.cuda.is_initialized()
    before_policies = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    dependencies = input_n4_chain.require_n4_runtime_dependencies(n4_backend="simpleitk")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    output = args.output_directory / "nu0.mgz"
    started = time.perf_counter()
    detail = input_n4_chain.run_n4_stage(input_path=args.input, output_path=output,
        n4_backend="simpleitk", n4_execution="in-process", threads=4, device="cpu")
    api_seconds = time.perf_counter() - started
    # 完整生产调用只接T1。旧参考在候选完成后才作为比较器输入读取。
    reference_report = json.loads(args.reference_report.read_bytes())
    if reference_report["status"] != "complete" or reference_report["source_sha256"] != args.input_sha256:
        raise ValueError("reference report status/input binding invalid")
    actual, reference, source = [nib.load(str(path)) for path in (output, args.reference, args.input)]
    values, expected = np.asarray(actual.dataobj), np.asarray(reference.dataobj)
    difference = np.abs(values.astype(np.int16) - expected.astype(np.int16))
    geometry = {"shape": actual.shape == reference.shape == source.shape,
        "dtype": actual.get_data_dtype() == reference.get_data_dtype() == np.dtype("uint8"),
        "affine": bool(np.array_equal(actual.affine, reference.affine) and np.array_equal(actual.affine, source.affine)),
        "vox2ras_tkr": bool(np.array_equal(actual.header.get_vox2ras_tkr(), reference.header.get_vox2ras_tkr()) and
                             np.array_equal(actual.header.get_vox2ras_tkr(), source.header.get_vox2ras_tkr()))}
    passed = not np.count_nonzero(difference) and all(geometry.values())
    detail.pop("output", None)  # 公共数值报告不含私有绝对路径；输出名另记录。
    report = {"status": "complete" if passed else "failed_comparison", "case": "ds000114_sub06_ses-test",
        "scope": "complete same-input N4 file API; dependency recovery, not whole reconstruction",
        "code_version": args.code_version, "driver_sha256": sha(__file__),
        "module_sha256": {name: sha(module.__file__) for name, module in
            (("input_n4_chain", input_n4_chain), ("n4_simpleitk", n4_simpleitk), ("n4_itk", n4_itk), ("mgh_compat", mgh_compat))},
        "input_sha256_before": args.input_sha256, "input_sha256_after": sha(args.input),
        "reference_report_sha256": sha(args.reference_report), "reference_output_sha256": sha(args.reference),
        "reference_module_sha256": reference_report["module_sha256"], "output": output.name,
        "output_sha256": sha(output), "raw_output_file_exact": sha(output) == sha(args.reference),
        "different_voxels": int(np.count_nonzero(difference)), "maximum_absolute_error": int(difference.max()),
        "p99_absolute_error": float(np.percentile(difference, 99)), "geometry_exact": geometry,
        "library": dependencies, "api_seconds_including_image_io_and_preflight": api_seconds, "detail": detail,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads_budget": 4, "fitting_threads": 1,
        "reconstruction_threads": 4, "shared_host_load": "other FNIT jobs active; observation without paired speed claim",
        "cuda_initialized_before": before_cuda, "cuda_initialized_after": torch.cuda.is_initialized(),
        "tf32_before": before_policies, "tf32_after": [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
        "used_native_programs": False, "overall_equivalence": "not_assessed", "full_workflow_complete": False}
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed or report["input_sha256_after"] != args.input_sha256:
        raise RuntimeError("frozen full N4 changed decoded output/input or geometry")
    (args.output_directory / "COMPLETE.json").write_text(json.dumps({"status": "complete",
        "report_sha256": sha(args.output_directory / "report.json"), "output_sha256": sha(output)}, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "different_voxels": report["different_voxels"],
        "geometry_exact": geometry, "api_seconds": api_seconds}), flush=True)


if __name__ == "__main__":
    main()
