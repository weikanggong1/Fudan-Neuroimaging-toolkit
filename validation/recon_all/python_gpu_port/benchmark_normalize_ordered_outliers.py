"""同输入配对归一化：交换两种控制点实现，单独保留诊断图与性能运行。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace

import nibabel as nib
import numba
import numpy as np
import scipy
import torch

from fnit.recon_all.normalization import aseg_pipeline, pipeline
from fnit.recon_all.normalization import normalize_3d_controls as candidate
from fnit.recon_all.normalization import normalize_gentle_source as candidate_gentle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_baseline(path: Path, suffix: str):
    name = "fnit.recon_all.normalization._baseline_benchmark_" + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_checkout() -> dict:
    """记录可访问的实际 Git 状态；归档源码没有 Git 时明确记为不可用。"""
    repository = Path(candidate.__file__).resolve().parents[4]
    if not (repository / ".git").exists():
        return {"root": str(repository), "head": None, "status": "source archive without Git metadata"}
    result = {"root": str(repository)}
    for name, command in (("head", ["git", "rev-parse", "HEAD"]),
                          ("status", ["git", "status", "--short"])):
        completed = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
        result[name] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def _compare(left: Path, right: Path) -> dict:
    images = [nib.load(str(path)) for path in (left, right)]
    a, b = [np.asarray(image.dataobj) for image in images]
    result = {"left": str(left), "right": str(right),
              "sha256": [_sha256(left), _sha256(right)],
              "shape": [list(a.shape), list(b.shape)],
              "dtype": [str(a.dtype), str(b.dtype)],
              "affine_equal": bool(np.array_equal(images[0].affine, images[1].affine)),
              "header_equal": images[0].header.binaryblock == images[1].header.binaryblock}
    if a.shape == b.shape:
        error = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result.update(different_values=int(np.count_nonzero(a != b)),
                      maximum_absolute_error=float(error.max(initial=0)),
                      p99_absolute_error=float(np.quantile(error, 0.99)))
    result["numerically_identical"] = bool(
        a.shape == b.shape and a.dtype == b.dtype and result["affine_equal"]
        and result.get("different_values") == 0)
    return result


@contextlib.contextmanager
def _patched_controls(module, captures=None):
    """临时替换同一算子；诊断拷贝在独立运行中执行，离开后恢复原函数。"""
    saved = []

    def install(owner, name, function):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)

    def select_controls(source, wm_peak=None, gm_peak=None):
        output, detail = module.controls_3d(source, wm_peak=wm_peak, gm_peak=gm_peak)
        if captures is not None:
            index = 1 + sum(name.startswith("three_d_") and name.endswith("_control")
                            for name in captures)
            captures[f"three_d_{index}_source"] = np.array(source, copy=True)
            captures[f"three_d_{index}_control"] = np.array(output, copy=True)
        return output, detail

    install(pipeline, "controls_3d", select_controls)
    install(aseg_pipeline, "controls_3d", select_controls)
    for owner in (pipeline, aseg_pipeline):
        def gentle(source):
            output, detail = module.gentle_controls(source)
            if captures is not None:
                captures["gentle_control"] = output.detach().cpu().numpy().copy()
            return output, detail

        install(owner, "gentle_controls", gentle)
    if captures is not None:
        original_ridge = aseg_pipeline.medial_ridge
        original_filter = aseg_pipeline.filter_aseg_ridge

        def ridge(aseg):
            output, detail = original_ridge(aseg)
            captures["aseg_ridge"] = np.array(output, copy=True)
            return output, detail

        def filter_ridge(masked, ridge):
            output, removed, wm_peak = original_filter(masked, ridge)
            captures["aseg_control"] = np.array(output, copy=True)
            captures["aseg_removed_control"] = np.array(removed, copy=True)
            return output, removed, wm_peak

        install(aseg_pipeline, "medial_ridge", ridge)
        install(aseg_pipeline, "filter_aseg_ridge", filter_ridge)
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


def _run(args, stage: str, implementation, directory: Path, diagnostic: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("T1.mgz" if stage == "first" else "brain.mgz")
    captures = {} if diagnostic else None
    cuda = torch.device(args.device).type == "cuda"
    cache_disabled = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
    if cuda:
        torch.cuda.synchronize(args.device)
        if not cache_disabled:
            torch.cuda.reset_peak_memory_stats(args.device)
    with _patched_controls(implementation, captures=captures):
        helper_signatures_before = [str(signature) for signature in candidate._remove_outliers_ordered.signatures]
        started = time.perf_counter()
        if stage == "first":
            result = pipeline.normalize_t1(
                input_file=args.nu, xfm_file=args.xfm, output_file=output,
                device=args.device, three_d_iterations=2, diagnostic_dir=None)
        else:
            result = aseg_pipeline.normalize_t1_aseg(
                norm_file=args.norm, aseg_file=args.aseg,
                brainmask_file=args.brainmask, output_file=output,
                device=args.device, three_d_iterations=2)
        if cuda:
            torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - started
    report = {"output": str(output), "seconds_including_input_and_output_io": seconds,
              "function_report": result, "diagnostic_run": diagnostic,
              "candidate_helper_signatures_before": helper_signatures_before,
              "candidate_helper_signatures_after": [str(signature) for signature in candidate._remove_outliers_ordered.signatures],
              "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(args.device)
                  if cuda and not cache_disabled else None),
              "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(args.device)
                  if cuda and not cache_disabled else None)}
    if captures is not None:
        affine = nib.load(str(args.nu if stage == "first" else args.norm)).affine
        paths = {}
        for name, data in captures.items():
            path = directory / f"{name}.mgh"
            if data.dtype == bool:
                data = data.astype(np.uint8)
            nib.save(nib.MGHImage(data, affine), str(path))
            paths[name] = str(path)
        report["diagnostic_maps"] = paths
        report["diagnostic_wall_including_map_write_seconds"] = time.perf_counter() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nu", "xfm", "norm", "aseg", "brainmask", "reference-t1", "reference-brain"):
        parser.add_argument("--" + name, type=Path)
    for name in ("baseline-controls", "baseline-gentle", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-commit", "code-commit", "device"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("first", "second", "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timing-context", default="not specified")
    args = parser.parse_args()
    stages = ("first", "second") if args.stage == "both" else (args.stage,)
    required = set()
    if "first" in stages:
        required.update(("nu", "xfm"))
    if "second" in stages:
        required.update(("norm", "aseg", "brainmask"))
    if any(getattr(args, name) is None for name in required):
        parser.error("requested stages require: " + ", ".join(sorted(required)))
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    # 空目录防止把旧文件误记为本次诊断或连续运行产物。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty")
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1234)
    baseline = SimpleNamespace(
        controls_3d=_load_baseline(args.baseline_controls, "controls").controls_3d,
        gentle_controls=_load_baseline(args.baseline_gentle, "gentle").gentle_controls)
    source_dir = Path(candidate.__file__).parent
    report = {
        "scope": "frozen same-input stages; controls_3d and gentle_controls are replaced; shared stages use candidate sources",
        "candidate_code_commit": args.code_commit, "baseline_controls_commit": args.baseline_commit,
        "host": platform.node(), "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(), "device": args.device, "threads": args.threads,
        "actual_checkout": _source_checkout(),
        "timing_context": args.timing_context, "seed": 1234,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "half_precision": False,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                     "numba": numba.__version__, "nibabel": nib.__version__},
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
            "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_VISIBLE_DEVICES")},
        "input_sha256": {name: _sha256(getattr(args, name)) for name in sorted(required)},
        "input_paths": {name: str(getattr(args, name)) for name in sorted(required)},
        "baseline_controls_sha256": _sha256(args.baseline_controls),
        "baseline_gentle_sha256": _sha256(args.baseline_gentle),
        "candidate_source_sha256": {str(path): _sha256(path) for path in sorted(source_dir.glob("*.py"))},
        "mgh_io_source_sha256": _sha256(source_dir.parent / "mgh_compat.py"),
        "benchmark_script_sha256": _sha256(Path(__file__)),
        "jit_policy": "no explicit warmup; two paired rounds expose first compilation/cache loading and later in-process reuse",
        "process_gpu_memory": "not sampled here; pair external parent/child process sampling with this report",
        "stages": {},
    }
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        report["cpu_model"] = next((line.split(":", 1)[1].strip()
                                    for line in cpu_info.read_text().splitlines()
                                    if line.startswith("model name")), "unavailable")
    if torch.device(args.device).type == "cuda":
        properties = torch.cuda.get_device_properties(args.device)
        report["gpu"] = {"name": properties.name, "uuid": str(getattr(properties, "uuid", "unavailable")),
                         "total_memory_bytes": properties.total_memory}
    implementations = {"baseline": baseline, "candidate": SimpleNamespace(
        controls_3d=candidate.controls_3d, gentle_controls=candidate_gentle.gentle_controls)}
    for stage in stages:
        stage_report = {"performance_runs": [], "diagnostic_runs": {}}
        report["stages"][stage] = stage_report
        for index in range(args.repeats):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            pair = {}
            for name in order:
                pair[name] = _run(args, stage, implementations[name],
                                  args.output_dir / stage / f"performance_{index + 1}" / name, False)
            pair["comparison"] = _compare(Path(pair["baseline"]["output"]),
                                           Path(pair["candidate"]["output"]))
            pair["order"] = list(order)
            stage_report["performance_runs"].append(pair)
        for name, implementation in implementations.items():
            stage_report["diagnostic_runs"][name] = _run(
                args, stage, implementation, args.output_dir / stage / "diagnostic" / name, True)
        left, right = [stage_report["diagnostic_runs"][name] for name in implementations]
        stage_report["diagnostic_comparison"] = {
            name: _compare(Path(left["diagnostic_maps"][name]), Path(right["diagnostic_maps"][name]))
            for name in left["diagnostic_maps"]}
        stage_report["diagnostic_comparison"]["final"] = _compare(Path(left["output"]), Path(right["output"]))
        reference = args.reference_t1 if stage == "first" else args.reference_brain
        if reference is not None:
            stage_report["official_reference_sha256"] = _sha256(reference)
            stage_report["official_comparison"] = {
                name: _compare(reference, Path(run["output"]))
                for name, run in stage_report["diagnostic_runs"].items()}
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    unchanged = all(
        pair["comparison"]["numerically_identical"]
        for stage_report in report["stages"].values() for pair in stage_report["performance_runs"])
    unchanged &= all(
        comparison["numerically_identical"]
        for stage_report in report["stages"].values()
        for comparison in stage_report["diagnostic_comparison"].values())
    report["optimization_introduced_no_numeric_or_geometry_change"] = bool(unchanged)
    report["whole_pipeline_speedup"] = None
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "unchanged": bool(unchanged)}))
    if not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
