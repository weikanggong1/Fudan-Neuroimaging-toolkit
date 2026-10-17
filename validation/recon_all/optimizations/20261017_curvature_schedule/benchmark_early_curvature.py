"""两例四侧自产表面早期曲率 ABBA；完整最终八图/统计不变性回归。

prepare 从两份已成功的生产 benchmark 冻结输入、全部 FNIT Python 源码及
独立构建程序 SHA；run 校验显式清单 SHA 后在新目录串行运行 A/B/B/A。
A 为原早期三表面拟合，B 为两表面拟合；均接相同完整 native curv.stats。
CUDA 同步、总四线程；报告设置/复制成本及含读写 API 墙钟，不称整例提速。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback


INPUT_NAMES = ("white.preaparc", "inflated", "smoothwm", "curv", "sulc")
FINAL_MAPS = ("H", "K", "K1", "K2", "BE", "C", "FI", "S")
NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha(path)}


def verify_record(record):
    path = Path(record["path"])
    if not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError("frozen file changed: " + str(path))
    return path


def source_hashes(root):
    root = Path(root).resolve(strict=True)
    values = {str(path.relative_to(root)): sha(path)
              for path in sorted((root / "src/fnit").rglob("*.py"))}
    if not values:
        raise ValueError("source must contain src/fnit/*.py")
    return values


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.code_version) != 40 or any(c not in "0123456789abcdef" for c in args.code_version):
        raise ValueError("code-version must be the actual complete Git commit")
    source = args.source.resolve(strict=True)
    git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                         text=True, capture_output=True, check=False)
    head = git.stdout.strip() if git.returncode == 0 else None
    if head is not None and head != args.code_version:
        raise ValueError("source Git HEAD differs from code-version")
    cases, names = [], set()
    for benchmark in args.benchmark:
        record = file_record(benchmark)
        producer = json.loads(Path(record["path"]).read_text())
        if producer.get("execution") != "complete" or producer.get("exit_code") != 0:
            raise ValueError("input producer benchmark is not complete")
        subject = Path(record["path"]).parent / "subject"
        runtime = file_record(subject / "fnit-native-free-run.json")
        if json.loads(Path(runtime["path"]).read_text()).get("status") != "complete":
            raise ValueError("input producer runtime is not complete")
        case = subject.parent.name
        if case in names:
            raise ValueError("case names must be distinct")
        names.add(case)
        for hemi in ("lh", "rh"):
            cases.append({"case": case, "hemisphere": hemi, "benchmark": record,
                          "runtime": runtime, "producer_code_version": producer.get("code_version"),
                          "raw_t1_sha256": producer.get("input_sha256"),
                          "inputs": {name: file_record(subject / "surf" / f"{hemi}.{name}")
                                     for name in INPUT_NAMES}})
    if len(cases) != 4:
        raise ValueError("exactly two subjects and four hemispheres are required")
    binary = file_record(args.native_program)
    if not os.access(binary["path"], os.X_OK):
        raise PermissionError(binary["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, {"schema": "fnit-early-curvature-abba-v1", "source": str(source),
        "code_version": args.code_version, "actual_git_head": head,
        "source_sha256": source_hashes(source), "native_program": binary,
        "assets": str(args.assets.resolve(strict=True)), "cases": cases,
        "scope": "FNIT-produced fixed stage inputs; not raw-T1 reconstruction"})
    print(json.dumps({"manifest": str(args.output), "sha256": sha(args.output)}))


def compare(first, second):
    import nibabel.freesurfer.io as fsio
    import numpy as np

    a, b = fsio.read_morph_data(str(first)), fsio.read_morph_data(str(second))
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid or mismatched final morph arrays")
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"same_bytes": sha(first) == sha(second),
            "different_elements": int(np.count_nonzero(a != b)),
            "max_abs": float(difference.max(initial=0)),
            "p99_abs": float(np.percentile(difference, 99)),
            "mean_abs": float(difference.mean())}


def stats_comparison(first, second):
    """完整非注释行按顺序比较结构和全部数值；注释头差异单列不作数值失败。

    测试前固定：只排除空行和以#开头的注释行，其他非数值文本也须一致
    （空白标准化）。全部可解析数值要求精确相等，无结果驱动的容差调整。
    原始字节、非注释原文和所有头/数据差异均保留，供严格复现诊断。
    """
    documents = [Path(path).read_text() for path in (first, second)]
    headers, data = [], []
    for document in documents:
        lines = [line.strip() for line in document.splitlines() if line.strip()]
        headers.append([line for line in lines if line.startswith("#")])
        data.append([line for line in lines if not line.startswith("#")])
    structures = [[" ".join(NUMBER.sub("<number>", line).split()) for line in lines] for lines in data]
    values = [[float(match.group()) for line in lines for match in NUMBER.finditer(line)] for lines in data]
    finite = all(math.isfinite(value) for group in values for value in group) and not any(
        re.search(r"\b(?:nan|inf|infinity)\b", line, re.IGNORECASE) for lines in data for line in lines)
    same_structure = structures[0] == structures[1] and bool(data[0])
    same_values = len(values[0]) == len(values[1]) and bool(values[0]) and finite
    differences = [abs(a - b) for a, b in zip(*values)] if same_values else []
    numeric_equal = same_values and not any(a != b for a, b in zip(*values))
    return {"same_bytes": Path(first).read_bytes() == Path(second).read_bytes(),
            "header_lines_equal": headers[0] == headers[1],
            "reference_header_lines": headers[0], "candidate_header_lines": headers[1],
            "noncomment_rows": [len(rows) for rows in data], "noncomment_text_equal": data[0] == data[1],
            "ordered_structure_equal": same_structure, "numeric_fields": [len(group) for group in values],
            "finite": finite, "numeric_fields_exact": numeric_equal,
            "different_numeric_fields": sum(a != b for a, b in zip(*values)) if same_values else None,
            "max_numeric_abs": max(differences, default=0.0) if same_values else None,
            "different_noncomment_lines": [
                {"row": index + 1, "reference": data[0][index] if index < len(data[0]) else None,
                 "candidate": data[1][index] if index < len(data[1]) else None}
                for index in range(max(map(len, data)))
                if index >= len(data[0]) or index >= len(data[1]) or data[0][index] != data[1][index]],
            "numerical_and_semantic_equal": same_structure and numeric_equal}


def run(args):
    outer_started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest SHA changed")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "fnit-early-curvature-abba-v1":
        raise ValueError("unsupported manifest")
    cases = manifest.get("cases", [])
    identities = {(row["case"], row["hemisphere"]) for row in cases}
    names = {row["case"] for row in cases}
    if len(cases) != 4 or len(names) != 2 or identities != {(name, hemi) for name in names for hemi in ("lh", "rh")}:
        raise ValueError("two distinct subjects and all four hemispheres are required")
    if source_hashes(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("source files or coverage changed")
    native = verify_record(manifest["native_program"])
    for case in cases:
        verify_record(case["benchmark"])
        verify_record(case["runtime"])
        if set(case["inputs"]) != set(INPUT_NAMES):
            raise ValueError("stage input coverage changed")
        for value in case["inputs"].values():
            verify_record(value)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 4:
        raise ValueError("four CPUs required for fixed total thread budget")
    os.sched_setaffinity(0, affinity[:4])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(Path(manifest["source"]) / "src"))
    import torch
    import fnit.recon_all.native_free as pipeline
    from fnit.recon_all.profiling import configure_cuda_allocator, ProcessTreeDeviceSampler
    from fnit.recon_all.thread_budget import thread_budget

    if Path(pipeline.__file__).resolve() != Path(manifest["source"]) / "src/fnit/recon_all/native_free.py":
        raise ValueError("imported source is outside frozen candidate")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be explicit cuda:N")
    allocator = configure_cuda_allocator(str(device), policy="enabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(device)
    torch.empty(1, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    args.output.mkdir(parents=True)
    report_path = args.output / "report.private.json"
    report = {"status": "running", "scope": manifest["scope"],
        "manifest": file_record(args.manifest), "code_version": manifest["code_version"],
        "source_sha256": manifest["source_sha256"], "native_program": manifest["native_program"],
        "script_sha256": sha(__file__), "host": platform.node(), "cpu_affinity": affinity[:4],
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), None),
        "threads": 4, "device": str(device), "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
        "precision": {"matmul_tf32": True, "cudnn_tf32": True, "dtype": "float32", "half_precision": False},
        "allocator": allocator, "setup_seconds": time.perf_counter() - outer_started,
        "order": ["legacy", "deferred", "deferred", "legacy"],
        "comparison_definition_before_test": {
            "morphs": "all final eight maps and four preaparc/inflated maps require exact bytes and decoded arrays",
            "stats": "all ordered noncomment rows: identical nonnumeric structure and exact finite numeric fields; header differences and raw bytes reported separately"},
        "timing_scope": "API includes early validation/read/compute/transfer/write, native process and final write; input copying and comparisons separately included in trial total; setup/JIT costs retained in observed trials",
        "whole_recon_all": "not_run", "metric_equivalence": "not_assessed", "cases": []}
    save(report_path, report)
    try:
        with thread_budget(threads=4) as budget:
            report["thread_budget"] = budget
            for index, case in enumerate(cases):
                hemi = case["hemisphere"]
                row = {"case": case["case"], "hemisphere": hemi, "inputs": case["inputs"], "trials": []}
                report["cases"].append(row)
                for iteration, mode in enumerate(report["order"]):
                    trial_start = time.perf_counter()
                    subject = args.output / f"case_{index}" / f"trial_{iteration}" / "subject"
                    for folder in ("surf", "stats", "scripts"):
                        (subject / folder).mkdir(parents=True)
                    for value in case["inputs"].values():
                        source = verify_record(value)
                        shutil.copyfile(source, subject / "surf" / source.name)
                    copy_seconds = time.perf_counter() - trial_start
                    sampler = ProcessTreeDeviceSampler(device=str(device), parent_pid=os.getpid(), interval=.5)
                    stop = threading.Event()
                    def sample_loop():
                        while not stop.wait(.1):
                            sampler.sample_if_due()
                    sampler.sample_if_due(force=True)
                    thread = threading.Thread(target=sample_loop, daemon=True)
                    thread.start()
                    try:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        tick = time.perf_counter()
                        pipeline._write_principal_curvature_maps(
                            surf=subject / "surf", hemi=hemi, device=str(device),
                            include_smoothwm=mode == "legacy")
                        torch.cuda.synchronize(device)
                        early_seconds = time.perf_counter() - tick
                        native_tick = time.perf_counter()
                        pipeline._run_curvature_stats(binary=native, subject=subject, hemi=hemi,
                            assets=Path(manifest["assets"]), device=str(device), python_only=False)
                        native_seconds = time.perf_counter() - native_tick
                        torch.cuda.synchronize(device)
                        api_seconds = time.perf_counter() - tick
                        memory = {"allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                                  "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device))}
                    finally:
                        stop.set()
                        thread.join()
                        sampler.sample_if_due(force=True)
                        sampler.close()
                    outputs = {f"smoothwm.{name}.crv": subject / "surf" / f"{hemi}.smoothwm.{name}.crv"
                               for name in FINAL_MAPS}
                    outputs.update({f"{name}.{kind}": subject / "surf" / f"{hemi}.{name}.{kind}"
                                    for name in ("white.preaparc", "inflated") for kind in ("H", "K")})
                    outputs["curv.stats"] = subject / "stats" / f"{hemi}.curv.stats"
                    output_records = {name: file_record(path) for name, path in outputs.items()}
                    if iteration == 0:
                        reference = outputs
                    comparisons = {name: compare(reference[name], path)
                                   for name, path in outputs.items() if name != "curv.stats"}
                    comparisons["curv.stats"] = stats_comparison(reference["curv.stats"], outputs["curv.stats"])
                    trial = {"iteration": iteration, "mode": mode, "input_copy_seconds": copy_seconds,
                             "early_seconds_including_io": early_seconds, "native_stats_seconds_including_io": native_seconds,
                             "complete_api_seconds": api_seconds, "output_sha256": output_records,
                             "comparison_to_first_legacy": comparisons, "torch_memory": memory,
                             "process_tree_memory": sampler.report(), "trial_total_seconds": time.perf_counter() - trial_start}
                    row["trials"].append(trial)
                    save(report_path, report)
                row["all_outputs_byte_identical"] = all(
                    comparison["same_bytes"] for trial in row["trials"]
                    for comparison in trial["comparison_to_first_legacy"].values())
                row["all_output_values_and_semantics_equal"] = all(
                    comparison["numerical_and_semantic_equal"] if name == "curv.stats" else
                    comparison["same_bytes"] and comparison["different_elements"] == 0
                    for trial in row["trials"] for name, comparison in trial["comparison_to_first_legacy"].items())
                row["median_complete_api_seconds"] = {
                    mode: statistics.median(trial["complete_api_seconds"] for trial in row["trials"] if trial["mode"] == mode)
                    for mode in ("legacy", "deferred")}
                for value in case["inputs"].values():
                    verify_record(value)
                save(report_path, report)
        if source_hashes(manifest["source"]) != manifest["source_sha256"]:
            raise ValueError("source changed while benchmark was running")
        verify_record(manifest["native_program"])
        report["status"] = "complete" if all(row["all_output_values_and_semantics_equal"] for row in report["cases"]) else "comparison_failed"
        report["outer_wall_seconds"] = time.perf_counter() - outer_started
        save(report_path, report)
        if report["status"] != "complete":
            raise SystemExit(2)
    except BaseException as error:
        report.update(status="failed" if report["status"] == "running" else report["status"],
                      error=repr(error), traceback=traceback.format_exc(),
                      outer_wall_seconds=time.perf_counter() - outer_started)
        save(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("--benchmark", type=Path, nargs=2, required=True)
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--code-version", required=True)
    freeze.add_argument("--native-program", type=Path, required=True)
    freeze.add_argument("--assets", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--manifest", type=Path, required=True)
    measure.add_argument("--manifest-sha256", required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
