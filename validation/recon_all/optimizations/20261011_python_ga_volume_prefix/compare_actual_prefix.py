"""直接比较两份已完成filled的实际自产前段；允许历史整例之后失败。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def compare_actual_prefix(*, installed_target: Path, private_dependency: Path,
        comparator: Path, expected_comparator_sha256: str, baseline_benchmark: Path,
        candidate_benchmark: Path, expected_input_sha256: str,
        output_directory: Path, threads: int = 4) -> dict:
    """同原始T1、双方真实filled检查点→14MRI+LTA直接数据/几何回归。

    Path与两个SHA全部显式提供，output_directory须不存在。复用冻结
    比较器的_volume/_label_dice/read_voxel_lta，只改变本诊断的前置门，
    不调用evaluate_prefix或改变其完整control要求。安装target和私有
    SimpleITK指定实际比较依赖，不读dirty源码；四核预算/CUDA隐藏。
    原样捕获四份producer/runtime字节与SHA，MRI和LTA读前/后及全局
    结束时再校验文件SHA。数值差异保留，完成不代表整体等效。
    返回并写direct.private.json及execution.private.json；失败抛异常，
    不重跑/修补生产、不调用官方程序。所有MRI按自产网格直接比较；
    rawavg在原导入网格，其余conform网格，LTA为VOX_TO_VOX，无重采样。
    """
    started = time.perf_counter()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if threads != 4 or len(os.sched_getaffinity(0)) != 4 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("four assigned CPU cores and CUDA hidden required")
    if sha(comparator) != expected_comparator_sha256:
        raise ValueError("frozen comparator SHA differs")
    if not installed_target.is_dir() or not private_dependency.is_dir():
        raise ValueError("declared installation and private dependency required")
    captured = {}
    producers, runtimes = {}, {}
    for label, path in (("baseline", baseline_benchmark), ("candidate", candidate_benchmark)):
        raw = path.read_bytes()
        producer = json.loads(raw)
        runtime_bytes = (path.parent / "subject/fnit-native-free-run.json").read_bytes()
        runtime = json.loads(runtime_bytes)
        if producer.get("input_sha256") != expected_input_sha256:
            raise ValueError("declared original T1 SHA differs")
        if not any(row.get("name") == "filled_auto_checkpoint" and not row.get("error") for row in runtime["stages"]):
            raise ValueError("both actual continuous filled checkpoints required")
        captured[label+"_producer"] = raw
        captured[label+"_runtime"] = runtime_bytes
        producers[label], runtimes[label] = producer, runtime
    sys.path.insert(0, str(private_dependency))
    sys.path.insert(0, str(installed_target))
    spec = importlib.util.spec_from_file_location("frozen_direct_prefix_dependencies", comparator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import fnit
    import numpy as np
    import nibabel as nib
    import numba
    import SimpleITK
    if not Path(fnit.__file__).resolve().is_relative_to(installed_target.resolve()):
        raise RuntimeError("FNIT did not resolve to declared installation")
    numba.set_num_threads(threads)
    output_directory.mkdir(mode=0o700, parents=True)
    for label, raw in captured.items():
        (output_directory / ("direct."+label+".json")).write_bytes(raw)
    report = dict(schema_version=1, status="running", input_sha256=expected_input_sha256,
        scope="same raw-T1 actual Python prefix regression; historical whole failure is not a completed control",
        baseline_whole_execution_at_capture=producers["baseline"]["execution"],
        baseline_exit_code_at_capture=producers["baseline"].get("exit_code"),
        candidate_whole_execution_at_capture=producers["candidate"]["execution"],
        baseline_code_version=producers["baseline"]["code_version"],
        candidate_code_version=producers["candidate"]["code_version"],
        actual_filled_checkpoints={label: True for label in producers},
        captured_receipt_sha256={label: hashlib.sha256(raw).hexdigest() for label, raw in captured.items()},
        comparator_sha256=sha(comparator), runner_sha256=sha(__file__),
        volumes={}, original_stage_program_called=False, whole_candidate_claim=False,
        overall_metric_equivalence="not_assessed")
    checked = {}
    roots = (baseline_benchmark.parent / "subject", candidate_benchmark.parent / "subject")
    for name in module.VOLUMES:
        paths = tuple(root / "mri" / name for root in roots)
        before = {label: sha(path) for label, path in zip(("reference", "candidate"), paths)}
        images = tuple(nib.load(path) for path in paths)
        arrays = tuple(np.asarray(image.dataobj) for image in images)
        row = module._volume(*paths)
        row.update(file_sha256={label: sha(path) for label, path in zip(("reference", "candidate"), paths)},
            shape_equal=arrays[0].shape == arrays[1].shape,
            dtype_equal=arrays[0].dtype == arrays[1].dtype,
            arrays_equal=bool(np.array_equal(*arrays)),
            reference_finite=bool(np.isfinite(arrays[0]).all()), candidate_finite=bool(np.isfinite(arrays[1]).all()),
            affine_equal=bool(np.array_equal(images[0].affine, images[1].affine)),
            vox2ras_tkr_equal=bool(np.array_equal(images[0].header.get_vox2ras_tkr(), images[1].header.get_vox2ras_tkr())),
            vox2ras_tkr_max_abs=float(np.abs(images[0].header.get_vox2ras_tkr()-images[1].header.get_vox2ras_tkr()).max()),
            zooms_equal=images[0].header.get_zooms() == images[1].header.get_zooms())
        if row["file_sha256"] != before:
            raise RuntimeError("MRI changed during direct regression: "+name)
        row["input_files_unchanged_during_read"] = True
        if row["shape_equal"]:
            row["p99_abs_error"] = float(np.quantile(np.abs(arrays[0].astype(np.float64)-arrays[1].astype(np.float64)), .99))
            if name == "filled.mgz":
                row["per_label"] = module._label_dice(*arrays)
            if name.startswith("wm"):
                masks = tuple(array != 0 for array in arrays)
                total = sum(int(mask.sum()) for mask in masks)
                row["foreground_dice"] = 2*int((masks[0] & masks[1]).sum())/total if total else 1.0
        report["volumes"][name] = row
        checked.update({path: before[label] for label, path in zip(("reference", "candidate"), paths)})
    paths = tuple(root / "mri/transforms/talairach.lta" for root in roots)
    before = {label: sha(path) for label, path in zip(("reference", "candidate"), paths)}
    matrices = tuple(module.read_voxel_lta(path) for path in paths)
    after = {label: sha(path) for label, path in zip(("reference", "candidate"), paths)}
    if before != after:
        raise RuntimeError("LTA changed during direct regression")
    report["talairach_lta"] = dict(type="VOX_TO_VOX", reference=matrices[0].tolist(),
        candidate=matrices[1].tolist(), different_elements=int(np.count_nonzero(matrices[0] != matrices[1])),
        max_abs=float(np.abs(matrices[0]-matrices[1]).max()), file_sha256=after)
    checked.update({path: before[label] for label, path in zip(("reference", "candidate"), paths)})
    if any(sha(path) != digest for path, digest in checked.items()):
        raise RuntimeError("prefix input changed before diagnostic completion")
    report.update(status="complete_direct_prefix_regression", all_files_unchanged_at_completion=True,
        strict_data_geometry_dtype_equal=all(all(row[key] for key in (
            "arrays_equal", "shape_equal", "dtype_equal", "affine_equal", "vox2ras_tkr_equal", "zooms_equal",
            "reference_finite", "candidate_finite")) for row in report["volumes"].values())
            and report["talairach_lta"]["different_elements"] == 0)
    (output_directory / "direct.private.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    names = ("fnit", "fnit.recon_all.ca_normalize_python", "fnit.recon_all.compare_native_free")
    receipt = dict(status="complete_readonly_direct_prefix_regression", prefix_report_sha256=sha(output_directory / "direct.private.json"),
        comparator_sha256=sha(comparator), runner_sha256=sha(__file__), input_sha256=expected_input_sha256,
        loaded_module_sha256={name: sha(sys.modules[name].__file__) for name in names},
        versions=dict(python=platform.python_version(), nibabel=nib.__version__, numpy=np.__version__,
            numba=numba.__version__, SimpleITK=SimpleITK.Version_VersionString()),
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=threads, gpu_computation=False, gpu_memory_bytes=None,
        self_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        diagnostic_wall_seconds_before_execution_receipt=time.perf_counter()-started,
        whole_pipeline_speedup=None, overall_metric_equivalence="not_assessed")
    (output_directory / "execution.private.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("installed-target", "private-dependency", "comparator", "baseline-benchmark",
                 "candidate-benchmark", "output-directory"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-comparator-sha256", "expected-input-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--threads", type=int, default=4)
    result = compare_actual_prefix(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "prefix_report_sha256")}))


if __name__ == "__main__":
    main()
