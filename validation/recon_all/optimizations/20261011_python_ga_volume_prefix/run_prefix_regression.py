"""在实际独立安装包上只读运行冻结前段比较器，保留三份原样收据。"""
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


def run(*, installed_target: Path, private_dependency: Path, comparator: Path,
        reference_benchmark: Path, candidate_benchmark: Path,
        expected_input_sha256: str, output_directory: Path, threads: int = 4) -> dict:
    """全部Path显式提供；同原T1、已完成control与filled检查点→只读诊断。

    output_directory须存在且为空；比较器、安装包及SimpleITK依赖独立
    指定，不导入工作区dirty源码。threads默认4，须与环境及亲和预算一致。
    写prefix.private.json、比较器的三份原样收据和execution.private.json；
    计时含检查、导入、哈希和比较器最终读写，末次执行收据写出由外层
    冷CLI另计。错误抛异常，不修改/补跑生产；显存未采样为None。
    """
    started = time.perf_counter()
    if not output_directory.is_dir() or any(output_directory.iterdir()):
        raise ValueError("existing empty diagnostic directory required")
    if threads != 4 or len(os.sched_getaffinity(0)) != threads:
        raise ValueError("this frozen diagnostic requires four assigned CPU cores")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("read-only comparator requires CUDA hidden")
    for path in (installed_target, private_dependency):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (reference_benchmark, candidate_benchmark):
        producer = json.loads(path.read_bytes())
        if producer.get("input_sha256") != expected_input_sha256:
            raise ValueError("declared original T1 SHA differs")
    sys.path.insert(0, str(private_dependency))
    sys.path.insert(0, str(installed_target))
    spec = importlib.util.spec_from_file_location("frozen_volume_prefix_comparator", comparator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import fnit
    import nibabel
    import numpy
    import numba
    import SimpleITK
    if not Path(fnit.__file__).resolve().is_relative_to(installed_target.resolve()):
        raise RuntimeError("FNIT did not resolve to the declared installation")
    numba.set_num_threads(threads)
    output = output_directory / "prefix.private.json"
    result = module.evaluate_prefix(reference_benchmark=reference_benchmark,
        candidate_benchmark=candidate_benchmark, output=output)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    loaded_names = ("fnit", "fnit.recon_all.ca_normalize_python", "fnit.recon_all.compare_native_free")
    receipt = dict(status="complete_readonly_prefix_regression",
        prefix_report_sha256=sha(output), comparator_sha256=sha(comparator),
        runner_sha256=sha(__file__), input_sha256=expected_input_sha256,
        loaded_module_sha256={name: sha(sys.modules[name].__file__) for name in loaded_names},
        versions=dict(python=platform.python_version(), nibabel=nibabel.__version__,
            numpy=numpy.__version__, numba=numba.__version__, SimpleITK=SimpleITK.Version_VersionString()),
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=threads,
        gpu_computation=False, gpu_memory_bytes=None,
        self_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        diagnostic_wall_seconds_before_execution_receipt=time.perf_counter()-started,
        candidate_whole_execution=result["candidate_whole_execution"],
        whole_pipeline_speedup=None, overall_metric_equivalence="not_assessed")
    (output_directory / "execution.private.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("installed-target", "private-dependency", "comparator", "reference-benchmark",
                 "candidate-benchmark", "output-directory"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--threads", type=int, default=4)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "prefix_report_sha256")}))


if __name__ == "__main__":
    main()
