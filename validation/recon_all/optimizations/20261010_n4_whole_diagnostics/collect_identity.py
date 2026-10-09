"""重新核验N4整例诊断的输入、552项源码、声明资源和官方参考文件SHA。

只生成新目录内的身份收据和原始JSON副本；不复制影像、表面或许可证。
所有生成收据先在服务器脱敏后才可下载。官方参考仅用于事后比较。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_files(*, root: Path, expected: dict) -> dict:
    """核对相对路径→SHA或{sha256,bytes}清单，返回大小与实测SHA。

    root/expected必须显式提供；单位bytes。缺文件、路径越界、大小或SHA
    变化均报错，不改变原始文件。该校验没有独立官方命令。
    """
    rows = {}
    for relative, binding in expected.items():
        component = Path(relative)
        if component.is_absolute() or ".." in component.parts:
            raise ValueError("invalid bound relative path")
        path = root / component
        expected_hash = binding["sha256"] if isinstance(binding, dict) else binding
        actual = sha(path)
        size = path.stat().st_size
        if actual != expected_hash or isinstance(binding, dict) and binding.get("bytes", size) != size:
            raise ValueError("bound file changed: " + relative)
        rows[relative] = {"bytes": size, "sha256": actual}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference-root", "native-bin-dir", "weights-dir", "assets-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    result = {"status": "verifying", "scope": "read-only identity/hash verification, not a rerun or deployment isolation proof",
              "script_sha256": sha(Path(__file__)), "benchmarks": {}}
    manifest = args.reference_root / "TRANSFER_MANIFEST.private.json"
    references = json.loads(manifest.read_text())
    result["reference_manifest_sha256"] = sha(manifest)
    result["reference_files"] = verify_files(root=args.reference_root, expected=references["files"])
    result["reference_cases"] = references["cases"]
    previous_resources = None
    for index, benchmark_path in enumerate(args.benchmark):
        benchmark = json.loads(benchmark_path.read_text())
        runtime_path = benchmark_path.parent / "subject/fnit-native-free-run.json"
        runtime = json.loads(runtime_path.read_text())
        subject = benchmark_path.parent / "subject"
        raw_hash = sha(Path(runtime["input"]))
        if raw_hash != benchmark["input_sha256"]:
            raise ValueError("runtime raw input differs from producer receipt")
        if raw_hash not in {value["input_sha256"] for value in references["cases"].values()}:
            raise ValueError("original T1 not bound in archived official manifest")
        sources = verify_files(root=args.source_root / "src/fnit", expected=benchmark["source_sha256"])
        resource_identity = {"resources": benchmark["resource_sha256"], "native": benchmark["native_program_sha256"]}
        if previous_resources is None:
            result["weights"] = verify_files(root=args.weights_dir, expected=benchmark["resource_sha256"]["weights"])
            result["assets"] = verify_files(root=args.assets_dir, expected=benchmark["resource_sha256"]["assets"])
            result["native_programs"] = verify_files(root=args.native_bin_dir, expected=benchmark["native_program_sha256"])
            previous_resources = resource_identity
        elif resource_identity != previous_resources:
            raise ValueError("producer resources differ across these paired runs")
        key = f"producer_{index}"
        snapshot = args.output / key
        snapshot.mkdir()
        shutil.copyfile(benchmark_path, snapshot / "benchmark.json")
        shutil.copyfile(runtime_path, snapshot / "fnit-native-free-run.json")
        mesh_report = subject / "scripts/mni-mesh-parallel.json"
        if mesh_report.exists():
            shutil.copyfile(mesh_report, snapshot / "mni-mesh-parallel.json")
        result["benchmarks"][key] = {"path": str(benchmark_path), "sha256": sha(benchmark_path),
            "runtime_sha256": sha(runtime_path), "raw_input_sha256": raw_hash,
            "execution": benchmark["execution"], "exit_code": benchmark["exit_code"],
            "runtime_status": runtime["status"], "failed_stage": runtime.get("failed_stage"),
            "code_version": benchmark["code_version"], "source_files": sources,
            "cli_wall_seconds": benchmark["cli_wall_seconds"], "full_harness_seconds": benchmark["full_harness_seconds"],
            "hardware": {key: benchmark[key] for key in ("host", "cpu", "cpu_affinity", "threads", "device")},
            "environment": benchmark["environment"],
            "n4_backend": benchmark["n4_backend"], "n4_execution": benchmark["n4_execution"],
            "precision": runtime["precision"], "output_validation": runtime.get("output_validation"),
            "mesh_validation": runtime.get("mesh_validation"),
            "n4_runtime": runtime.get("n4_runtime"),
            "process_memory": {key: value for key, value in benchmark.get("process_memory", {}).items() if key != "samples"}}
    result.update(status="all_bound_files_verified", wall_seconds=time.perf_counter() - started)
    (args.output / "identity.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "reference_files": len(result["reference_files"]),
                      "sources_per_producer": [len(row["source_files"]) for row in result["benchmarks"].values()],
                      "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
