"""只读收集两例已完成的原始T1整例、配对和实际官方评分的JSON与PNG。

fnit_directory为已授权计算环境的统一FNIT目录，output_directory为新的
私有检查点目录；本函数不运行或修改生产算法，不复制MRI/权重/许可证。
收集后必须使用export_receipts脱敏，再向公开仓库传输。运行未完成、
源码绑定不同、输出缺失或目的目录存在均报错，原始文件不改。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PRODUCER = "recon_e2e_gcsa_remesh_79a41cdd_20261010_v1"
COMMIT = "79a41cdda2397525a8952dd5af303372b352916c"


def collect_receipts(*, fnit_directory: Path, output_directory: Path) -> dict:
    """返回完整私有收据SHA清单；数值单位与坐标约定保持原报告。

    两个Path参数均必填，没有默认主机或账户；返回files/source_commit/
    collector_sha256。每例检查实际exit0、runtime complete、138输出与
    生产网格passed；官方/自产配对必须分别complete，不能用文件存在
    代替完成。图片只复制字节；JSON后续脱敏须保留数值类型和值。
    """
    if output_directory.exists():
        raise FileExistsError(output_directory)
    workspace = fnit_directory / "workspaces" / PRODUCER
    source = json.loads((workspace / "source.json").read_text())
    provenance_file = fnit_directory / "runs" / PRODUCER / "PRODUCER_SOURCE_PROVENANCE.json"
    provenance = json.loads(provenance_file.read_text())
    for key in ("git_commit", "source_archive_sha256", "source_archive_bytes"):
        if source[key] != provenance[key]:
            raise ValueError("producer provenance differs: " + key)
    if source["git_commit"] != COMMIT:
        raise ValueError("producer commit differs")
    archive = workspace / "source.tar"
    if (archive.stat().st_size != source["source_archive_bytes"] or
            hashlib.sha256(archive.read_bytes()).hexdigest() != source["source_archive_sha256"]):
        raise ValueError("actual source archive differs")
    files: list[Path] = []
    for case in ("06", "07"):
        raw = fnit_directory / "runs" / PRODUCER / f"sub{case}-candidate"
        benchmark = json.loads((raw / "benchmark.json").read_text())
        runtime = json.loads((raw / "subject/fnit-native-free-run.json").read_text())
        if (benchmark["code_version"].split(":")[0] != COMMIT[:8] or
                source["source_archive_sha256"] not in benchmark["code_version"] or
                len(benchmark["source_sha256"]) != provenance["source_py_count"]):
            raise ValueError("benchmark source version differs: " + case)
        for relative, expected_sha in benchmark["source_sha256"].items():
            file = workspace / "src/fnit" / relative
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected_sha:
                raise ValueError("frozen source changed: " + relative)
        if (benchmark["exit_code"] != 0 or runtime["status"] != "complete" or
                benchmark["output_completeness"]["present"] != 138 or
                benchmark["grid_quality"]["status"] != "passed"):
            raise ValueError("raw T1 producer did not pass execution/completeness/mesh: " + case)
        for role, filename in (("pair", "pair.json"), ("official", "evaluation.json")):
            folder = fnit_directory / "runs" / f"evaluate_gcsa_remesh_79a41cdd_sub{case}_{role}_20261010_v1"
            receipt = json.loads((folder / filename).read_text())
            if receipt["status"] != "complete":
                raise ValueError("evaluation unfinished: " + case + "/" + role)
            bound = receipt["source_sha256"]["candidate"] if role == "pair" else receipt["generator_source_sha256"]
            if bound != benchmark["source_sha256"]:
                raise ValueError("evaluation source binding differs: " + case + "/" + role)
            files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".json", ".png"))
            files.append(folder.with_name(folder.name + "_supervisor.json"))
        files.extend((raw / "benchmark.json", raw / "subject/fnit-native-free-run.json"))
        files.extend((raw / "subject/scripts").glob("*.json"))
    files.extend((workspace / "source.json", provenance_file))
    launch_files = [fnit_directory / "logs" / (PRODUCER + ".sh")]
    if hashlib.sha256(launch_files[0].read_bytes()).hexdigest() != provenance["launch_script_sha256"]:
        raise ValueError("producer launch script changed")
    launch_files.extend(fnit_directory / "logs" / f"evaluate_gcsa_remesh_79a41cdd_sub{case}_{role}_20261010_v1.sh"
                        for case in ("06", "07") for role in ("pair", "official"))
    launch = {str(p.relative_to(fnit_directory)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                   "text": p.read_text()} for p in launch_files}
    for p in files:
        if not p.is_file() or p.is_symlink():
            raise ValueError("missing or symlink receipt: " + str(p))
    output_directory.mkdir(parents=True)
    hashes = {}
    for p in sorted(set(files)):
        relative = p.relative_to(fnit_directory)
        target = output_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    result = {"scope": "actual completed raw T1 producers and separately executed diagnostics; no MRI",
              "source_commit": COMMIT, "files": hashes,
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_directory / "LAUNCH_SCRIPTS.json").write_text(json.dumps(launch, indent=2) + "\n")
    (output_directory / "COLLECTION_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    report = collect_receipts(fnit_directory=args.fnit_directory, output_directory=args.output_directory)
    print(json.dumps({"source_commit": report["source_commit"], "files": len(report["files"])}))
