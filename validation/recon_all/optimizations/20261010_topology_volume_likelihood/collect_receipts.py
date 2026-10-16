"""只导出本子项的公开数值/PNG；不读取或传输MRI、NPZ、原源码/二进制。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(value):
    if isinstance(value, str):
        # 原命令参数不变；只隐藏绝对存储前缀/主机名，不更改数值。
        return re.sub(r"/(?:cpfs[^/\s]*|cwStorage|home|mnt|tmp)/[^\s'\"\[\]]*", "[PRIVATE_PATH]", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {sanitize(key): "[SAME_AUTHORIZED_HOST]" if key == "host" else sanitize(item) for key, item in value.items()}
    return value


def numeric_leaves(value, prefix=""):
    if value is None or isinstance(value, (bool, int, float)):
        return [(prefix, type(value).__name__, value)]
    if isinstance(value, list):
        return sum((numeric_leaves(item, prefix + "/" + str(index)) for index, item in enumerate(value)), [])
    if isinstance(value, dict):
        return sum((numeric_leaves(item, prefix + "/" + key) for key, item in value.items()), [])
    return []


def collect(*, fnit_root: Path, output: Path) -> dict:
    """固定FNIT任务路径→逐数值保真的公开JSON和授权公开T1衍生PNG。

    fnit_root是服务器统一私有FNIT根，output必须不存在。输入仅本次
    candidate/launch、原生repeat/build、已授权公开影像的来源manifest。
    原始T1的producer绑定与全部自产输入SHA在服务器核验，文件不拷出。返回清单含原JSON
    SHA/publicSHA、PNG实际SHA/size、完整源码与代码版本及叶级保真结果。
    来源状态/SHA不符、意外私有字符串、非JSON/PNG载荷或目标已存在报错。
    """
    if output.exists():
        raise FileExistsError("fresh public export required")
    run = fnit_root / "runs/topology_volume_likelihood_20261010_v2"
    native_run = fnit_root / "runs/topology_source_volume_observer_20261010_v1"
    source = fnit_root / "workspaces/topology_volume_likelihood_20261010_v2"
    paths = {
        "report.json": run / "candidate/report.private.json",
        "launch.json": run / "LAUNCH_RECEIPT.private.json",
        "native_repeat.json": native_run / "receipt.private.json",
        "native_build.json": fnit_root / "workspaces/topology_source_volume_observer_20261010_v1/build.private.json",
        "input_manifest.json": fnit_root / "workspaces/recon_topology_source_mri_edges_20261010_v3/INPUT_MANIFEST.json",
    }
    raw_objects = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        raw_objects[name] = (raw, json.loads(raw))
    report = raw_objects["report.json"][1]
    if report["status"] != "complete_frozen_unmri_volume_likelihood_regression" or not all(report[key] for key in ["strict_source_fields_and_feedback_passed", "candidate_repeat_exact", "native_full_capture_repeat_exact"]):
        raise ValueError("actual complete same-input gates required")
    if (run / "candidate.exitcode").read_text().strip() != "0" or (run / "contracts.exitcode").read_text().strip() != "0":
        raise ValueError("actual candidate/contracts successful exit required")
    native = raw_objects["native_repeat.json"][1]
    if len(native["references"]) != 2 or any(row["exit_code"] != 0 or row["files_sha256"]["volume.bin"] != report["capture_sha256"] for row in native["references"]):
        raise ValueError("both native processes must bind the complete frozen capture")
    if report["native_source_receipt_sha256"] != sha(raw_objects["native_build.json"][0]):
        raise ValueError("candidate native build provenance mismatch")
    manifest = raw_objects["input_manifest.json"][1]
    # Actual benchmark used the v2 subject copy; its decoded fields were not
    # regenerated from this newer manifest. Prove each original file is identical.
    native_inputs = fnit_root / "workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject"
    input_verified = {name: sha((native_inputs / name).read_bytes()) == expected for name, expected in manifest["inputs"].items()}
    if not all(input_verified.values()):
        raise ValueError("actual native input copy differs from declared producer")
    producer_path = fnit_root / "runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub07-candidate/benchmark.json"
    producer_raw = producer_path.read_bytes(); producer = json.loads(producer_raw)
    if producer["input_sha256"] != "59ef7bed60d4db64d56d947ebed2ef62a9257c26fe879848fe24d10a490f74fc":
        raise ValueError("public raw T1 source binding changed")
    provenance = {"dataset": "OpenNeuro ds000114", "participant": "sub-07", "session": "ses-test", "hemisphere": "lh",
        "producer_code_version": producer["code_version"], "producer_source_sha256": producer["source_sha256"],
        "producer_runtime_status": producer["runtime_status"], "producer_exit_code": producer["exit_code"],
        "producer_benchmark_original_sha256": sha(producer_raw), "raw_t1_sha256": producer["input_sha256"],
        "native_actual_copy_inputs_verified": input_verified, "actual_inputs_sha256": manifest["inputs"],
        "scope": "real FNIT-produced first defect from real FNIT surface; frozen source-operator diagnostic only"}
    output.mkdir(parents=True)
    entries = []
    for name, (raw, original) in raw_objects.items():
        public = sanitize(original)
        if numeric_leaves(original) != numeric_leaves(public):
            raise ValueError("export changed numeric/bool/null leaves or locations")
        data = (json.dumps(public, indent=2) + "\n").encode()
        if re.search(rb"(?:/cpfs|/data/shared|/(?:home|mnt)/|(?:\d{1,3}\.){3}\d{1,3})", data):
            raise ValueError("public private-string scan failed")
        (output / name).write_bytes(data)
        entries.append({"path": name, "original_sha256": sha(raw), "published_sha256": sha(data),
            "size": len(data), "numeric_bool_null_preserved": True,
            "numeric_bool_null_leaves": len(numeric_leaves(original))})
    image = run / "candidate/aggregate.png"; image_raw = image.read_bytes()
    if not image_raw.startswith(b"\x89PNG\r\n\x1a\n") or sha(image_raw) != report["figure_sha256"]:
        raise ValueError("actual authorized derived PNG binding mismatch")
    shutil.copyfile(image, output / "aggregate.png")
    entries.append({"path": "aggregate.png", "original_sha256": sha(image_raw), "published_sha256": sha(image_raw),
                    "size": len(image_raw), "scope": "authorized public ds000114-derived first-defect volume/feedback diagnostic"})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    contract_log = (run / "contracts.log").read_bytes()
    match = re.search(rb"Ran (\d+) tests in ([0-9.]+)s", contract_log)
    if match is None or not contract_log.endswith(b"OK\n"):
        raise ValueError("actual successful unittest log required")
    contracts = {"exit_code": 0, "tests": int(match[1]), "observed_seconds": float(match[2]),
        "log_sha256": sha(contract_log),
        "numba_cache_shared_between_contracts_and_benchmark": True,
        "first_api_timing_scope": "first benchmark API with available contract cache; not fresh compilation claim",
        "source_module_sha256": sha((source / "src/fnit/recon_all/topology_defect_volume_likelihood.py").read_bytes()),
        "test_sha256": sha((source / "tests/recon_all/test_topology_defect_volume_likelihood.py").read_bytes()),
        "benchmark_sha256": sha((source / "tools/benchmark_recon_topology_volume_likelihood.py").read_bytes())}
    (output / "contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
    for name in ["contracts.json", "provenance.json"]:
        data = (output / name).read_bytes()
        entries.append({"path": name, "published_sha256": sha(data), "size": len(data), "source": "actual server-side readonly receipts"})
    result = {"status": "complete_public_metadata_export", "exporter_sha256": sha(Path(__file__).read_bytes()),
        "entries": entries, "no_mri_npz_weights_license_or_upstream_code_exported": True,
        "declared_scope": "first unMRI volume aggregation and feedback; not full GA or whole reconstruction"}
    (output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(**vars(parser.parse_args()))
