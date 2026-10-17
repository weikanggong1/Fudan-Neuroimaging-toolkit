"""用既有完整sphere worker验证真实候选源码，共四侧trace，不重复ABBA。

输入为SHA固定的私有plan：已有worker及其SHA、候选完整源码清单、同一份
四侧自产输入清单、已完成v2目录、四份mode=trace/arm=B payload和新输出根。
CUDA、线程与亲和性由外层共用锁launcher固定。复用worker原API/完整轨迹/
FP32和FP64质量判据；不修改生产函数、质量门或已有v2目录。
输出summary.private.json、四份payload/log/report/trace/球面及逐项相等诊断。
坐标为surface RAS毫米；trace用于同输入数值回归，不作为性能样本。
必要文件未完成、SHA/参数不符在计算前报错；worker失败保留现场、不重试。
原软件对应mris_sphere完整单侧步骤，轨迹观测无独立官方CLI。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    """读取明确文件，返回SHA-256；读失败直接传播。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    """读取已有JSON对象，不修改原文件。"""
    return json.loads(Path(path).read_text())


def save(path, value):
    """以同目录临时文件原子写本轮JSON；单位由各字段声明。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


def require(test, message):
    if not test:
        raise ValueError(message)


def bound(path, expected):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "missing/non-plain frozen file: " + str(path))
    require(sha(path) == expected, "frozen SHA differs: " + str(path))
    return path


def prepare_gate(plan):
    """只读验证四侧v2已完成及真实source71参数，不运行球面。"""
    bound(plan["candidate_archive_manifest"], plan["candidate_archive_manifest_sha256"])
    require(sha(plan["python_interpreter"]) == plan["python_interpreter_sha256"],
            "declared interpreter changed")
    worker = bound(plan["worker"], plan["worker_sha256"])
    spec = importlib.util.spec_from_file_location("_same_sphere_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.verify_source(plan["candidate_source"], plan["candidate_manifest"],
                                  plan["candidate_manifest_sha256"])
    require(source["commit"] == plan["candidate_commit"], "actual candidate commit differs")
    inputs = module.verify_inputs(plan["inputs_manifest"], plan["inputs_manifest_sha256"])
    require(len(plan["payloads"]) == len(inputs["cases"]) == 4, "four hemisphere payloads required")
    baseline = read(bound(plan["baseline_manifest"], plan["baseline_manifest_sha256"]))
    require(baseline.get("commit") == plan["control_commit"], "control source identity differs")
    relevant = ("sphere_", "place_surface_normals", "mris_register_average", "mris_register_overlap")
    relevant_names = {name for m in (baseline, source) for name in m["files"]
                      if name.startswith("src/fnit/recon_all/") and Path(name).name.startswith(relevant)}
    changes = {name: {"control_sha256": baseline["files"].get(name),
                      "candidate_sha256": source["files"].get(name)}
               for name in sorted(relevant_names)
               if baseline["files"].get(name) != source["files"].get(name)}
    require(set(changes) == {"src/fnit/recon_all/mris_register_average_numba.py"},
            "sphere dependency changes exceed the declared averaging-buffer regression")
    root = Path(plan["control_root"])
    summary_path = root / "summary.private.json"
    summary = read(summary_path)
    require(summary.get("status") == "complete" and len(summary.get("comparisons", [])) == 4,
            "all four original v2 ABBA+trace groups must complete first")
    require(len(summary.get("runs", [])) == 24 and
            all(r.get("status") == "complete" and r.get("exit_code") == 0 for r in summary["runs"]),
            "a v2 timing/trace child is incomplete")
    controls = {}
    for payload in plan["payloads"]:
        row = payload["row"]
        require(row in inputs["cases"] and payload["source"] == plan["candidate_source"] and
                payload["source_manifest"] == plan["candidate_manifest"] and
                payload["source_manifest_sha256"] == plan["candidate_manifest_sha256"] and
                payload["inputs_manifest"] == plan["inputs_manifest"] and
                payload["inputs_manifest_sha256"] == plan["inputs_manifest_sha256"] and
                payload["mode"] == "trace" and payload["arm"] == "B" and payload["device"] == "cuda:0",
                "candidate worker payload differs from fixed input/parameters")
        identifier = row["case"] + "_" + row["hemisphere"]
        folder = root / identifier / "trace-01-B"
        old_payload_path = folder.parent / "trace-01-B.payload.private.json"
        old_payload = read(old_payload_path)
        old_report_path = folder / "report.private.json"
        old = read(old_report_path)
        require(old_payload["row"] == row and old_payload["mode"] == "trace" and
                old_payload["arm"] == "B" and old_payload["device"] == "cuda:0" and
                old_payload["source_manifest_sha256"] == plan["baseline_manifest_sha256"] and
                old.get("status") == "complete" and old.get("quality", {}).get("passed") is True and
                old.get("benchmark_script_sha256") == plan["worker_sha256"],
                "control is not the original complete v2 B trace")
        wanted = {"finish_device": "cuda:0", "averaging_device": "cuda:0", "normals_device": "cuda:0",
                  "bbox_mode": "source_float", "require_unfolded": True, "overlap_diagnostics": True}
        require(old.get("kwargs") == wanted, "control parameters differ")
        record = next((r for r in summary["runs"] if r.get("worker_report") == str(old_report_path)), None)
        require(record and record["worker_report_sha256"] == sha(old_report_path),
                "control report changed after v2 controller")
        bound(folder / "trace.private.json", old["trace_sha256"])
        bound(folder / (row["hemisphere"] + ".sphere"), old["quality"]["sha256"])
        controls[identifier] = {"folder": str(folder), "payload_sha256": sha(old_payload_path),
                                "report_sha256": sha(old_report_path), "report": old}
    return {"v2_summary_sha256": sha(summary_path), "changed_sphere_dependencies": changes,
            "candidate_python_files": len(source["files"]), "controls": controls}


def run(plan):
    """按plan串行运行四个完整trace；输出新目录，失败保留现场。"""
    os.umask(0o077)
    root = Path(plan["output_root"])
    require(not os.path.lexists(root), "new output root already exists")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["cuda_visible_devices"] and
            sorted(os.sched_getaffinity(0)) == [0, 1, 2, 3], "GPU visibility/CPU affinity differs")
    for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(name) == "4", "four-thread environment required: " + name)
    binding = prepare_gate(plan)
    root.mkdir(parents=True, mode=0o700)
    summary_path = root / "summary.private.json"
    summary = {"status": "running", "scope": "candidate71 versus original v2 B, four complete same-input standard-sphere traces; not a performance ABBA or recon-all",
               "candidate_commit": plan["candidate_commit"], "control_commit": plan["control_commit"],
               "preflight": binding, "runs": [], "comparisons": [],
               "strict_reproduction": "not_assessed", "overall_metric_equivalence": "not_assessed"}
    save(summary_path, summary)
    try:
        for payload in plan["payloads"]:
            row = payload["row"]
            identifier = row["case"] + "_" + row["hemisphere"]
            folder = Path(payload["output"])
            require(folder == root / identifier / "trace-00-B", "payload output escapes fresh root")
            folder.parent.mkdir(parents=True)
            payload_path = folder.parent / "trace-00-B.payload.private.json"
            save(payload_path, payload)
            env = dict(os.environ)
            env.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
            for name in ("NUMBA_CACHE_DIR", "TRITON_CACHE_DIR"):
                cache = root / "cache" / name.lower()
                cache.mkdir(parents=True, exist_ok=True)
                env[name] = str(cache)
            command = [plan["python_interpreter"], "-X", "faulthandler", "-u", plan["worker"],
                       "worker", "--payload", str(payload_path), "--payload-sha256", sha(payload_path)]
            tick = time.perf_counter()
            with (folder.parent / "trace-00-B.log").open("x") as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            report_path = folder / "report.private.json"
            report = read(report_path) if report_path.is_file() else {"status": "failed_before_report"}
            record = {"case": row["case"], "hemisphere": row["hemisphere"], "exit_code": process.returncode,
                      "status": report["status"], "command": command,
                      "worker_report": str(report_path), "worker_report_sha256": sha(report_path) if report_path.exists() else None,
                      "parent_wall_seconds_not_performance_sample": time.perf_counter() - tick}
            summary["runs"].append(record)
            save(summary_path, summary)
            require(process.returncode == 0 and report["status"] == "complete", "candidate trace failed; preserve scene: " + identifier)
            old = binding["controls"][identifier]
            old_folder = Path(old["folder"])
            reference_trace = old_folder / "trace.private.json"
            trace = folder / "trace.private.json"
            bound(reference_trace, old["report"]["trace_sha256"])
            bound(trace, report["trace_sha256"])
            surface = folder / (row["hemisphere"] + ".sphere")
            old_surface = old_folder / surface.name
            bound(surface, report["quality"]["sha256"])
            bound(old_surface, old["report"]["quality"]["sha256"])
            quality_keys = ("sha256", "coordinates_fp32_sha256", "faces_sha256", "vertices", "faces",
                            "ordered_faces_equal_input", "volume_geometry_equal", "all_finite",
                            "fp32_negative_faces", "fp64_negative_faces", "fp64_negative_area_mm2", "structure", "passed")
            same = {key: report["quality"].get(key) == old["report"]["quality"].get(key) for key in quality_keys}
            traces_equal = read(trace) == read(reference_trace)
            output_equal = surface.stat().st_size == old_surface.stat().st_size and sha(surface) == sha(old_surface)
            comparison = {"case": row["case"], "hemisphere": row["hemisphere"], "all_quality_fingerprints_equal": same,
                          "complete_trajectory_equal": traces_equal, "trace_bytes_sha_equal": sha(trace) == sha(reference_trace),
                          "surface_bytes_sha_equal": output_equal, "candidate_quality_passed": report["quality"]["passed"],
                          "strict_reproduction": "passed" if all(same.values()) and traces_equal and output_equal else "difference_detected"}
            summary["comparisons"].append(comparison)
            save(summary_path, summary)
        summary["status"] = "complete"
        summary["strict_reproduction"] = "passed" if all(c["strict_reproduction"] == "passed" for c in summary["comparisons"]) else "difference_detected"
        summary["optimization_regression"] = "not_observed_in_this_fixed_input_test" if summary["strict_reproduction"] == "passed" else "requires_diagnosis"
    except BaseException as error:
        summary.update(status="failed", error=str(error), traceback=traceback.format_exc())
    save(summary_path, summary)
    return 0 if summary["status"] == "complete" else 1


def main():
    """--plan为私有JSON路径，--plan-sha256为预先固定摘要，无默认参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = read(bound(args.plan, args.plan_sha256))
    bound(__file__, plan["runner_sha256"])
    return run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
