"""本轮新下载 pilot 的固定真实 TCK 多 atlas 搜索几何 A/B 回归。

输入 --atlas-manifest 为 JSON {"atlas-name": "/absolute/atlas_dwi.nii.gz"}，
--track-metrics NPZ 为同一 TCK 顺序的 weights/lengths/mean_fa；输出仅报告，
不保存原始数据。--baseline-assignment 指冻结提交 assignment.py。
计时含 CPU 图像解码/H2D、矩阵计算与 D2H；轨迹/metrics 读取单列。
AB/BA、GPU allocated/reserved 单列；NVML 自输入读取前采样本进程；外层调度仍监控整例。
必须由协调调度在共享 flock 内启动。此工具是组件回归，不能代表 raw 整例。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import os
import subprocess
import threading

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import EndpointSearchGeometry, build_connectomes
import hashlib


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tracks", "track-metrics", "atlas-manifest", "baseline-assignment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else lambda: None
    spec = importlib.util.spec_from_file_location("frozen_assignment", args.baseline_assignment)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    samples, failures = [], []
    stopped = threading.Event()
    def sample_memory():
        while not stopped.is_set():
            try:
                rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=2).splitlines()
                own = [row.split(",") for row in rows if int(row.split(",")[0].strip()) == os.getpid()]
                samples.append((time.monotonic(), sum(int(row[2].strip()) * 1024**2 for row in own), [row[1].strip() for row in own]))
            except Exception as error:
                failures.append(type(error).__name__)
            stopped.wait(.2)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    if device.type == "cuda": sampler.start()
    start = time.perf_counter()
    tracks = nib.streamlines.load(str(args.tracks), lazy_load=True)
    endpoints = torch.as_tensor(np.asarray([[track[0], track[-1]] for track in tracks.streamlines]),
                                dtype=torch.float32, device=device)
    metrics = np.load(args.track_metrics)
    inputs = {key: torch.as_tensor(metrics[source], device=device)
              for key, source in (("weights", "weights"), ("lengths", "lengths"), ("fa", "mean_fa"))}
    if any(value.shape != (len(endpoints),) for value in inputs.values()):
        raise ValueError("metrics must contain one value per serialized track in the same order")
    endpoint_neq = None
    if "endpoints" in metrics:
        original_endpoints = metrics["endpoints"]
        serialized_endpoints = endpoints.cpu().numpy()
        if original_endpoints.shape != serialized_endpoints.shape:
            raise ValueError("checkpoint endpoints shape differs from serialized TCK")
        endpoint_neq = int(np.count_nonzero(original_endpoints != serialized_endpoints))
        if endpoint_neq:
            raise ValueError("serialized TCK endpoints differ from actual pipeline endpoints")
    sync(); input_seconds = time.perf_counter() - start
    input_peak = ({"allocated": torch.cuda.max_memory_allocated(device), "reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {})
    atlas_paths = json.loads(args.atlas_manifest.read_text())
    report = {"scope": "newraw pilot fixed TCK component; not raw end-to-end", "device": str(device),
              "input_seconds": input_seconds, "tracks": len(endpoints),
              "input_sha256": {"tracks": file_sha256(args.tracks), "metrics": file_sha256(args.track_metrics),
                               "baseline_source": file_sha256(args.baseline_assignment),
                               "candidate_source": file_sha256(Path(__file__).resolve().parents[1] / "src/fnit/connectome/assignment.py")},
              "runs": []}
    # Tolerances declared before computation. Identical operation sequence should be exact.
    report["input_torch_peak_bytes"] = input_peak
    report["checkpoint_endpoint_neq"] = endpoint_neq
    report["metric_dtypes"] = {key: str(value.dtype) for key, value in inputs.items()}
    report["tolerance"] = {"count_neq": 0, "continuous_max_absolute": 0.0}
    outputs = []
    report["torch_version"] = torch.__version__
    report["threads"] = torch.get_num_threads()
    report["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}
    report["allocator_no_cache_environment"] = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
    report["torch_memory_stats_valid"] = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") is None
    for order in (("baseline", "candidate"), ("candidate", "baseline")):
        for arm in order:
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            sync(); total_start = time.perf_counter()
            geometry = None
            results, stages = {}, {}
            for name, entry in atlas_paths.items():
                path = entry["path"] if isinstance(entry, dict) else entry
                sync(); stage_start = time.perf_counter()
                atlas = nib.load(path)
                labels = torch.as_tensor(np.asarray(atlas.dataobj).astype(np.int32), device=device)
                affine = torch.as_tensor(atlas.affine, dtype=torch.float32, device=device)
                kwargs = dict(inputs, radius=4.0, batch_size=args.batch_size)
                if isinstance(entry, dict):
                    kwargs["node_count"] = entry["node_count"]
                if arm == "candidate":
                    if (geometry is None or not torch.equal(geometry.affine, affine)):
                        geometry = EndpointSearchGeometry(affine, radius=4.0, device=device)
                    kwargs["search_geometry"] = geometry
                function = baseline.build_connectomes if arm == "baseline" else build_connectomes
                result = function(endpoints, labels, affine, **kwargs)
                results[name] = {k: v.cpu().numpy() for k, v in result.items()}
                sync(); stages[name] = time.perf_counter() - stage_start
            sync(); elapsed = time.perf_counter() - total_start
            peak = ({"allocated": torch.cuda.max_memory_allocated(device),
                     "reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {})
            row = {"arm": arm, "wall_seconds": elapsed, "atlas_seconds": stages, "torch_peak_bytes": peak}
            report["runs"].append(row)
            outputs.append({"arm": arm, "results": results})
    stopped.set()
    if device.type == "cuda": sampler.join()
    report["nvml"] = {"peak_process_bytes": max((x[1] for x in samples), default=None), "samples": len(samples),
                      "gpu_uuids": sorted({uuid for row in samples for uuid in row[2]}), "failures": failures,
                      "max_interval_seconds": max((samples[i][0]-samples[i-1][0] for i in range(1, len(samples))), default=None)}
    differences = {}
    for run_index, output in enumerate(outputs[1:], start=1):
        comparison = {}
        for name in atlas_paths:
            comparison[name] = {}
            for metric in outputs[0]["results"][name]:
                a, b = outputs[0]["results"][name][metric], output["results"][name][metric]
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                comparison[name][metric] = {"neq": int(np.count_nonzero(a != b)), "max": float(delta.max()),
                                            "p99": float(np.percentile(delta, 99)), "rmse": float(np.sqrt(np.mean(delta**2)))}
        differences[f"baseline0_vs_run{run_index}_{output['arm']}"] = comparison
    report["differences"] = differences
    report["matrix_parity_passed"] = all(v["neq"] == 0 for comparison in differences.values() for d in comparison.values() for v in d.values())
    peak = report["nvml"]["peak_process_bytes"]
    report["memory_budget_passed"] = (device.type != "cuda" or
        (peak is not None and 0 < peak < 20_000_000_000 and len(report["nvml"]["gpu_uuids"]) == 1
         and max(input_peak.values()) < 20_000_000_000
         and all(max(row["torch_peak_bytes"].values()) < 20_000_000_000 for row in report["runs"])))
    report["passed"] = report["matrix_parity_passed"] and report["memory_budget_passed"]
    report["atlas_sha256"] = {name: file_sha256(entry["path"] if isinstance(entry, dict) else entry)
                              for name, entry in atlas_paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]: raise SystemExit("fixed-TCK matrix regression failed")


if __name__ == "__main__": main()
