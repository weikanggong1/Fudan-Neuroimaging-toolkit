"""真实同输入 WM/aseg 子阶段回归；不读取参考结果作为候选流程输入。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np
import torch


class ProcessMemorySampler:
    """显式UUID、进程PID的0.25秒采样；未捕获进程时不把0解释为零显存。"""
    def __init__(self, gpu_uuid, interval=0.25):
        self.gpu_uuid, self.interval = gpu_uuid, interval
        self.pid = os.getpid()
        self.samples, self.errors = [], []
        self.stop_event = threading.Event()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _capture(self):
        try:
            text = subprocess.check_output(["nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                text=True, timeout=5)
            rows = []
            for line in text.splitlines():
                uuid, pid, memory = [part.strip() for part in line.split(",")]
                if uuid == self.gpu_uuid and int(pid) == self.pid:
                    rows.append({"pid": int(pid), "used_bytes": int(memory) * 1048576
                                 if memory.isdigit() else None})
            self.samples.append({"t_seconds": time.perf_counter() - self.started,
                                 "processes": rows})
        except (subprocess.SubprocessError, ValueError, OSError) as error:
            self.errors.append(str(error))

    def _run(self):
        while not self.stop_event.is_set():
            self._capture()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        values = [p["used_bytes"] for row in self.samples for p in row["processes"]
                  if p["used_bytes"] is not None]
        intervals = np.diff([row["t_seconds"] for row in self.samples])
        return {"scope": "benchmark parent process on explicit GPU; native child uses CPU only",
            "gpu_uuid": self.gpu_uuid, "pid": self.pid,
            "requested_interval_seconds": self.interval,
            "max_interval_seconds": float(intervals.max()) if len(intervals) else None,
            "peak_process_bytes": max(values) if values else None,
            "status": "sampled" if values else "no_process_memory_observed",
            "samples": self.samples, "errors": self.errors}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected):
    absolute = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    dice = {}
    for label in np.union1d(np.unique(actual), np.unique(expected)):
        a, b = actual == label, expected == label
        denominator = int(a.sum()) + int(b.sum())
        dice[str(label)] = float(2 * np.count_nonzero(a & b) / denominator) if denominator else 1.
    return {"different_voxels": int(np.count_nonzero(absolute)),
            "max_abs": int(absolute.max()), "p99_abs": float(np.percentile(absolute, 99)),
            "label_dice": dice}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mri-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--native-binary", type=Path)
    p.add_argument("--native-sha256")
    p.add_argument("--native-source", type=Path,
                   help="fixed reference command C++ source; hashed without copying into FNIT")
    p.add_argument("--native-header", type=Path,
                   help="fixed reference cma.h label definition; hashed without copying into FNIT")
    p.add_argument("--complete-only", action="store_true",
                   help="skip previously measured resident-array and late-file microbenchmarks")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    context_tick = time.perf_counter()
    # Initialize the measured target context explicitly before decompression.
    # Failure propagates; this is not an automatic retry or CPU fallback.
    probe = torch.zeros(1, device=device, dtype=torch.uint8)
    torch.cuda.synchronize(device)
    context_seconds = time.perf_counter() - context_tick
    context_free, context_total = torch.cuda.mem_get_info(device)
    del probe
    global cpu_core, cpu_late, gpu, wm_edits_gpu
    from fnit.recon_all import edit_wm_aseg_core_python as cpu_core
    from fnit.recon_all import edit_wm_aseg_late_python as cpu_late
    from fnit.recon_all import edit_wm_aseg_torch as gpu
    from fnit.recon_all import wm_edits_gpu
    torch.cuda.reset_peak_memory_stats(device)
    files = {name: args.mri_dir / f"{name}.mgz" for name in
             ("wm.seg", "brain", "aseg.presurf", "entowm", "wm.asegedit")}
    images = {name: nib.load(str(path)) for name, path in files.items()}
    base = images["wm.seg"]
    if any(image.shape != base.shape or not np.allclose(image.affine, base.affine,
            rtol=0, atol=1e-4) for image in images.values()):
        raise ValueError("real inputs do not share one voxel grid")
    storage_dtypes = {name: str(image.get_data_dtype()) for name, image in images.items()}
    arrays = {name: np.array(image.dataobj, dtype=np.asarray(image.dataobj).dtype.newbyteorder("="),
                           copy=True, order="C") for name, image in images.items()}
    for name in ("aseg.presurf", "entowm"):
        arrays[name] = gpu._integer_label_array(images[name])
    wm, brain, aseg, ento = [arrays[key] for key in ("wm.seg", "brain", "aseg.presurf", "entowm")]
    arrays_cuda = {name: torch.as_tensor(array, device=device) for name, array in arrays.items()}
    wm_gpu, aseg_gpu, ento_gpu = [arrays_cuda[key] for key in ("wm.seg", "aseg.presurf", "entowm")]
    baseline_report = args.mri_dir.parent / "fnit-native-free-run.json"
    native_receipt = json.loads(baseline_report.read_text()) if baseline_report.is_file() else {}
    report = {"scope": "frozen_same_input_substages", "code_commit": args.code_commit,
              "code_commit_role": "frozen baseline before overlay; executed modules bound by sources SHA-256",
              "benchmark_sha256": sha(__file__),
              "host": platform.node(), "cpu": platform.processor() or next(
                  (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name")), "unknown"), "threads": args.threads,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "python_executable": sys.executable, "python": sys.version,
              "device": str(device), "pid": __import__("os").getpid(),
              "context_init_seconds": context_seconds,
              "context_free_bytes": context_free, "context_total_bytes": context_total,
              "gpu": torch.cuda.get_device_name(device), "precision": {
                  "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                  "arithmetic": "uint8/int32 predicates and float32 binary max_pool, no FP16/BF16"},
              "inputs": {name: {"sha256": sha(path), "shape": [int(value) for value in images[name].shape],
                                 "storage_dtype": storage_dtypes[name],
                                 "compute_dtype": str(arrays[name].dtype),
                                 "affine_mm": images[name].affine.tolist()} for name, path in files.items()},
              "sources": {Path(mod.__file__).name: sha(mod.__file__) for mod in
                           (cpu_core, cpu_late, gpu, wm_edits_gpu)}, "stages": {},
              "existing_native_reference": {
                  "receipt_sha256": sha(baseline_report) if baseline_report.is_file() else None,
                  "white_matter_chain": native_receipt.get("white_matter_chain"),
                  "code_commit": native_receipt.get("code_commit", native_receipt.get("git_commit")),
                  "use": "output-only diagnostic; never supplied to candidate algorithm"}}
    report["gpu_uuid"] = subprocess.check_output(["nvidia-smi", f"--id={device.index}",
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    report["reference_source_sha256"] = sha(args.native_source) if args.native_source else None
    report["reference_label_header_sha256"] = sha(args.native_header) if args.native_header else None
    sampler = ProcessMemorySampler(report["gpu_uuid"])
    sampler.start()

    cases = {
        "spackle_superior_mtl": (
            lambda: cpu_core.spackle_wm_superior_to_mtl(wm, aseg),
            lambda: gpu.spackle_wm_superior_to_mtl_torch(wm_gpu, aseg_gpu)),
        "below_hippocampus": (
            lambda: _below_cpu(wm, aseg),
            lambda: gpu.add_aseg_wm_below_hippocampus_torch(wm_gpu, aseg_gpu)),
        "late_no_fill": (
            lambda: cpu_late.apply_late_wm_edits(wm.copy(), aseg, ento, wm, fill_seg_wm=False),
            lambda: gpu.apply_late_wm_edits_torch(wm_gpu, aseg_gpu, ento_gpu, wm_gpu, fill_seg_wm=False)),
        "late_fill_expression": (
            lambda: cpu_late.apply_late_wm_edits(wm.copy(), aseg, ento, wm, fill_seg_wm=True),
            lambda: gpu.apply_late_wm_edits_torch(wm_gpu, aseg_gpu, ento_gpu, wm_gpu, fill_seg_wm=True)),
    }
    for name, (cpu, cuda) in ([] if args.complete_only else cases.items()):
        row = {"measurement": "resident array substage, excludes file IO and transfers",
               "order": ["cpu", "cuda", "cuda", "cpu"]}
        cold = time.perf_counter()
        expected = cpu()
        row["cpu_cold_seconds"] = time.perf_counter() - cold
        torch.cuda.synchronize(device)
        cold = time.perf_counter()
        actual = cuda()
        torch.cuda.synchronize(device)
        row["cuda_cold_seconds"] = time.perf_counter() - cold
        row["comparison"] = compare(actual.cpu().numpy(), expected)
        row["samples_seconds"] = {"cpu": [], "cuda": []}
        for backend in row["order"]:
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = cpu() if backend == "cpu" else cuda()
            torch.cuda.synchronize(device)
            row["samples_seconds"][backend].append(time.perf_counter() - tick)
        row["medians_seconds"] = {key: float(np.median(value)) for key, value in row["samples_seconds"].items()}
        row["speedup"] = row["medians_seconds"]["cpu"] / row["medians_seconds"]["cuda"]
        report["stages"][name] = row
        print(name, json.dumps(row), flush=True)

    # Stage wall time includes reads, exact storage conversion, H2D/D2H, kernels,
    # compression and write. Page cache is warm; this is not raw T1 recon-all.
    file_row = {"measurement": "same-input late edit API including load/transfers/write, warm file cache",
                "order": ["cpu", "cuda", "cuda", "cpu"],
                "samples_seconds": {"cpu": [], "cuda": []}}
    written = {}
    for index, backend in enumerate([] if args.complete_only else file_row["order"]):
        output_file = args.output_dir / f"late-{index}-{backend}.mgz"
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        if backend == "cpu":
            cpu_late.write_late_wm_edits(files["wm.seg"], files["aseg.presurf"],
                files["entowm"], files["wm.seg"], output_file, fill_seg_wm=False)
        else:
            gpu.write_late_wm_edits_torch(files["wm.seg"], files["aseg.presurf"],
                files["entowm"], files["wm.seg"], output_file,
                device=str(device), fill_seg_wm=False)
        torch.cuda.synchronize(device)
        file_row["samples_seconds"][backend].append(time.perf_counter() - tick)
        written[backend] = output_file
    if written:
        left, right = [nib.load(str(written[key])) for key in ("cpu", "cuda")]
        file_row["comparison"] = compare(np.asarray(right.dataobj), np.asarray(left.dataobj))
        file_row["geometry_exact"] = bool(np.array_equal(left.affine, right.affine))
        file_row["dtype_exact"] = left.get_data_dtype() == right.get_data_dtype()
        file_row["medians_seconds"] = {key: float(np.median(value))
                                        for key, value in file_row["samples_seconds"].items()}
        file_row["speedup"] = file_row["medians_seconds"]["cpu"] / file_row["medians_seconds"]["cuda"]
        report["late_file_api"] = file_row
        print("late_file_api", json.dumps(file_row), flush=True)

    # Complete diagnostic using unchanged ordered CPU rules. This is not enabled
    # in production and does not erase the public fixed-input proof guard.
    try:
        tick = time.perf_counter()
        paths, changed = cpu_core.remove_paths_to_cortex(wm, brain, aseg)
        ordered = cpu_core.edit_segmentation_profile(paths, brain, aseg, fill_seg_wm=True)
        core = cpu_core.spackle_wm_superior_to_mtl(ordered, aseg)
        candidate_cpu = cpu_late.apply_late_wm_edits(core, aseg, ento, wm, fill_seg_wm=False)
        report["diagnostic_core_cpu_seconds"] = time.perf_counter() - tick
        candidate_cuda, hybrid_report = gpu.edit_wm_aseg_hybrid_diagnostic(
            wm=wm, brain=brain, aseg=aseg, entowm=ento,
            device=str(device), fill_seg_wm=True)
        nib.save(nib.MGHImage(candidate_cuda, base.affine, header=base.header.copy()),
                 str(args.output_dir / "wm.asegedit.hybrid-diagnostic.mgz"))
        report["diagnostic_full"] = {"path_proof_changed": changed,
            "torch_vs_cpu": compare(candidate_cuda, candidate_cpu),
            "torch_vs_existing_native_output": compare(candidate_cuda, arrays["wm.asegedit"]),
            "hybrid_report": hybrid_report,
            "full_native_equivalence": "same-input WM stage tested; remove-paths proof still required; whole recon not_assessed"}
    except NotImplementedError as error:
        report["diagnostic_full"] = {"status": "unsupported_path", "reason": str(error)}

    if args.native_binary:
        if not args.native_sha256 or sha(args.native_binary) != args.native_sha256:
            raise ValueError("native diagnostic binary does not match the declared reference hash")
        native_api = {"measurement": "complete same-input native vs hybrid API including reads/transfers/write, warm file cache",
            "native_program_sha256": sha(args.native_binary),
            "order": ["native", "hybrid", "hybrid", "native"],
            "samples_seconds": {"native": [], "hybrid": []}}
        outputs = {}
        try:
            for index, backend in enumerate(native_api["order"]):
                output_path = args.output_dir / f"complete-{index}-{backend}.mgz"
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                if backend == "native":
                    with (args.output_dir / f"complete-{index}-native.log").open("w") as stream:
                        subprocess.run([str(args.native_binary), "-keep-in", "-fix-ento-wm",
                            str(files["entowm"]), "3", "255", "255", "-fix-acj",
                            str(files["aseg.presurf"]), "255", "255", "-fill-seg-wm",
                            "-fix-scm-ha", "1", str(files["wm.seg"]), str(files["brain"]),
                            str(files["aseg.presurf"]), str(output_path)],
                            cwd=args.output_dir, stdout=stream, stderr=subprocess.STDOUT, check=True)
                else:
                    gpu.write_wm_asegedit_hybrid_diagnostic(wm_file=files["wm.seg"],
                        brain_file=files["brain"], aseg_file=files["aseg.presurf"],
                        entowm_file=files["entowm"], output_file=output_path,
                        device=str(device), fill_seg_wm=True)
                torch.cuda.synchronize(device)
                native_api["samples_seconds"][backend].append(time.perf_counter() - tick)
                outputs[backend] = output_path
            native_image, hybrid_image = [nib.load(str(outputs[key])) for key in ("native", "hybrid")]
            native_api["comparison"] = compare(np.asarray(hybrid_image.dataobj), np.asarray(native_image.dataobj))
            native_api["native_repeat_comparison"] = compare(
                np.asarray(nib.load(str(args.output_dir / "complete-0-native.mgz")).dataobj),
                np.asarray(nib.load(str(args.output_dir / "complete-3-native.mgz")).dataobj))
            native_api["geometry_exact"] = bool(np.array_equal(native_image.affine, hybrid_image.affine))
            native_api["dtype_exact"] = native_image.get_data_dtype() == hybrid_image.get_data_dtype()
            native_api["medians_seconds"] = {key: float(np.median(value)) for key, value in native_api["samples_seconds"].items()}
            native_api["speedup"] = native_api["medians_seconds"]["native"] / native_api["medians_seconds"]["hybrid"]
        except (NotImplementedError, subprocess.CalledProcessError) as error:
            native_api["status"] = "incomplete"
            native_api["reason"] = str(error)
        report["complete_native_vs_hybrid_api"] = native_api

    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report["cuda_process_sampling"] = {"status": "not_measured",
        "note": "allocated/reserved cover PyTorch tensors, not CUDA context/driver allocation or concurrent unrelated jobs"}
    report["benchmark_total_seconds"] = time.perf_counter() - started
    report["whole_recon_speedup"] = "not_measured"
    native_pair = report.get("complete_native_vs_hybrid_api", {})
    report["official_program_repeatability"] = (
        {"scope": "complete native WM edit command, fixed input/environment/threads",
         "comparison": native_pair["native_repeat_comparison"]}
        if "native_repeat_comparison" in native_pair else
        "not_retested in this run; internal steps have no standalone official CLI")
    report["isolated_deployment"] = "not_verified"
    memory = sampler.finish()
    (args.output_dir / "process_memory.json").write_text(json.dumps(memory, indent=2))
    report["process_memory"] = {key: value for key, value in memory.items() if key != "samples"}
    # Older snapshots kept a not_measured sentinel; keep the compatibility key
    # accurate after a real process sample has been collected.
    report["cuda_process_sampling"] = report["process_memory"]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def _below_cpu(wm, aseg):
    output = wm.copy()
    cpu_core._add_aseg_wm_below_hippocampus(output, aseg)
    return output


if __name__ == "__main__":
    main()
