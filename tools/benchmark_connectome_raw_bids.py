"""Measure the real raw-BIDS CLI; never substitute reference stage outputs.

Usage: python tools/benchmark_connectome_raw_bids.py --mode wall --report RUN.json
       --eddy-gp-seed 12345 -- UKBConnectome_pipeline --bids-root BIDS ...

The caller selects the actual FNIT checkout with PYTHONPATH. This script does
not prepend its own repository's src directory. Reports contain private input
paths; anonymize them before publication. See benchmark_connectome_raw_bids.md.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import inspect
import json
import math
import numbers
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch


def json_safe(value):
    """Emit standards-compliant JSON, including failed/unavailable measurements."""
    if isinstance(value, numbers.Integral):
        return int(value) if not isinstance(value, bool) else value
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__.startswith("numpy") and getattr(value, "ndim", None) == 0:
        return json_safe(value.item())
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(json_safe(value), indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cli_value(arguments, name, default=None):
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return default


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("wall", "diagnostic"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eddy-gp-seed", type=int, default=12345)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--gpu-uuid", help="physical target GPU UUID; recommended with CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-sample-interval", type=float, default=0.5)
    parser.add_argument("cli_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    arguments = options.cli_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if arguments and Path(arguments[0]).name == "fnit":
        arguments = arguments[1:]
    if not arguments or arguments[0] != "UKBConnectome_pipeline":
        parser.error("after -- supply [fnit] UKBConnectome_pipeline and its normal arguments")
    if not cli_value(arguments, "--bids-root"):
        parser.error("this tool requires the real --bids-root CLI path")
    if options.mode == "wall" and options.checkpoint_dir is not None:
        parser.error("--checkpoint-dir is diagnostic only; formal wall runs do not export checkpoints")
    if not math.isfinite(options.memory_sample_interval) or options.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be finite and positive")
    if not 1 <= options.eddy_gp_seed <= 2**32 - 1:
        parser.error("--eddy-gp-seed must be between 1 and 2**32-1")
    options.cli_arguments = arguments
    return options


class PeakLedger:
    """Keep maxima across the pipeline's existing allocator peak resets."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.intervals = []
        self.original = torch.cuda.reset_peak_memory_stats

    def capture(self, reason, device=None):
        if not str(self.device).startswith("cuda") or not self.torch.cuda.is_initialized():
            return
        selected = self.device if device is None else device
        def index(value):
            if isinstance(value, int):
                return value
            text = str(value)
            return int(text.split(":", 1)[1]) if ":" in text else self.torch.cuda.current_device()
        if index(selected) != index(self.device):
            return
        try:
            self.intervals.append({
                "reason": reason, "device": str(selected),
                "allocated_bytes": int(self.torch.cuda.max_memory_allocated(selected)),
                "reserved_bytes": int(self.torch.cuda.max_memory_reserved(selected)),
            })
        except Exception as error:
            self.intervals.append({"reason": reason, "device": str(selected),
                                   "error": f"{type(error).__name__}: {error}"})

    def reset(self, *args, **kwargs):
        selected = kwargs.get("device", args[0] if args else None)
        if selected is None and self.torch.cuda.is_initialized():
            selected = self.torch.cuda.current_device()
        self.capture("before_existing_reset", selected)
        return self.original(*args, **kwargs)

    def report(self):
        valid = [item for item in self.intervals if "allocated_bytes" in item]
        # This is one process on the requested single GPU; no interval sum.
        allocated = max((item["allocated_bytes"] for item in valid), default=None)
        reserved = max((item["reserved_bytes"] for item in valid), default=None)
        return {
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "allocated_gb": allocated / 1e9 if allocated is not None else None,
            "allocated_gib": allocated / 2**30 if allocated is not None else None,
            "reserved_gb": reserved / 1e9 if reserved is not None else None,
            "reserved_gib": reserved / 2**30 if reserved is not None else None,
            "intervals": self.intervals,
            "scope": "process PyTorch allocator maxima, collected before each existing reset and at exit; excludes CUDA context and native allocators",
        }


def descendant_pid(pid, root_pid):
    """Test ancestry without enumerating unrelated processes or reading argv."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            pid = int(text[text.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return pid == root_pid


def parse_smi_processes(text):
    records = []
    for line in text.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            records.append((fields[0], int(fields[1]), int(fields[2]) * 2**20))
        except ValueError:
            continue
    return records


class GPUProcessMonitor:
    """Independent NVML/SMI sampler; never initializes or synchronizes CUDA."""

    def __init__(self, torch, device, interval=0.5, gpu_uuid=None):
        self.torch, self.device = torch, device
        self.interval, self.uuid = interval, gpu_uuid
        self.uuid_source = "explicit" if gpu_uuid else None
        self.stop_event = threading.Event()
        self.samples, self.errors = [], []
        self.failed_samples, self.unresolved_samples = 0, 0
        self.thread = None
        self.backend, self.nvml = None, None
        self.root_pid = os.getpid()

    def start(self):
        if not str(self.device).startswith("cuda"):
            self.backend = "not_requested_cpu"
            return
        try:
            self.nvml = importlib.import_module("pynvml")
            self.nvml.nvmlInit()
            self.backend = "pynvml"
        except Exception:
            self.nvml = None
            self.backend = "nvidia-smi" if shutil.which("nvidia-smi") else "unavailable"
        if self.backend != "unavailable":
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _resolve_uuid(self):
        if self.uuid is not None:
            return
        # is_initialized() is a state query. Only read properties after the
        # production pipeline has initialized CUDA itself.
        if self.torch.cuda.is_initialized():
            properties = self.torch.cuda.get_device_properties(self.device)
            value = getattr(properties, "uuid", None)
            if value:
                self.uuid = value.decode() if isinstance(value, bytes) else str(value)
                self.uuid_source = "torch_initialized_device_properties"

    def _processes(self):
        if self.backend == "pynvml":
            handle = self.nvml.nvmlDeviceGetHandleByUUID(self.uuid)
            rows = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            unavailable = getattr(self.nvml, "NVML_VALUE_NOT_AVAILABLE", (1 << 64) - 1)
            return [(self.uuid, int(row.pid), int(row.usedGpuMemory))
                    for row in rows if row.usedGpuMemory != unavailable]
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True, timeout=3)
        return parse_smi_processes(result.stdout)

    def _sample(self):
        self._resolve_uuid()
        if self.uuid is None:
            self.unresolved_samples += 1
            return
        matching = [(pid, size) for uuid, pid, size in self._processes()
                    if uuid == self.uuid and descendant_pid(pid, self.root_pid)]
        self.samples.append({"monotonic_seconds": time.perf_counter(),
                             "bytes": sum(size for _, size in matching),
                             "pids": [pid for pid, _ in matching]})

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception as error:
                self.failed_samples += 1
                if len(self.errors) < 20:
                    self.errors.append(f"{type(error).__name__}: {error}")
            self.stop_event.wait(self.interval)

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=4)
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        gaps = [right["monotonic_seconds"] - left["monotonic_seconds"]
                for left, right in zip(self.samples, self.samples[1:])]
        peak = max((sample["bytes"] for sample in self.samples), default=None)
        return {
            "backend": self.backend, "gpu_uuid": self.uuid,
            "uuid_source": self.uuid_source,
            "sample_interval_seconds": self.interval,
            "max_observed_interval_seconds": max(gaps, default=None),
            "samples": len(self.samples), "errors": self.errors,
            "failed_samples": self.failed_samples,
            "unresolved_device_samples": self.unresolved_samples,
            "observed_span_seconds": (self.samples[-1]["monotonic_seconds"] - self.samples[0]["monotonic_seconds"]
                                      if len(self.samples) > 1 else None),
            "peak_process_tree_bytes": peak,
            "peak_process_tree_gb": peak / 1e9 if peak is not None else None,
            "peak_process_tree_gib": peak / 2**30 if peak is not None else None,
            "status": ("measured" if self.samples else "not_measured"),
            "scope": "simultaneous compute-process memory on the requested GPU for this PID and live descendants; sample maximum, not a continuous bound; excludes unrelated processes",
        }


class Checkpoints:
    """Export only real function results, one file at a time, diagnostic only."""

    def __init__(self, directory):
        self.directory = Path(directory) if directory is not None else None
        self.seconds = 0.0
        self.paths = []
        self.affine = None
        if self.directory is not None:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise FileExistsError(f"checkpoint directory must be empty: {self.directory}")
            self.directory.mkdir(parents=True, exist_ok=True)

    def _write(self, action):
        if self.directory is None:
            return
        started = time.perf_counter()
        try:
            action()
        finally:
            self.seconds += time.perf_counter() - started

    @staticmethod
    def array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value

    def image(self, name, value, affine=None):
        def write():
            import nibabel as nib
            import numpy as np
            geometry = self.affine if affine is None else self.array(affine)
            if geometry is None:
                raise RuntimeError(f"checkpoint {name} lacks the actual image affine")
            array = self.array(value)
            if array.dtype == np.bool_:
                array = array.astype(np.uint8)
            path = self.directory / name
            nib.save(nib.Nifti1Image(array, geometry), path)
            self.paths.append(path)
        self._write(write)

    def arrays(self, name, values):
        def write():
            import numpy as np
            path = self.directory / name
            np.savez(path, **{key: self.array(value) for key, value in values.items()})
            self.paths.append(path)
        self._write(write)

    def core(self, result):
        if self.directory is None:
            return
        self.arrays("geometry.npz", {
            "dwi_affine": result.dwi_affine,
            "five_tissue_affine": result.five_tissue_affine,
            "dwi_to_t1_world": result.dwi_to_t1_world,
        })
        for name, field, affine in (
            ("wm_fod_normalized.nii.gz", "wm_sh", result.dwi_affine),
            ("fa.nii.gz", "fa", result.dwi_affine),
            ("brain_mask.nii.gz", "brain_mask", result.dwi_affine),
            ("five_tissue.nii.gz", "five_tissue", result.five_tissue_affine),
            ("gmwmi.nii.gz", "gmwmi", result.five_tissue_affine),
        ):
            self.image(name, getattr(result, field), affine)
        self.arrays("track_metrics.npz", {
            "weights": result.sift2_weights,
            "lengths": result.tractogram.lengths_mm,
            "mean_fa": result.tractogram.mean_fa,
            "endpoints": result.tractogram.endpoints,
        })

        def write_tracks():
            import nibabel as nib
            import numpy as np
            from nibabel.streamlines.tractogram import TractogramItem
            def data():
                for path in result.tractogram.paths:
                    yield TractogramItem(self.array(path), {}, {})
            tracks = nib.streamlines.LazyTractogram.from_data_func(data)
            tracks.affine_to_rasmm = np.eye(4)
            path = self.directory / "tracks.tck"
            nib.streamlines.save(tracks, path)
            self.paths.append(path)
            transform = self.directory / "dwi_to_t1_world.csv"
            np.savetxt(transform, self.array(result.dwi_to_t1_world), delimiter=",", fmt="%.17g")
            self.paths.append(transform)
        self._write(write_tracks)


class Measure:
    def __init__(self, torch, bids, pipeline, options):
        self.torch, self.bids, self.pipeline, self.options = torch, bids, pipeline, options
        self.device = cli_value(options.cli_arguments, "--device", "cuda:0")
        self.stack = ExitStack()
        self.ledger = PeakLedger(torch, self.device)
        self.checkpoints = Checkpoints(options.checkpoint_dir)
        self.stages, self.preprocessing, self.eddy_seeds = {}, [], []
        self.qc, self.external_commands = {}, []
        self.corrected_dwi = None
        self.selected_inputs = {}

    def synchronize(self):
        if self.device.startswith("cuda") and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize(self.device)

    def timed(self, name, function, after=None):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            self.synchronize()
            started = time.perf_counter()
            status = "completed"
            try:
                result = function(*args, **kwargs)
                self.synchronize()
            except BaseException:
                status = "failed"
                raise
            finally:
                self.stages.setdefault(name, []).append({
                    "seconds_inclusive": time.perf_counter() - started,
                    "status": status,
                })
                self.ledger.capture("stage_exit/" + name)
            if after is not None:
                after(result, args, kwargs)
            return result
        return measured

    def _after_stage(self, name, function, result, args, kwargs):
        cp = self.checkpoints
        if cp.directory is None:
            return
        if name == "_image" and self.corrected_dwi is not None and Path(args[0]).resolve() == self.corrected_dwi:
            cp._write(lambda: setattr(cp, "affine", cp.array(result[1])))
        elif name == "_gradients":
            cp.arrays("gradients.npz", {"bvals": result[0], "bvecs_world": result[1]})
        elif name == "estimate_mrtrix_dhollander":
            bound = inspect.signature(function).bind_partial(*args, **kwargs).arguments
            cp.image("response_mask.nii.gz", bound["brain_mask"])
            cp.arrays("response.npz", dict(zip(("shells", "wm", "gm", "csf"), result[:4])))
            for key in ("voxels_sfwm", "voxels_gm", "voxels_csf", "safe_mask"):
                if key in result[4]:
                    cp.image("response_" + key + ".nii.gz", result[4][key])
        elif name == "fit_mrtrix_msmt_csd":
            bound = inspect.signature(function).bind_partial(*args, **kwargs).arguments
            cp.image("fod_mask.nii.gz", bound["mask"])
            for key, array in zip(("wm", "gm", "csf"), result):
                cp.image(key + "_raw.nii.gz", array)
        elif name == "normalise_mrtrix_three_tissue":
            bound = inspect.signature(function).bind_partial(*args, **kwargs).arguments
            cp.image("normalise_mask.nii.gz", bound["mask"])

    def __enter__(self):
        try:
            self.stack.enter_context(patch.object(self.torch.cuda, "reset_peak_memory_stats", self.ledger.reset))
            original_eddy = self.bids.TorchEDDY.run
            @functools.wraps(original_eddy)
            def seeded_eddy(instance, *args, **kwargs):
                if kwargs.get("gp_seed") is None:
                    kwargs["gp_seed"] = self.options.eddy_gp_seed
                self.eddy_seeds.append(int(kwargs["gp_seed"]))
                result = original_eddy(instance, *args, **kwargs)
                self.qc["eddy"] = json_safe(result.qc)
                return result
            eddy = (self.timed("eddy_run_with_save", seeded_eddy)
                    if self.options.mode == "diagnostic" else seeded_eddy)
            self.stack.enter_context(patch.object(self.bids.TorchEDDY, "run", eddy))
            original_prepare = self.bids.prepare_bids_connectome
            @functools.wraps(original_prepare)
            def prepare(*args, **kwargs):
                selected = original_prepare(*args, **kwargs)
                self.preprocessing.append(dict(selected.stages))
                self.corrected_dwi = Path(selected.dwi).resolve()
                self.selected_inputs = {name: str(getattr(selected, name)) for name in
                    ("dwi", "bvals", "bvecs", "freesurfer_subject_dir")}
                return selected
            prepare_wrapper = (self.timed("raw_preparation_inclusive", prepare)
                               if self.options.mode == "diagnostic" else prepare)
            self.stack.enter_context(patch.object(self.bids, "prepare_bids_connectome", prepare_wrapper))
            if self.options.mode == "diagnostic":
                self._diagnostic()
            return self
        except BaseException:
            self.stack.close()
            raise

    def _diagnostic(self):
        for name in ("locate_bids_dwi", "stage_bids_dwi", "prepare_ukb_eddy", "_prepare_ap_only"):
            original = getattr(self.bids, name)
            self.stack.enter_context(patch.object(self.bids, name, self.timed(name, original)))
        original_topup = self.bids.run_ukb_topup
        def topup_after(result, args, kwargs):
            self.qc["topup"] = json_safe(result[0].qc)
            self.qc["topup_input_preparation"] = json_safe(result[1])
        self.stack.enter_context(patch.object(self.bids, "run_ukb_topup",
            self.timed("topup_prepare_solve_save", original_topup, topup_after)))
        for name in (
            "_image", "_gradients", "mean_bzero", "_bet_on_dwi_grid", "_scalar_on_grid",
            "dwi2mask_legacy", "maskfilter_six_connected", "fit_mrtrix_dhollander_tensor",
            "freesurfer_five_tissue", "gmwmi_from_five_tissue", "_registration",
            "estimate_mrtrix_dhollander", "fit_mrtrix_msmt_csd", "normalise_mrtrix_three_tissue",
            "probabilistic_tractography", "estimate_sift2_weights", "sample_streamline_mean_precise",
            "fs_aparc_atlas", "fs_aparc_a2009s_atlas", "native_annotation_to_t1",
            "schaefer_to_t1", "glasser_to_t1", "synthmorph_tian_to_t1", "fnirt_tian_to_t1",
            "combine_cortical_tian", "resample_labels_nearest", "build_connectomes",
        ):
            if not hasattr(self.pipeline, name):
                continue  # The same tool can measure older and newer FNIT checkouts.
            original = getattr(self.pipeline, name)
            after = functools.partial(self._after_stage, name, original)
            self.stack.enter_context(patch.object(self.pipeline, name, self.timed(name, original, after)))
        original_core = self.pipeline.UKBConnectome_pipeline.__call__
        def core_after(result, args, kwargs):
            self.checkpoints.core(result)
        self.stack.enter_context(patch.object(self.pipeline.UKBConnectome_pipeline, "__call__",
            self.timed("connectome_core_inclusive", original_core, core_after)))
        original_run = subprocess.run
        def external(*args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            words = shlex.split(command) if isinstance(command, str) else command
            if not words or Path(str(words[0])).name != "recon-all":
                return original_run(*args, **kwargs)
            started = time.perf_counter()
            record = {"command": list(map(str, words)), "status": "failed"}
            try:
                result = original_run(*args, **kwargs)
                record.update(status="completed" if result.returncode == 0 else "failed",
                              returncode=result.returncode)
                return result
            except subprocess.CalledProcessError as error:
                record["returncode"] = error.returncode
                raise
            finally:
                record["seconds"] = time.perf_counter() - started
                self.external_commands.append(record)
        self.stack.enter_context(patch.object(subprocess, "run", external))

    def __exit__(self, *exception):
        self.ledger.capture("run_exit")
        return self.stack.__exit__(*exception)


def source_provenance(torch):
    import fnit
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name == "fnit" or name.startswith("fnit."):
            if path and Path(path).is_file():
                loaded[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    package = Path(fnit.__file__).resolve().parent
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(package), *arguments],
                                capture_output=True, check=False)
        return result.stdout if result.returncode == 0 else None
    git_root = git("rev-parse", "--show-toplevel")
    tracked = git("ls-files", "--error-unmatch", "--", "__init__.py") is not None
    commit = git("rev-parse", "HEAD") if tracked else None
    status = git("status", "--porcelain") if tracked else None
    diff = git("diff", "HEAD") if tracked else None
    return {
        "fnit_version": fnit.__version__, "fnit_package": str(package),
        "git_root": git_root.decode().strip() if git_root is not None else None,
        "git_tracks_fnit_init": tracked,
        "git_commit": commit.decode().strip() if commit is not None else None,
        "git_status": status.decode().splitlines() if status is not None else None,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "loaded_fnit_files": loaded,
        "benchmark_sha256": sha256(__file__),
        "python_executable": {"path": sys.executable, "sha256": sha256(sys.executable)},
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "host": platform.node(), "pid": os.getpid(),
        "threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                    **{name: os.environ.get(name) for name in (
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        "cuda_environment": {name: os.environ.get(name) for name in
                             ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "precision_at_exit": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                              "default_dtype": str(torch.get_default_dtype()),
                              "autocast_enabled": torch.is_autocast_enabled()},
    }


def file_record(path):
    path = Path(path)
    record = {"path": str(path.resolve()), "exists": path.is_file()}
    if record["exists"]:
        record.update(size_bytes=path.stat().st_size, sha256=sha256(path))
    return record


def collect_inputs(options, bids, selected):
    """Hash after timing; the cohort controller freezes inputs before execution."""
    arguments = options.cli_arguments
    inputs = {}
    selection = bids.locate_bids_dwi(
        cli_value(arguments, "--bids-root"), subject=cli_value(arguments, "--subject"),
        session=cli_value(arguments, "--session"), run=cli_value(arguments, "--run"),
        acquisition=cli_value(arguments, "--acquisition"), direction=cli_value(arguments, "--direction"),
        t1=cli_value(arguments, "--t1"),
        select_t1=cli_value(arguments, "--freesurfer-subject-dir") is None,
    )
    for name in ("image", "bval", "bvec", "reverse", "reverse_bval", "t1w"):
        path = getattr(selection, name)
        if path is not None:
            inputs["raw/" + name] = file_record(path)
    inputs["raw/dataset_description"] = file_record(selection.root / "dataset_description.json")
    # Include the original inherited JSONs, not only the generated AP/PA JSON.
    from fnit.dmri_pipeline.bids import _applicable
    for name in ("image", "reverse"):
        path = getattr(selection, name)
        if path is not None:
            for sidecar in _applicable(selection.root, path, "json"):
                inputs["raw_metadata/" + sidecar.relative_to(selection.root).as_posix()] = file_record(sidecar)
    for name, path in selected.items():
        if name != "freesurfer_subject_dir":
            inputs["prepared/" + name] = file_record(path)
    if selected.get("freesurfer_subject_dir"):
        root = Path(selected["freesurfer_subject_dir"])
        for folder in ("mri", "surf", "label"):
            # Hash declared anatomy/atlas inputs; never read license files.
            patterns = {"mri": ("brain.mgz", "aparc*aseg.mgz", "ribbon.mgz"),
                        "surf": ("*.white", "*.pial", "*.sphere.reg"),
                        "label": ("*.annot",)}[folder]
            for pattern in patterns:
                for path in sorted((root / folder).glob(pattern)):
                    inputs["anatomy/" + path.relative_to(root).as_posix()] = file_record(path)
    for flag in ("--mni-template", "--tian-fnirt-coeff", "--brain-mask", "--response-mask",
                 "--fod-mask", "--normalise-mask", "--fa-map", "--dwi-to-t1-world"):
        path = cli_value(arguments, flag)
        if path is not None:
            inputs[flag.removeprefix("--")] = file_record(path)
    for flag, suffixes in (("--synthmorph-weights", (".pt", ".pth", ".h5", ".hdf5")),
                           ("--atlas-templates-dir", (".nii.gz", ".annot", ".txt", ".dlabel.nii"))):
        text = cli_value(arguments, flag)
        if text is None:
            continue
        root = Path(text)
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in sorted(paths):
            if path.is_file() and path.name.endswith(suffixes):
                inputs[flag.removeprefix("--") + "/" + (path.relative_to(root).as_posix() if root.is_dir() else path.name)] = file_record(path)
    return inputs


def cli_atlases(arguments):
    for index, value in enumerate(arguments):
        if value == "--atlas":
            values = []
            for name in arguments[index + 1:]:
                if name.startswith("-"):
                    break
                values.append(name)
            return values
        if value.startswith("--atlas="):
            return [value.split("=", 1)[1]]
    return ["fs-aparc"]


def inspect_outputs(options):
    """Post-run existence/shape checks, not an accuracy-equivalence claim."""
    import csv
    import nibabel as nib
    import numpy as np
    root = Path(cli_value(options.cli_arguments, "--output-dir", "."))
    required = ["five_tissue_dwi_world.nii.gz", "gmwmi_dwi_world.nii.gz", "fa_dwi.nii.gz",
                "brain_mask_dwi.nii.gz", "dwi_to_t1_world.csv", "dataset_description.json", "run_state.json"]
    atlases = {}
    for name in cli_atlases(options.cli_arguments):
        relative = Path("atlases") / name
        required += [str(relative / suffix) for suffix in (
            "connectome_count.csv", "connectome_sift2_fbc.csv", "connectome_mean_length.csv",
            "connectome_mean_fa.csv", "atlas_dwi.nii.gz", "region_labels.csv", "nodes.tsv")]
        checks = {}
        try:
            labels = np.loadtxt(root / relative / "region_labels.csv", delimiter=",", ndmin=1)
            image = nib.load(root / relative / "atlas_dwi.nii.gz")
            checks.update(nodes=int(labels.size), atlas_shape=list(image.shape),
                          atlas_affine=image.affine.tolist())
            with (root / relative / "nodes.tsv").open() as stream:
                nodes = list(csv.DictReader(stream, delimiter="\t"))
            checks["node_table_rows_match"] = len(nodes) == labels.size
            checks["matrices"] = {}
            for matrix_name in ("count", "sift2_fbc", "mean_length", "mean_fa"):
                array = np.loadtxt(root / relative / f"connectome_{matrix_name}.csv", delimiter=",", ndmin=2)
                checks["matrices"][matrix_name] = {
                    "shape": list(array.shape), "finite": bool(np.isfinite(array).all()),
                    "symmetric": bool(np.array_equal(array, array.T)),
                    "shape_matches_nodes": array.shape == (labels.size, labels.size),
                }
        except Exception as error:
            checks["error"] = f"{type(error).__name__}: {error}"
        atlases[name] = checks
    files = {name: file_record(root / name) for name in required}
    complete = all(item["exists"] and item.get("size_bytes", 0) > 0 for item in files.values())
    valid = all("error" not in checks and checks.get("node_table_rows_match", False)
                and all(item["finite"] and item["symmetric"] and item["shape_matches_nodes"]
                        for item in checks.get("matrices", {}).values()) for checks in atlases.values())
    return {"status": "complete" if complete and valid else "incomplete",
            "files": files, "atlases": atlases,
            "scope": "normal CLI output completeness and finite symmetric matrix geometry; does not establish numerical equivalence or scientific validity"}


def run(options):
    output_dir = cli_value(options.cli_arguments, "--output-dir")
    initial = {"output_directory_existed": bool(output_dir and Path(output_dir).exists()),
               "preexisting_state_files": ([str(path) for path in Path(output_dir).rglob("state.json")]
                                           if output_dir and Path(output_dir).exists() else []),
               "preexisting_run_state": bool(output_dir and (Path(output_dir) / "run_state.json").exists())}
    report = {
        "schema_version": 1, "mode": options.mode,
        "cli_arguments": options.cli_arguments,
        "eddy_gp_seed_default": options.eddy_gp_seed,
        "initial_output_state": initial,
        "timing_scope": "FNIT/torch CLI imports, instrumentation installation, real raw-BIDS preparation, official recon-all when needed, actual core and normal CLI output writes; excludes report/source/input hashing, report write, sampler shutdown and any external queue wait",
        "diagnostic_note": "diagnostic stage boundaries synchronize the requested initialized CUDA device; inclusive nested timings cannot be summed; checkpoint exports introduce extra I/O and synchronization and are not formal wall benchmark results",
        "scientific_policy": "actual FNIT functions and outputs; no reference substitution, voxel reduction or precision conversion; missing/None EDDY gp_seed gets the same declared default on both checkouts; explicit non-None seeds remain unchanged",
        "start_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch = measure = monitor = None
    exit_code = 0
    started = time.perf_counter()
    try:
        torch = importlib.import_module("torch")
        cli = importlib.import_module("fnit.cli")
        bids = importlib.import_module("fnit.connectome.bids")
        pipeline = importlib.import_module("fnit.connectome.pipeline")
        measure = Measure(torch, bids, pipeline, options)
        monitor = GPUProcessMonitor(torch, measure.device, options.memory_sample_interval, options.gpu_uuid)
        monitor.start()
        with measure:
            cli.main(options.cli_arguments)
        report["status"] = "completed"
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            report["status"] = "completed"
        else:
            exit_code = int(error.code) if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
            if exit_code == 0:
                exit_code = 1
            report.update(status="failed", exception={"type": type(error).__name__,
                          "message": str(error), "traceback": traceback.format_exc()})
    report["total_runtime_seconds"] = time.perf_counter() - started
    report["end_utc"] = datetime.now(timezone.utc).isoformat()
    report["exit_code"] = exit_code
    report["gpu_process_memory"] = monitor.finish() if monitor is not None else {"status": "not_measured"}
    if measure is not None:
        report.update(preprocessing=measure.preprocessing, actual_eddy_gp_seeds=measure.eddy_seeds,
                      stages=measure.stages, stage_qc=measure.qc,
                      selected_inputs=measure.selected_inputs,
                      official_recon_all_calls=measure.external_commands,
                      cuda_allocator=measure.ledger.report(),
                      checkpoint_export_seconds=measure.checkpoints.seconds)
        report["checkpoints"] = {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
                                 for path in measure.checkpoints.paths if path.is_file()}
        for record in report["official_recon_all_calls"]:
            path = Path(record["command"][0])
            if path.is_file():
                record["executable_sha256"] = sha256(path)
    if torch is not None:
        try:
            report["provenance"] = source_provenance(torch)
        except Exception as error:
            report["provenance_error"] = f"{type(error).__name__}: {error}"
    if measure is not None:
        try:
            report["inputs"] = collect_inputs(options, bids, measure.selected_inputs)
        except Exception as error:
            report["input_hash_error"] = f"{type(error).__name__}: {error}"
        try:
            report["outputs"] = inspect_outputs(options)
        except Exception as error:
            report["output_inspection_error"] = f"{type(error).__name__}: {error}"
    report["hash_timing_scope"] = "source/input/output/checkpoint hashing and output inspections happen after the recorded runtime; the external cohort manifest must freeze source inputs before execution"
    allocated = report.get("cuda_allocator", {}).get("allocated_bytes")
    sampled = report["gpu_process_memory"].get("peak_process_tree_bytes")
    report["memory_budget"] = {
        "limit_bytes": 20_000_000_000, "criterion": "strictly_less",
        "allocator_observed_below_limit": allocated < 20_000_000_000 if allocated is not None else None,
        "process_tree_sampled_below_limit": sampled < 20_000_000_000 if sampled is not None else None,
        "continuous_process_tree_bound_verified": False,
        "note": "allocator and sampled driver peaks have distinct scopes; missing samples or gaps do not prove a strict continuous whole-pipeline memory bound",
    }
    atomic_json(options.report, report)
    return exit_code


def main(argv=None):
    return run(parse_options(argv))


if __name__ == "__main__":
    raise SystemExit(main())
