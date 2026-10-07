#!/usr/bin/env python3
"""Run complete MS-HBM surface, matched-profile or volume API validation.

All input/output paths are provided through a private JSON binding. Public
reports contain hashes, array shapes and anonymous case names. The coordinator
sets CPU affinity and wraps the process with /usr/bin/time for the complete
process clock. No frames, vertices or iteration budgets are shortened here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ENTRY_TIME = time.perf_counter()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--kind", choices=("surface", "surface-cli", "profiles", "volume", "volume-api", "volume-cli"),
                        required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-core-lines", action="store_true",
                        help="Diagnostic line timing; exclude this run from speed comparisons")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    if args.output_dir.exists():
        raise FileExistsError("Benchmark requires a new output directory")
    args.output_dir.mkdir(parents=True)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(args.threads)
    sys.path.insert(0, str(Path(binding["source_root"]) / "src"))
    import nibabel as nib
    import numpy as np
    import scipy
    from scipy.io import loadmat
    import torch
    from fnit.mshbm import load_assets, parcellate, profiles_from_timeseries
    from fnit.mshbm.cli import read_cortex, main as cli_main
    from fnit.mshbm.core import _profile
    from fnit.mshbm.output import save_results
    from fnit.mshbm.volume import project_volume, labels_to_volume, parcellate_volume

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {"schema_version": 1, "case": binding.get("case", "CASE01"),
              "kind": args.kind, "device": args.device, "threads": args.threads,
              "inputs": [], "stages_seconds": {}, "source_sha256": {},
              "profiling_run": args.trace_core_lines,
              "numerics": {"profiles": "float32", "inference_state": "float64",
                           "float16": False, "budgets": {"max_outer": 50,
                               "max_em": 101, "max_m": 300, "max_lambda": 101}},
              "environment": {"hostname": os.uname().nodename, "affinity": sorted(os.sched_getaffinity(0)),
                  "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                  "torch": torch.__version__, "nibabel": nib.__version__,
                  "torch_threads": torch.get_num_threads(),
                  "torch_interop_threads": torch.get_num_interop_threads()}}
    try:
        from threadpoolctl import threadpool_info
        report["environment"]["threadpools"] = threadpool_info()
    except ImportError:
        report["environment"]["threadpools"] = "unavailable"
    actual_package = Path(sys.modules["fnit.mshbm"].__file__).resolve().parent
    expected_package = (Path(binding["source_root"]) / "src/fnit/mshbm").resolve()
    if actual_package != expected_package:
        raise RuntimeError("Actual FNIT import differs from the frozen source binding")
    report["actual_import_matches_frozen_source"] = True
    report["frozen_baseline_commit"] = binding.get("frozen_baseline_commit")
    for name in ("__init__.py", "core.py", "cli.py", "volume.py", "output.py"):
        module = "fnit.mshbm" if name == "__init__.py" else "fnit.mshbm." + name[:-3]
        actual_file = Path(sys.modules[module].__file__).resolve()
        report["source_sha256"]["src/fnit/mshbm/" + name] = sha256(actual_file)
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=binding["source_root"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        report["git_commit"] = None
    for path in (binding["profiles"] if args.kind == "profiles" else
                 binding["timeseries"] if args.kind.startswith("surface") else [binding["volume"]]):
        p = Path(path)
        report["inputs"].append({"size": p.stat().st_size, "sha256": sha256(p)})
    chain_started = time.perf_counter()
    started = time.perf_counter()
    assets = load_assets(binding.get("assets"))
    report["stages_seconds"]["load_assets"] = time.perf_counter() - started
    assets_path = (Path(binding["assets"]) if binding.get("assets") else
                   actual_package / "assets/hcp40_fslr32k_17.npz")
    report["assets_sha256"] = sha256(assets_path)
    mask = assets["cortex_mask"]
    w, c = binding.get("w", 200.0), binding.get("c", 50.0)
    report["w"], report["c"] = w, c

    def clock(name, function, *positional, **named):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        value = function(*positional, **named)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        report["stages_seconds"][name] = time.perf_counter() - start
        return value

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.kind.endswith("-cli"):
        observed_frames = []
        for index, path in enumerate(binding["timeseries"] if args.kind == "surface-cli" else [binding["volume"]]):
            if str(path).endswith(".npy"):
                shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
                count = shape[0] if shape[1] in (59412, 64984) else shape[1]
            else:
                shape = nib.load(path).shape
                count = shape[0] if args.kind == "surface-cli" else shape[-1]
            expected = binding["expected_frames"]
            expected = expected[index] if isinstance(expected, list) else expected
            if count != expected:
                raise ValueError("CLI must retain every frame in the private binding")
            observed_frames.append(count)
        cli_arguments = ["--output-dir", str(args.output_dir), "--w", str(w), "--c", str(c)]
        if binding.get("assets"):
            cli_arguments += ["--assets", binding["assets"]]
        if binding.get("censor"):
            cli_arguments += ["--censor", *binding["censor"]]
        if args.kind == "surface-cli":
            cli_arguments += ["--timeseries", *binding["timeseries"]]
        else:
            cli_arguments += ["--volume", binding["volume"], "--left-surface", binding["left_surface"],
                              "--right-surface", binding["right_surface"], "--cortical-mask", binding["cortical_mask"],
                              "--device", args.device, "--frame-chunk", str(binding.get("frame_chunk", 8)),
                              "--max-distance-mm", str(binding.get("max_distance_mm", 3.0))]
        clock("complete_cli", cli_main, cli_arguments)
        provenance = json.loads((args.output_dir / "provenance.json").read_text())
        labels = np.load(args.output_dir / "labels_fslr32k_64984.npy", allow_pickle=False)
        history = provenance["history"]
        report["sessions"] = provenance["sessions"]
        report["complete_frames_from_binding"] = binding["expected_frames"]
        report["observed_input_frames"] = observed_frames
    elif args.kind == "volume-api":
        image = nib.load(binding["volume"])
        if image.shape[-1] != binding["expected_frames"]:
            raise ValueError("Complete time axis does not match the binding")
        labels, history = clock("complete_volume_api", parcellate_volume,
            volume=binding["volume"], left_surface=binding["left_surface"],
            right_surface=binding["right_surface"], cortical_mask=binding["cortical_mask"],
            output_dir=args.output_dir, assets=assets,
            censor=np.loadtxt(binding["censor"][0]) if binding.get("censor") else None,
            w=w, c=c, device=args.device, frame_chunk=binding.get("frame_chunk", 8),
            max_distance_mm=binding.get("max_distance_mm", 3.0))
        report["input_shape"] = list(image.shape)
    else:
        series = None
        if args.kind == "profiles":
            profiles = []
            started = time.perf_counter()
            for path in binding["profiles"]:
                try:
                    raw_profile = np.asarray(loadmat(path)["profile_mat"])
                except NotImplementedError:
                    import h5py
                    with h5py.File(path, "r") as handle:
                        raw_profile = np.asarray(handle["profile_mat"]).T
                if raw_profile.shape != (64984, 1483):
                    raise ValueError("Official profile must retain all 64984 vertices and 1483 seeds")
                profile = raw_profile[mask].astype(np.float32)
                profile -= profile.mean(axis=1, keepdims=True)
                profile /= np.maximum(np.sqrt(np.einsum("ij,ij->i", profile, profile,
                                                        optimize=True))[:, None], 1e-20)
                profiles.append(profile)
            report["stages_seconds"]["load_normalize_profiles"] = time.perf_counter() - started
        elif args.kind == "volume":
            series = clock("read_project_volume", project_volume, binding["volume"],
                           binding["left_surface"], binding["right_surface"], assets=assets,
                           device=args.device, frame_chunk=binding.get("frame_chunk", 8))
            if len(series) != binding["expected_frames"]:
                raise ValueError("Complete time axis does not match the binding")
            censor = np.loadtxt(binding["censor"][0]) if binding.get("censor") else None
            profiles = clock("build_profiles", profiles_from_timeseries, series, assets, censor)
        else:
            sessions, profiles = [], []
            started = time.perf_counter()
            for i, path in enumerate(binding["timeseries"]):
                item = read_cortex(path, mask)
                expected = binding["expected_frames"]
                expected = expected[i] if isinstance(expected, list) else expected
                if len(item) != expected:
                    raise ValueError("Complete time axis does not match the binding")
                censor = np.loadtxt(binding["censor"][i]) if binding.get("censor") else None
                if censor is not None:
                    if censor.shape != (len(item),) or not np.isin(censor, [0, 1]).all():
                        raise ValueError("Invalid censor")
                    item = item[censor.astype(bool)]
                sessions.append(item)
            report["stages_seconds"]["read_cortex"] = time.perf_counter() - started
            series = np.concatenate(sessions)
            started = time.perf_counter()
            if len(sessions) == 1:
                profiles = profiles_from_timeseries(sessions[0], assets)
            else:
                seeds = np.searchsorted(np.flatnonzero(mask), assets["seed_vertices"])
                profiles = [_profile(item, seeds) for item in sessions]
            report["stages_seconds"]["build_profiles"] = time.perf_counter() - started
        line_costs = {}
        last_line = [None, None]
        actual_core = str(Path(sys.modules["fnit.mshbm.core"].__file__).resolve())

        def trace(frame, event, argument):
            if frame.f_code.co_filename != actual_core or frame.f_code.co_name != "parcellate":
                return None
            now = time.perf_counter()
            if last_line[0] is not None:
                value = line_costs.setdefault(str(last_line[0]), {"calls": 0, "seconds": 0.0})
                value["calls"] += 1
                value["seconds"] += now - last_line[1]
            last_line[0] = frame.f_lineno if event == "line" else None
            last_line[1] = now
            return trace

        if args.trace_core_lines:
            sys.settrace(trace)
        try:
            labels, history = clock("parcellate", parcellate, profiles, assets, w=w, c=c)
        finally:
            if args.trace_core_lines:
                sys.settrace(None)
        if args.trace_core_lines:
            report["core_line_profile"] = line_costs
        if series is not None:
            clock("save_results", save_results, args.output_dir, labels, series, mask,
                  censor=censor if args.kind == "volume" else None)
            report["input_shape"] = list(series.shape)
        else:
            clock("save_labels", np.save, args.output_dir / "labels_fslr32k_64984.npy", labels)
        if args.kind == "volume":
            clock("labels_to_volume", labels_to_volume, labels, binding["volume"],
                  binding["left_surface"], binding["right_surface"], binding["cortical_mask"],
                  args.output_dir / "labels_mni.nii.gz", device=args.device,
                  max_distance_mm=binding.get("max_distance_mm", 3.0))
        report["sessions"] = len(profiles)
    report["function_chain_seconds"] = time.perf_counter() - chain_started
    if args.kind not in ("volume-api", "surface-cli", "volume-cli"):
        report["profiles_sha256"] = [hashlib.sha256(np.ascontiguousarray(p).view(np.uint8)).hexdigest()
                                      for p in profiles]
        if binding.get("save_projected_series") and args.kind == "volume":
            np.save(args.output_dir / "projected_series.npy", series)
    report["labels_shape"] = list(labels.shape)
    report["label_counts"] = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    report["history"] = history
    report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.device.startswith("cuda"):
        report["gpu"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                         "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                         "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    report["adapter_entry_seconds"] = time.perf_counter() - ENTRY_TIME
    report["outputs_sha256"] = {p.name: sha256(p) for p in args.output_dir.iterdir()
                               if p.is_file()}
    (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": report["case"], "kind": args.kind,
                      "seconds": report["adapter_entry_seconds"],
                      "outer_iterations": len(history)}))


if __name__ == "__main__":
    main()
