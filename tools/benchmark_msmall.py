#!/usr/bin/env python3
"""Compare FNIT MSMAll to saved official spheres using server-local real data.

The private case supplies identical MSMAllInputs, an official configuration,
official_spheres and optional fixed-volume projection parameters. No official
registration executable is called here. MRI, spheres, feature maps and private
manifests stay in the output directory; report.safe.json contains only aggregate
quality, time, memory and software hashes.
"""

import argparse
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def support():
    path = Path(__file__).with_name("benchmark_msmsulc.py")
    spec = importlib.util.spec_from_file_location("msmall_benchmark_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_fixed(case, spheres, inputs, directory, helpers):
    """Compose a 32k warp onto native MSMSulc geometry before projection."""
    import nibabel as nib
    import numpy as np
    directory = Path(directory)
    native = {}
    for hemi, key in (("L", "left"), ("R", "right")):
        vertices, faces = helpers.sphere(spheres[hemi])
        image = nib.load(case["projection"][key]["midthickness"])
        points = next(np.asarray(a.data) for a in image.darrays if a.intent == 1008)
        triangles = next(np.asarray(a.data) for a in image.darrays if a.intent == 1009)
        if len(vertices) == len(points) and np.array_equal(faces, triangles):
            native[hemi] = spheres[hemi]
            continue
        if "native_projection_basis" not in case:
            raise ValueError("32k solver outputs require explicit native_projection_basis L/R spheres")
        executable = shutil.which(str(case["projection"].get("wb_command", "wb_command")))
        if executable is None:
            raise FileNotFoundError("Connectome Workbench is required for native composition")
        native[hemi] = directory / "native_composition" / f"{hemi}.sphere.MSMAll.native.surf.gii"
        native[hemi].parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            executable, "-surface-sphere-project-unproject", case["native_projection_basis"][hemi],
            str(inputs[hemi].source_sphere), str(spheres[hemi]), str(native[hemi]),
        ], check=True, capture_output=True, text=True)
    return helpers.project(case, native, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution", choices=("optimized", "reference"), default="optimized")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warm-repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warm_repeats < 0:
        parser.error("threads must be positive and warm-repeats nonnegative")
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source / "src"))
    import torch
    from fnit.msm import MSMAllConfig, MSMAllInputs, run_msmall, _fastpd_native
    helpers = support()
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    case = json.loads(Path(args.case_json).read_text())
    if set(case["inputs"]) != {"L", "R"} or set(case["official_spheres"]) != {"L", "R"}:
        raise ValueError("both input and official sphere dictionaries require L and R")
    inputs = {hemi: MSMAllInputs(**{
        name: Path(value) if value is not None else None
        for name, value in case["inputs"][hemi].items()
    }) for hemi in ("L", "R")}
    for entry in inputs.values():
        for field in fields(entry):
            value = getattr(entry, field.name)
            if value is not None and not Path(value).is_file():
                raise FileNotFoundError("missing private MSMAll input file")
    configuration = MSMAllConfig.from_file(case["config_file"]) if "config_file" in case else MSMAllConfig()
    selected = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
        torch.cuda.init()
    files = sorted((source / "src/fnit/msm").glob("*.py"))
    files += sorted((source / "src/fnit/msm/_fastpd_src").glob("*"))
    report = {
        "schema_version": 1, "function": "run_msmall",
        "configuration": configuration.to_dict(), "execution": args.execution,
        "source_sha256": {str(p.relative_to(source)): helpers.sha256(p) for p in files if p.is_file()},
        "benchmark_tool_sha256": helpers.sha256(__file__),
        "native_extension_sha256": helpers.sha256(_fastpd_native.__file__),
        "environment": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "device": selected.type, "cpu_threads": args.threads,
                        "gpu_name": torch.cuda.get_device_properties(selected).name if selected.type == "cuda" else None},
        "timing_scope": "paired full configured registration, input reads and sphere/report writes; excludes imports, CUDA initialization, offline precision metrics and BOLD projection",
        "cold_definition": "first registration call after imports and CUDA initialization",
        "runs": [],
    }
    oracle = None
    if "projection" in case:
        began = time.perf_counter()
        oracle = project_fixed(case, case["official_spheres"], inputs,
                               directory / "official_projection", helpers).dtseries
        report["projection_reference"] = {
            "wall_seconds": time.perf_counter() - began,
            "sphere_specific_area_surfaces": True,
        }
    for index in range(args.warm_repeats + 1):
        run_dir = directory / ("cold_0" if index == 0 else f"warm_{index}")
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
            torch.cuda.reset_peak_memory_stats(selected)
        with helpers.load_support().GPUMonitor(torch, selected) as monitor:
            began = time.perf_counter()
            result = run_msmall(inputs, run_dir, device=args.device,
                                config=configuration, execution=args.execution)
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
            elapsed = time.perf_counter() - began
        internal = json.loads((run_dir / "registration_report.json").read_text())
        record = {
            "condition": run_dir.name, "wall_seconds": elapsed,
            "feature_count_by_hemisphere": {hemi: int(internal[hemi]["feature_count"]) for hemi in "LR"},
            "weighted_cost_by_hemisphere": {hemi: bool(internal[hemi]["weighted_cost"]) for hemi in "LR"},
            "peak_allocated_gb": max((internal[hemi]["peak_allocated_gb"] or 0) for hemi in "LR"),
            "sphere_paired": {hemi: helpers.sphere_metrics(
                result[hemi], case["official_spheres"][hemi], inputs[hemi].source_sphere
            ) for hemi in ("L", "R")},
            "registration": helpers.safe_registration_report(internal),
            "whole_gpu_observation": monitor.report(),
        }
        if oracle is not None:
            began = time.perf_counter()
            projected = project_fixed(case, result, inputs, run_dir / "projection", helpers)
            record["projection_wall_seconds"] = time.perf_counter() - began
            record["fixed_bold_cifti_paired"] = helpers.cifti_metrics(projected.dtseries, oracle)
        report["runs"].append(record)
        (directory / "report.safe.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
