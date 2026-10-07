"""Explicit experimental robust registration CLI, independent of GEMS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib


def parser():
    command = argparse.ArgumentParser(description="Experimental symmetric robust rigid/affine registration")
    command.add_argument("--source", required=True, help="moving one-frame3D image in RAS mm")
    command.add_argument("--target", required=True, help="fixed one-frame3D image in RAS mm")
    command.add_argument("--output-directory", required=True, help="new directory, must not exist")
    command.add_argument("--mode", choices=("rigid", "affine", "rigid-affine"), default="rigid")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--saturation", type=float, default=50.)
    command.add_argument("--iterations-per-level", type=int, default=5)
    command.add_argument("--stop-distance", type=float, default=.01)
    command.add_argument("--no-initialize-translation", action="store_true")
    command.add_argument("--pyramid-min-size", type=int, default=16)
    command.add_argument("--pyramid-max-size", type=int, default=-1)
    command.add_argument("--highres-iterations", type=int, default=-1)
    command.add_argument("--spatial-chunk-size", type=int, default=131072)
    command.add_argument("--memory-budget-gb", type=float, default=20.)
    command.add_argument("--disable-tf32", action="store_true")
    return command


def main(argv=None):
    options = parser().parse_args(argv)
    from .registration import robust_register, robust_rigid_affine
    parameters = dict(device=options.device, saturation=options.saturation,
        iterations_per_level=options.iterations_per_level, stop_distance=options.stop_distance,
        initialize_translation=not options.no_initialize_translation,
        pyramid_min_size=options.pyramid_min_size, pyramid_max_size=options.pyramid_max_size,
        highres_iterations=options.highres_iterations, spatial_chunk_size=options.spatial_chunk_size,
        memory_budget_gb=options.memory_budget_gb, tf32=not options.disable_tf32)
    directory = Path(options.output_directory)
    if directory.exists():
        raise FileExistsError("output-directory must not exist")
    if options.mode == "rigid-affine":
        result = robust_rigid_affine(options.source, options.target, stage_directory=directory, **parameters)
        report = {"status": "completed_native_equivalence_not_assessed", "mode": options.mode,
                  "rigid": result["rigid"].report, "affine": result["affine"].report,
                  "combined_RAS_matrix": result["combined_RAS_matrix"].tolist(),
                  "two_stage_with_save_seconds": result["seconds"]}
    else:
        result = robust_register(options.source, options.target, mode=options.mode, **parameters)
        directory.mkdir(parents=True, exist_ok=False)
        result.transform.save(directory / "transform.lta")
        nib.save(result.header_image, directory / "mapped.header.mgz")
        report = result.report
    # Serialize before creating the report. Nonfinite scientific fields fail
    # visibly; no partial JSON or NaN-as-success artifacts are written.
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with (directory / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
