"""Single-subject CLI for the standalone CPU robust registration API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def positive_threads(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("threads must be positive")
    return number


def parser():
    command = argparse.ArgumentParser(description="FNIT CPU 刚性／仿射 robust 配准")
    command.add_argument("--source", required=True, help="3D moving MGH/MGZ or NIfTI; RAS mm")
    command.add_argument("--target", required=True, help="3D fixed MGH/MGZ or NIfTI; RAS mm")
    command.add_argument("--output-directory", required=True, help="new output directory; must not exist")
    command.add_argument("--mode", choices=("rigid", "affine", "rigid-affine"), default="rigid")
    command.add_argument("--saturation", type=float, default=50.)
    command.add_argument("--iterations-per-level", type=int, default=5)
    command.add_argument("--stop-distance", type=float, default=.01)
    command.add_argument("--no-initialize-translation", action="store_true")
    command.add_argument("--pyramid-min-size", type=int, default=16)
    command.add_argument("--pyramid-max-size", type=int, default=-1)
    command.add_argument("--highres-iterations", type=int, default=-1)
    command.add_argument("--spatial-chunk-size", type=int, default=131072)
    command.add_argument("--memory-budget-gb", type=float, default=20.)
    command.add_argument("--threads", type=positive_threads, help="optional Torch intra-op threads; omitted preserves caller settings")
    return command


def main(argv=None):
    options = parser().parse_args(argv)
    directory = Path(options.output_directory)
    if directory.exists():
        raise FileExistsError("output-directory must not exist")
    if options.threads is not None:
        import torch
        torch.set_num_threads(options.threads)
    from .cpu import load_cpu_registration
    parameters = dict(
        saturation=options.saturation,
        iterations_per_level=options.iterations_per_level,
        stop_distance=options.stop_distance,
        initialize_translation=not options.no_initialize_translation,
        pyramid_min_size=options.pyramid_min_size,
        pyramid_max_size=options.pyramid_max_size,
        highres_iterations=options.highres_iterations,
        spatial_chunk_size=options.spatial_chunk_size,
        memory_budget_gb=options.memory_budget_gb,
    )
    with load_cpu_registration() as registration:
        if options.mode == "rigid-affine":
            result = registration.cpu_robust_rigid_affine(
                options.source, options.target, stage_directory=directory, **parameters)
            report = dict(
                mode=options.mode, rigid=result["rigid"].report,
                affine=result["affine"].report,
                combined_RAS_matrix=result["combined_RAS_matrix"].tolist(),
                two_stage_with_save_seconds=result["seconds"],
            )
        else:
            result = registration.cpu_robust_register(
                options.source, options.target, mode=options.mode, **parameters)
            directory.mkdir(parents=True, exist_ok=False)
            result.transform.save(directory / "transform.lta")
            import nibabel as nib
            nib.save(result.header_image, directory / "mapped.header.mgz")
            report = result.report
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with (directory / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
