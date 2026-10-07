"""从 FNIT checkout 调用独立 CPU robust-register 验证候选。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def positive_threads(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("threads must be positive")
    return number


def parser():
    command = argparse.ArgumentParser(description="FNIT 独立 CPU 保序 robust-register 实验")
    command.add_argument("--source", required=True, help="moving one-frame3D image in scanner RAS mm")
    command.add_argument("--target", required=True, help="fixed one-frame3D image in scanner RAS mm")
    command.add_argument("--output-directory", required=True, help="new directory; must not exist")
    command.add_argument("--mode", choices=("rigid", "affine", "rigid-affine"), default="rigid")
    command.add_argument("--device", choices=("cpu",), default="cpu", help="this validation entry is CPU only")
    command.add_argument("--saturation", type=float, default=50.)
    command.add_argument("--iterations-per-level", type=int, default=5)
    command.add_argument("--stop-distance", type=float, default=.01)
    command.add_argument("--no-initialize-translation", action="store_true")
    command.add_argument("--pyramid-min-size", type=int, default=16)
    command.add_argument("--pyramid-max-size", type=int, default=-1)
    command.add_argument("--highres-iterations", type=int, default=-1)
    command.add_argument("--spatial-chunk-size", type=int, default=131072)
    command.add_argument("--memory-budget-gb", type=float, default=20.)
    command.add_argument("--disable-tf32", action="store_true", help="kept as original parameter; CPU does not use TF32")
    command.add_argument("--repository-root", type=Path, help="FNIT checkout; default inferred from this validation file")
    command.add_argument("--threads", type=positive_threads, help="optional CLI-only Torch intra-op setting; omitted preserves caller default")
    return command


def main(argv=None):
    options = parser().parse_args(argv)
    directory = Path(options.output_directory)
    if directory.exists():
        raise FileExistsError("output-directory must not exist")
    root = options.repository_root.resolve() if options.repository_root is not None else Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root/'src'))
    if options.threads is not None:
        import torch
        torch.set_num_threads(options.threads)
    specification = importlib.util.spec_from_file_location(
        "_fnit_explicit_CPU_experiment", Path(__file__).resolve().with_name("cpu_experiment_adapter.py"))
    if specification is None or specification.loader is None:
        raise ImportError("place the reviewed adapter beside this CLI")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    parameters = dict(device=options.device, saturation=options.saturation,
        iterations_per_level=options.iterations_per_level, stop_distance=options.stop_distance,
        initialize_translation=not options.no_initialize_translation,
        pyramid_min_size=options.pyramid_min_size, pyramid_max_size=options.pyramid_max_size,
        highres_iterations=options.highres_iterations, spatial_chunk_size=options.spatial_chunk_size,
        memory_budget_gb=options.memory_budget_gb, tf32=not options.disable_tf32)
    with module.load_cpu_candidate(repository_root=root) as candidate:
        if options.mode == "rigid-affine":
            result = candidate.cpu_robust_rigid_affine(options.source, options.target,
                stage_directory=directory, **parameters)
            report = dict(status="completed_experimental_current_input_equivalence_not_assessed",
                mode=options.mode, rigid=result['rigid'].report, affine=result['affine'].report,
                combined_RAS_matrix=result['combined_RAS_matrix'].tolist(),
                two_stage_with_save_seconds=result['seconds'])
        else:
            result = candidate.cpu_robust_register(options.source, options.target,
                mode=options.mode, **parameters)
            directory.mkdir(parents=True, exist_ok=False)
            result.transform.save(directory/'transform.lta')
            import nibabel as nib
            nib.save(result.header_image, directory/'mapped.header.mgz')
            report = result.report
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
    with (directory/'report.json').open('x', encoding='utf-8') as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
