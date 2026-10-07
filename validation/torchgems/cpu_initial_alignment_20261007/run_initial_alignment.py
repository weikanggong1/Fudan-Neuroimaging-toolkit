"""只运行实验初始对齐，不启动 GEMS 拟合；需要完整 FNIT checkout。"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--image", required=True, help="context.image 3D processing-grid image")
    command.add_argument("--coarse-segmentation", required=True, help="integer labels on the same grid")
    command.add_argument("--atlas-directory", required=True, type=Path, help="contains AtlasDump.mgz")
    command.add_argument("--side", choices=("left", "right"), required=True)
    command.add_argument("--output-directory", required=True, type=Path, help="new directory")
    command.add_argument("--repository-root", type=Path)
    command.add_argument("--device", choices=("cpu",), default="cpu")
    command.add_argument("--threads", type=int, help="optional CLI-only Torch intra-op setting")
    return command


def _load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    # dataclass resolves its declared module; only this own module is registered.
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def main(argv=None):
    options = parser().parse_args(argv)
    if options.threads is not None and options.threads < 1:
        raise ValueError("threads must be positive")
    root = options.repository_root.resolve() if options.repository_root else Path(__file__).resolve().parents[3]
    if options.output_directory.exists():
        raise FileExistsError("output-directory must not exist")
    sys.path.insert(0, str(root / "src"))
    if options.threads is not None:
        import torch
        torch.set_num_threads(options.threads)
    from fnit.gems.context import SubregionContext
    from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
    adapter = _load(root / "validation/robust_register/full_cpu_arithmetic_20261007/cpu_experiment_adapter.py", "_fnit_GEMS_cpu_adapter")
    bridge = _load(Path(__file__).with_name("alignment_experiment.py"), "_fnit_GEMS_cpu_alignment")
    # Provided coarse labels: no Synth inference or weights are requested.
    context = SubregionContext.prepare(
        options.image, need_coarse=True, need_parc=False,
        coarse_segmentation=options.coarse_segmentation, device="cpu")
    recipe = HippoAmygdalaRecipe(options.side, options.atlas_directory)
    with adapter.load_cpu_candidate(root) as cpu_candidate:
        result = bridge.initialize_recipe_alignment(
            recipe, context, method="cpu-robust-experiment", device="cpu",
            cpu_candidate=cpu_candidate, output_directory=options.output_directory,
            coarse_reference=options.coarse_segmentation)
    summary = dict(result.report, atlas_to_context_voxel=result.atlas_to_context_voxel.tolist(),
                   status="completed_experimental_initial_alignment_equivalence_not_assessed")
    with (options.output_directory / "alignment.report.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
