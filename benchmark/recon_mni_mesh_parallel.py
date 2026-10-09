"""真实自产同输入 MNI/网格两阶段完整串行与并行配对，包含读写与进程启动。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "overlay", "subject", "weights", "assets", "native-bin", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execution", choices=("serial", "parallel"), required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--parent-preinitialized", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.overlay))
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.recon_all.mni_mesh_parallel import run_mni_and_validate
    from fnit.recon_all.profiling import configure_cuda_allocator
    args.output.mkdir(parents=True, exist_ok=False)
    subject = args.output / "subject"
    transform = Path("mri/transforms/synthmorph.1.0mm.1.0mm")
    files = [Path("mri/orig.mgz"), transform / "invol.crop.nii.gz", transform / "aff.lta"]
    files += [Path("surf") / f"{hemi}.{name}" for hemi in ("lh", "rh")
              for name in ("orig", "white", "pial", "sphere.reg")]
    preparation = time.monotonic()
    inputs = {}
    for name in files:
        source, target = args.subject / name, subject / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[str(name)] = sha(source)
        assert sha(target) == inputs[str(name)]
    preparation = time.monotonic() - preparation
    allocator = configure_cuda_allocator(args.device, "disabled")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parent_live = None
    if args.parent_preinitialized:
        # 模拟已初始化 API，保留一个实际存活张量；选择目标 GPU，不影响 CUDA0。
        parent_live = torch.ones(1 << 20, dtype=torch.float32, device=args.device)
        torch.cuda.synchronize(args.device)
    started = time.monotonic()
    result = run_mni_and_validate(subject=subject, weights=args.weights, assets=args.assets,
        warp_binaries=tuple(args.native_bin / name for name in ("mri_warp_convert", "mri_ca_register", "mri_convert")),
        report_path=args.output / "group.json", device=args.device, threads=args.threads,
        execution=args.execution, profile_stages=True, code_version=args.code_version)
    elapsed = time.monotonic() - started
    comparisons = {}
    if args.reference:
        reference = json.loads((args.reference / "group.json").read_text())
        for key in ("forward", "inverse", "check"):
            old = args.reference / "subject" / Path(reference["mni_nonlinear"][key]).relative_to(args.reference / "subject")
            new = Path(result["mni_nonlinear"][key])
            first, second = nib.load(str(old)), nib.load(str(new))
            a, b = np.asarray(first.dataobj), np.asarray(second.dataobj)
            error = np.abs(a.astype(np.float64)-b.astype(np.float64))
            comparisons[key] = {"different_voxels": int(np.count_nonzero(a != b)),
                "maximum": float(error.max()), "p99": float(np.quantile(error, .99)),
                "dtype_equal": first.get_data_dtype() == second.get_data_dtype(),
                "affine_equal": bool(np.array_equal(first.affine, second.affine)),
                "shape_equal": a.shape == b.shape, "header_equal": first.header.binaryblock == second.header.binaryblock,
                "file_sha_equal": sha(old) == sha(new)}
        comparisons["mesh_equal"] = result["mesh_validation"] == reference["mesh_validation"]
    report = {"code_version": args.code_version, "execution": args.execution,
              "complete_api_wall_seconds": elapsed, "fixture_prepare_seconds_excluded": preparation,
              "scope": "complete MNI+mesh group; fixture copies excluded equally, parent-init tracked separately; not raw T1 whole",
              "inputs_sha256": inputs, "benchmark_sha256": sha(__file__),
              "allocator": allocator, "parent_preinitialized": args.parent_preinitialized,
              "parent_live_tensor_unchanged": None if parent_live is None else bool(torch.all(parent_live == 1).item()),
              "comparison": comparisons, "host": platform.node(), "CPU_affinity": sorted(os.sched_getaffinity(0)),
              "torch_version": torch.__version__, "python_version": platform.python_version(),
              "gpu_name": torch.cuda.get_device_name(args.device), "group": result}
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"execution": args.execution, "seconds": elapsed, "comparison": comparisons}), flush=True)


if __name__ == "__main__": main()
