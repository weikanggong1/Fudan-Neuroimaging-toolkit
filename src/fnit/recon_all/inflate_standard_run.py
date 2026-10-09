"""完整固定标准 inflation：读取smoothwm，写inflated和真实累积sulc。

这是显式实验接口，不切换recon-all默认。CPU复用已有NumPy/Numba算法，
Torch后端复用有序法向与梯度平均，保留完整默认积分和最终处理。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
_PROCESS_STARTED = time.perf_counter()

import nibabel.freesurfer.io as fsio
import numpy as np

from .inflate_python import face_area_total, inflate_updates, postprocess
from .sphere_standard_python import write_standard_sphere_surface


def run_standard_inflate(*, input_surface: str | Path, inflated_output: str | Path,
                         sulc_output: str | Path, backend: str = "numpy",
                         device: str = "cpu", profile: bool = False) -> dict:
    """三角网格surface RAS/mm→同序inflated及(N,)FP32 sulc/mm。

    input_surface为声明自产smoothwm；两个输出为不同的明确路径。backend
    默认numpy（已有CPU算法），可显式torch，device默认cpu，CUDA不回退。
    profile默认False；True同步Torch子段计时。自动沿固定源码由voxelsize
    调整高分辨率每档步数，其余默认16/8/4/2/1/0、dt=momentum=0.9、
    RMS目标0.015。保留原体积几何文本，nibabel读写其余格式。
    返回路径、后端、设备、顶点/面数、步数、RMS、子段与含I/O总墙钟。
    有限坐标/合法索引/无孤立顶点与正面积为必要条件；失败抛异常并可能
    留部分输出，不读参考修复、不生成占位sulc。生产默认尚未接入。
    """
    started = time.perf_counter()
    input_surface, inflated_output, sulc_output = map(Path, (input_surface, inflated_output, sulc_output))
    if len({path.resolve() for path in (input_surface, inflated_output, sulc_output)}) != 3:
        raise ValueError("input and two output paths must differ")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be numpy or torch")
    if backend == "numpy" and device != "cpu":
        raise ValueError("numpy backend requires explicit cpu device")
    vertices, faces, metadata = fsio.read_geometry(str(input_surface), read_metadata=True)
    vertices, faces = np.asarray(vertices, np.float32), np.asarray(faces, np.int32)
    if not np.isfinite(vertices).all() or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("surface must have finite nonempty (N,3) coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("surface faces must have valid (F,3) vertex indices")
    if np.any(np.bincount(faces.ravel(), minlength=len(vertices)) == 0):
        raise ValueError("standard inflation does not support isolated vertices")
    if not face_area_total(vertices, faces) > 0:
        raise ValueError("surface must have positive total area")
    xsize = float(metadata.get("voxelsize", (1.,))[0])
    if not np.isfinite(xsize) or xsize <= 0:
        raise ValueError("volume geometry x voxel size must be positive")
    niterations = int(np.floor(np.float32(10) / np.float32(xsize) + .5)) if xsize < .8 else 10
    read_validation_seconds = time.perf_counter() - started
    diagnostics = {}
    setup_start = time.perf_counter()
    if backend == "numpy":
        sulc = np.zeros(len(vertices), np.float32)
        coordinates = inflate_updates(xyz=vertices, faces=faces, niterations=niterations,
                                     sulc=sulc, diagnostics=diagnostics)
        coordinates = postprocess(coordinates, faces, face_area_total(vertices, faces))
        sulc = (sulc.astype(np.float64) - np.sum(sulc, dtype=np.float64) / len(sulc)).astype(np.float32)
        setup_seconds = 0.0
    else:
        import torch
        from .inflate_torch import TorchInflationContext
        context = TorchInflationContext(faces=faces, nvertices=len(vertices), device=device)
        original = torch.as_tensor(vertices, device=context.device)
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        setup_seconds = time.perf_counter() - setup_start
        diagnostics = context.integrate(vertices=original, niterations=niterations, profile=profile)
        coordinates_tensor, sulc_tensor = context.finalize(
            coordinates=diagnostics.pop("coordinates"), original_vertices=original,
            sulc=diagnostics.pop("sulc"))
        coordinates, sulc = coordinates_tensor.cpu().numpy(), sulc_tensor.cpu().numpy()
    compute_transfer_finalize_seconds = time.perf_counter() - setup_start
    if not np.isfinite(coordinates).all() or not np.isfinite(sulc).all():
        raise FloatingPointError("inflation coordinates or sulc became nonfinite")
    write_start = time.perf_counter()
    inflated_output.parent.mkdir(parents=True, exist_ok=True)
    sulc_output.parent.mkdir(parents=True, exist_ok=True)
    write_standard_sphere_surface(output=inflated_output, vertices=coordinates, faces=faces,
                                  source_surface=input_surface,
                                  create_stamp="created by FNIT standard inflation")
    fsio.write_morph_data(str(sulc_output), sulc, fnum=len(faces))
    return {"input_surface": str(input_surface), "inflated_output": str(inflated_output),
            "sulc_output": str(sulc_output), "backend": backend, "device": device,
            "profile_synchronizes_substages": profile, "vertices": len(vertices), "faces": len(faces),
            "niterations_per_scale": niterations, "read_validation_seconds": read_validation_seconds,
            "setup_and_transfer_seconds": setup_seconds,
            "compute_transfer_finalize_seconds": compute_transfer_finalize_seconds,
            "write_seconds": time.perf_counter() - write_start, "integration": diagnostics,
            "total_seconds_including_io": time.perf_counter() - started,
            "production_default_changed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--inflated-output", type=Path, required=True)
    parser.add_argument("--sulc-output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int, default=4, help="本CLI CPU/Torch预算，默认4")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    from numba import set_num_threads
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_num_threads(args.threads)
    # The CLI owns this process. Library calls preserve their caller's policy.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda_initialized_before_api = torch.cuda.is_initialized()
    report = run_standard_inflate(input_surface=args.input_surface, inflated_output=args.inflated_output,
                                  sulc_output=args.sulc_output, backend=args.backend,
                                  device=args.device, profile=args.profile)
    import hashlib
    from . import inflate_torch, inflate_topology, place_surface_normals, mris_register_average_numba
    modules = (Path(__file__), Path(inflate_torch.__file__), Path(inflate_topology.__file__),
               Path(place_surface_normals.__file__), Path(mris_register_average_numba.__file__))
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in modules}
    report["threads"] = torch.get_num_threads()
    report["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    report["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    report["half_precision"] = False
    report["cuda_initialized_before_api"] = cuda_initialized_before_api
    if args.backend == "torch" and torch.device(args.device).type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
    report["process_wall_including_module_imports_api_hashes_seconds"] = time.perf_counter() - _PROCESS_STARTED
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
