"""真实已保存球面的CPU逐算子诊断；不作benchmark、不改质量判据或生产。

original调用冻结face_area_normals；split只把其CPU表达式逐项展开并保存
回执，定位double复制/sqrt/float转换。split改变临时量生命周期，不能当作
原函数替换。每步记录shape/dtype/stride/地址对齐及NumPy只读有限值统计；
私有JSON还保存进程已映射的Torch/MKL/OpenMP/libm等库元数据，不复制库。
固定4线程、CUDA隐藏；新输出目录存在则拒绝。崩溃后保留最后phase与日志。
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maps_metadata():
    """只取本进程数值运行库的映射路径/stat，不读取库内容或其他进程。"""
    selected = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        name = fields[5]
        if not name.startswith("/") or not re.search(r"lib(torch|c10|mkl|gomp|iomp|omp|openblas|blas|lapack|m\.so|m-|stdc\+\+|c\.so|gcc_s|sleef)|ld-linux", Path(name).name):
            continue
        item = selected.setdefault(name, {"segments": []})
        item["segments"].append({"address_range": fields[0], "permissions": fields[1],
                                  "offset": fields[2], "device": fields[3], "inode": fields[4]})
        if "stat" not in item:
            try:
                stat = Path(name).stat()
                item["stat"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "device": stat.st_dev, "inode": stat.st_ino}
            except OSError as error:
                item["stat"] = {"unavailable": type(error).__name__}
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--surface-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--face-kernel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("original", "split"), required=True)
    parser.add_argument("--import-order", choices=("numpy-first", "torch-first"), default="numpy-first",
                        help="显式诊断条件；默认NumPy/nibabel先，顺序改变须另开新进程")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name, digest in ((args.surface, args.surface_sha256),
                         (args.source_root / "src/fnit/recon_all/mris_register_nonlinear.py", args.face_kernel_sha256)):
        if re.fullmatch("[0-9a-f]{64}", digest) is None or sha(name) != digest:
            raise ValueError("input/source SHA mismatch: " + str(name))
    args.output.mkdir(parents=True, mode=0o700)
    faulthandler.enable(all_threads=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
    state = {"scope": "read-only CPU tensor fault localization; not quality acceptance, not performance benchmark",
             "status": "running", "mode": args.mode, "import_order": args.import_order,
             "surface": str(args.surface), "surface_sha256": args.surface_sha256,
             "face_kernel_sha256": args.face_kernel_sha256, "script_sha256": sha(__file__),
             "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
             "python": sys.version, "python_executable": sys.executable,
             "python_executable_sha256": sha(Path(sys.executable).resolve()),
             "platform": platform.platform(), "steps": [], "maps": {}, "tensors": {}}
    path = args.output / "state.private.json"

    def save(phase):
        state["phase"] = phase
        state["updated_monotonic"] = time.monotonic()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)

    save("before_numeric_library_imports")
    state["maps"]["before_numeric_imports"] = maps_metadata()
    save("numeric_library_imports_starting")
    try:
        if args.import_order == "numpy-first":
            import numpy as np
            import nibabel.freesurfer.io as fsio
            import torch
        else:
            import torch
            import numpy as np
            import nibabel.freesurfer.io as fsio
        state["maps"]["after_numeric_imports"] = maps_metadata()
        state["versions"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "cuda_build": torch.version.cuda}
        state["torch_build_configuration"] = torch.__config__.show()
        state["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
        save("before_torch_thread_configuration")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        state["actual_threads"] = {"torch_intra": torch.get_num_threads(),
                                   "torch_interop": torch.get_num_interop_threads()}
        state["cuda_initialized"] = torch.cuda.is_initialized()
        state["maps"]["after_thread_configuration"] = maps_metadata()
        save("after_torch_thread_configuration")
        sys.path.insert(0, str(args.source_root / "src"))
        from fnit.recon_all import mris_register_nonlinear as operation
        if Path(operation.__file__).resolve() != (args.source_root / "src/fnit/recon_all/mris_register_nonlinear.py").resolve() or sha(operation.__file__) != args.face_kernel_sha256:
            raise ValueError("loaded CPU face kernel is outside the frozen source")
        state["loaded_face_kernel"] = str(operation.__file__)
        state["maps"]["after_fnit_import"] = maps_metadata()
        save("before_geometry_read")
        vertices, faces = fsio.read_geometry(str(args.surface))

        def array_info(array):
            pointer = int(array.__array_interface__["data"][0])
            finite = np.isfinite(array)
            data = array[finite]
            return {"shape": list(array.shape), "dtype": str(array.dtype),
                    "strides_bytes": list(array.strides), "c_contiguous": bool(array.flags.c_contiguous),
                    "f_contiguous": bool(array.flags.f_contiguous), "native_byteorder": bool(array.dtype.isnative),
                    "data_address": hex(pointer), "alignment_modulo": {str(n): pointer % n for n in (16, 32, 64, 128, 256, 4096)},
                    "finite": bool(finite.all()), "nonfinite_elements": int(array.size - finite.sum()),
                    "minimum_finite": float(data.min()) if data.size else None,
                    "maximum_finite": float(data.max()) if data.size else None,
                    "sha256_c_order_bytes": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}

        state["input_arrays"] = {"vertices": array_info(vertices), "faces": array_info(faces)}
        if not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("invalid source surface index range")
        save("after_geometry_read")

        def tensor_info(tensor):
            result = array_info(tensor.detach().numpy())
            result.update(torch_dtype=str(tensor.dtype), torch_shape=list(tensor.shape),
                          torch_stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                          device=str(tensor.device), torch_data_ptr=hex(tensor.data_ptr()))
            return result

        def step(name, function):
            save(name + ":before_operation")
            result = function()
            state["steps"].append(name)
            save(name + ":operation_returned_before_numpy_stats")
            state["tensors"][name] = ([tensor_info(item) for item in result] if isinstance(result, tuple)
                                      else tensor_info(result))
            state["maps"][name] = maps_metadata()
            save(name + ":after_stats")
            return result

        positions = step("positions_from_numpy_fp32", lambda: torch.from_numpy(np.asarray(vertices, np.float32)))
        triangles = step("faces_from_numpy_int64", lambda: torch.from_numpy(faces.astype(np.int64)))
        if args.mode == "original":
            step("existing_face_area_normals", lambda: operation.face_area_normals(positions, triangles, signed_sphere=True))
        else:
            first = step("first_edge", lambda: positions[triangles[:, 1]] - positions[triangles[:, 0]])
            second = step("second_edge", lambda: positions[triangles[:, 2]] - positions[triangles[:, 0]])
            cross = step("cross", lambda: torch.stack((
                first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
                first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
                first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]), dim=1))
            squared = step("squared_fp32", lambda: (cross[:, 0] * cross[:, 0] + cross[:, 1] * cross[:, 1]
                                                    + cross[:, 2] * cross[:, 2]))
            wide = step("squared_to_double", lambda: squared.double())
            roots = step("double_sqrt", lambda: wide.sqrt())
            step("sqrt_to_float", lambda: roots.float())
        if sha(args.surface) != args.surface_sha256 or sha(operation.__file__) != args.face_kernel_sha256:
            raise ValueError("input/source changed during read-only diagnosis")
        state["status"] = "complete_diagnostic"
        state["cuda_initialized_after"] = torch.cuda.is_initialized()
        state["quality_verdict"] = "not_assessed_by_this_diagnostic"
        save("complete")
        return 0
    except BaseException as error:
        state.update(status="failed", error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        save(state.get("phase", "failed"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
