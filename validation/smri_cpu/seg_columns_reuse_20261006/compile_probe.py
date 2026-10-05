"""Compile/load interface probe only: never invoke copy, SGEMM or allocate tensors."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
    os.sched_setaffinity(0, [32, 36, 40, 44, 48, 52, 56, 60])
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    source = args.root / "repo/src/fnit/synthseg_parc"
    source_before = {name: sha(source / name) for name in plan["source14"]}
    assert source_before == plan["source14"]
    for name, digest in plan["prototype_sources"].items():
        assert sha(args.workspace / name) == digest
    import torch
    from prototype import ColumnsReuse
    torch_root = Path(torch.__file__).resolve().parent
    flags = lambda: {"cuda_initialized": torch.cuda.is_initialized(),
                     "mkldnn": bool(torch.backends.mkldnn.enabled),
                     "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                     "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                     "grad_enabled": torch.is_grad_enabled(),
                     "autocast_CPU": torch.is_autocast_enabled("cpu")}
    before = flags()
    assert torch.__version__ == "2.5.1" and not before["cuda_initialized"]
    assert not torch._C._GLIBCXX_USE_CXX11_ABI
    for name, digest in plan["headers"].items():
        assert sha(torch_root / "include" / name) == digest
    assert sha(torch_root / "lib/libtorch_cpu.so") == plan["libtorch_cpu_sha256"]
    cxx = Path(plan["compiler_path"])
    assert cxx.exists() and cxx.resolve() == Path(plan["compiler_resolved_path"])
    library = args.output / "columns_reuse.so"
    command = [str(cxx), "-std=c++17", "-shared", "-fPIC", "-O2", "-fno-fast-math", "-ffp-contract=off",
               "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0",
               "-I" + str(torch_root / "include"), "-I" + str(torch_root / "include/torch/csrc/api/include"),
               str(args.workspace / "columns_reuse.cpp"), "-L" + str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
               "-Wl,--no-undefined", "-Wl,-z,relro", "-Wl,-z,now",
               "-Wl,-rpath," + str(torch_root / "lib"),
               "-Wl,-rpath-link," + str(torch_root / "lib"),
               "-Wl,-rpath-link," + str(Path(sys.prefix) / "lib"), "-o", str(library)]
    child_env = os.environ.copy()
    child_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    started = time.monotonic()
    process = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=120)
    (args.output / "compiler.stdout.log").write_text(process.stdout)
    (args.output / "compiler.stderr.log").write_text(process.stderr)
    report = {"schema": "fnit_seg_columns_compile_load/v1", "status": "compile_failed",
              "scope": "Interface compile/load only, no tensors or numerical copy/SGEMM calls",
              "plan_sha256": sha(args.workspace / "PLAN.json"), "compiler_argv": command,
              "compiler_rc": process.returncode, "compile_wall_seconds": time.monotonic() - started,
              "compiler_maxRSS_KiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "source_before": source_before, "flags_before": before,
              "Torch_version": torch.__version__, "actual_CPU_affinity": sorted(os.sched_getaffinity(0)),
              "copy_calls": 0, "SGEMM_calls": 0, "MRI_calls": 0, "tensor_allocation_calls": 0,
              "compute_activation": False, "global_allocator_changed": False, "thread_setters_called": False}
    if process.returncode == 0:
        prototype = ColumnsReuse(library, plan["provider_sha256"], allow_compute=False)
        assert not prototype.allow_compute and not prototype.allow_bounded_contracts
        global_sgemm = ctypes.cast(ctypes.CDLL(None).sgemm_, ctypes.c_void_p).value
        assert global_sgemm == prototype.sgemm_address
        report.update({"status": "compiled_loaded_interface_only", "library_sha256": sha(library),
                       "library_bytes": library.stat().st_size, "abi_description": 10404,
                       "provider_sha256": prototype.provider_sha256, "provider_basename": prototype.provider.name,
                       "Torch_handle_and_global_SGEMM_same_address": True,
                       "interface_exports_loaded": ["fnit_columns_abi_description", "fnit_copy_columns_f32", "fnit_same_provider_sgemm_f32"]})
    report["source_after"] = {name: sha(source / name) for name in source_before}
    report["flags_after"] = flags()
    report["source_unchanged"] = report["source_after"] == source_before
    report["flags_unchanged"] = report["flags_after"] == before
    assert report["source_unchanged"] and report["flags_unchanged"]
    (args.output / "COMPILE.private.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in ("status", "compiler_rc", "compile_wall_seconds", "copy_calls", "SGEMM_calls", "MRI_calls")}))
    if process.returncode:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
