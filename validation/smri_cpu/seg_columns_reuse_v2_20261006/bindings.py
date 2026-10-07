"""Private stdlib identity checks shared by metadata and bounded-contract workers."""
import hashlib
import json
from pathlib import Path


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check_sources(root, workspace, plan):
    root, workspace = Path(root), Path(workspace)
    results = {}
    for name, digest in plan["source14"].items():
        value = identity(root / "repo/src/fnit/synthseg_parc" / name)
        if value["sha256"] != digest:
            raise RuntimeError("production source changed: " + name)
        results["source/" + name] = value
    for name, digest in plan["prototype_sources"].items():
        value = identity(workspace / name)
        if value["sha256"] != digest:
            raise RuntimeError("frozen v2 source changed: " + name)
        results["prototype/" + name] = value
    for name, record in plan["previous_frozen_files"].items():
        value = identity(root / record["fnit_relative_path"])
        if value != {key: record[key] for key in ("bytes", "sha256")}:
            raise RuntimeError("v1 interface/binary identity changed: " + name)
        results["previous/" + name] = value
    previous_compile = json.loads((root / plan["previous_frozen_files"]["COMPILE.json"]["fnit_relative_path"]).read_text())
    if (previous_compile["status"] != "compiled_loaded_interface_only"
            or previous_compile["library_sha256"] != plan["previous_frozen_files"]["columns_reuse.so"]["sha256"]):
        raise RuntimeError("previous interface-only provenance invalid")
    if previous_compile["source_before"] != plan["source14"] or previous_compile["source_after"] != plan["source14"]:
        raise RuntimeError("v1 source did not match current frozen source14")
    for counter in ("copy_calls", "SGEMM_calls", "MRI_calls"):
        if previous_compile[counter] != 0:
            raise RuntimeError("prior stage was not interface-only")
    return results


def check_runtime(torch, plan):
    if torch.__version__ != "2.5.1" or torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("target Torch2.5.1/ABI0 required")
    torch_root = Path(torch.__file__).resolve().parent
    if identity(torch_root / "lib/libtorch_cpu.so")["sha256"] != plan["libtorch_cpu_sha256"]:
        raise RuntimeError("Torch CPU library bytes changed")
    for name, digest in plan["headers"].items():
        if identity(torch_root / "include" / name)["sha256"] != digest:
            raise RuntimeError("target header changed: " + name)


def flags(torch):
    return {"CUDA_initialized": torch.cuda.is_initialized(),
            "oneDNN": bool(torch.backends.mkldnn.enabled),
            "matmul_TF32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cuDNN_TF32": bool(torch.backends.cudnn.allow_tf32),
            "CPU_autocast": bool(torch.is_autocast_enabled("cpu")),
            "grad_enabled": torch.is_grad_enabled(), "default_dtype": str(torch.get_default_dtype())}
