"""Private stdlib phase1 bindings; no Torch/compiler/math on import."""
import hashlib
import json
import os
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
    result = {}
    for label, prefix, records in (
        ("baseline", root / "repo/src/fnit/synthseg_parc", plan["production_sources"]["baseline"]),
        ("candidate", workspace / "source_candidate/fnit/synthseg_parc", plan["production_sources"]["candidate"]),
        ("support_baseline", root / "repo/src", plan["common_support_sha256"]),
        ("support_candidate", workspace / "source_candidate", plan["common_support_sha256"]),
        ("worker", workspace, plan["validation_sources"]),
    ):
        for name, digest in records.items():
            record = identity(prefix / name)
            if record["sha256"] != digest:
                raise RuntimeError("frozen source changed: " + label + "/" + name)
            result[label + "/" + name] = record
    original = plan["phase1_interface_and_short_contracts"]["original_contract_worker"]
    actual = identity(root / original["fnit_relative"])
    if actual != {key: original[key] for key in ("bytes", "sha256")}:
        raise RuntimeError("original accepted short contract worker changed")
    result["original_contract_worker"] = actual
    if plan.get("_loaded_artifact") is not None:
        record = plan["_loaded_artifact"]
        if (identity(record["library"]) != record["binary"]
                or identity(Path(record["library"]).with_suffix(".json")) != record["manifest"]):
            raise RuntimeError("loaded cache binary/manifest changed after interface gate")
    return result


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


def require_environment(plan):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(name) != "8" for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise RuntimeError("CPU-only eight-thread environment required")
    if any(name in os.environ for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("loader/core overrides are outside the frozen contract environment")
    if sorted(os.sched_getaffinity(0)) != plan["whole_CPU"]["affinity"]:
        raise RuntimeError("declared physical-core affinity required")
    if os.environ.get("CXX") != plan["Conda_CXX"]:
        raise RuntimeError("declared Conda GCC11 executable required")


def loaded_artifact(artifact, plan):
    library = Path(artifact["library"]).resolve()
    cache = Path(os.environ["FNIT_SYNTHSEG_CPU_CACHE"]).resolve()
    if library.parent != cache or artifact["provider_sha256"] != plan["provider_sha256"]:
        raise RuntimeError("artifact outside declared cache/provider")
    manifest = library.with_suffix(".json")
    record = json.loads(manifest.read_text())
    binary = identity(library)
    if (record["key"] != artifact["key"] or record["abi"] != 10404
            or binary != {"bytes": record["library_bytes"], "sha256": record["library_sha256"]}
            or record["identity"]["provider_sha256"] != plan["provider_sha256"]
            or record["identity"]["source_sha256"] != plan["production_sources"]["candidate"]["_columns_reuse.cpp"]):
        raise RuntimeError("cached binary/manifest/source gate failed")
    return {"library": str(library), "binary": binary, "manifest": identity(manifest),
            "key": artifact["key"], "provider_sha256": artifact["provider_sha256"]}
