"""Source/file gates for a prepared single-layer trial; standard library only."""
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_bindings(root, workspace, plan):
    require(workspace.resolve() == (root / plan["workspace"]).resolve(), "canonical frozen workspace required")
    observed = {}
    for name, digest in plan["scientific_sources"].items():
        observed["worker/" + name] = identity(workspace / name)
        require(observed["worker/" + name]["sha256"] == digest, "frozen worker changed: " + name)
    for name, digest in plan["source14"].items():
        observed["production/" + name] = identity(root / "repo/src/fnit/synthseg_parc" / name)
        require(observed["production/" + name]["sha256"] == digest, "production source changed: " + name)
    for name, record in plan["dependencies"].items():
        observed["dependency/" + name] = identity(root / record["fnit_relative_path"])
        require(observed["dependency/" + name] == {key: record[key] for key in ("bytes", "sha256")}, "bound file changed: " + name)
    checkpoint = root / plan["checkpoint"]
    manifest = json.loads((checkpoint / "manifest.private.json").read_text())
    require(manifest["status"] == "partial_capture_complete", "existing partial checkpoint required")
    require(manifest["source_files"] == plan["producer_files"], "producer six source identities changed")
    require(manifest["input_sha256"] == plan["T1_input_sha256"]
            and manifest["prepared_values_sha256"] == plan["prepared_values_sha256"], "capture input identity changed")
    require(manifest["checkpoint_files"] == plan["checkpoint_files"], "capture array declarations changed")
    contract = json.loads((root / plan["dependencies"]["v2_contract_report"]["fnit_relative_path"]).read_text())
    require(contract["valid_bounded_contracts"] and contract["completed"]
            and contract["status"] == "bounded_contracts_passed_no_MRI", "accepted short contracts required")
    require(contract["PLAN"] == {key: plan["dependencies"]["v2_PLAN"][key] for key in ("bytes", "sha256")}, "short contract plan mismatch")
    require(len(contract["numeric_rows"]) == 6 and all(row["different_bits"] == 0 for row in contract["numeric_rows"]), "short full FP32 contract not exact")
    require(contract["sources_before"] == contract["sources_after"] and contract["flags_unchanged"], "short contract postconditions failed")
    return observed


def flags(torch):
    return {"CUDA_initialized": torch.cuda.is_initialized(), "oneDNN": torch.backends.mkldnn.enabled,
            "matmul_TF32": torch.backends.cuda.matmul.allow_tf32, "cuDNN_TF32": torch.backends.cudnn.allow_tf32,
            "CPU_autocast": torch.is_autocast_enabled("cpu"), "grad_enabled": torch.is_grad_enabled(),
            "default_dtype": str(torch.get_default_dtype())}


def check_runtime(torch, plan):
    package = Path(torch.__file__).resolve().parent
    runtime = {"libtorch_cpu": identity(package / "lib/libtorch_cpu.so")}
    require(runtime["libtorch_cpu"] == plan["TorchCPU_binary"], "installed Torch CPU binary changed")
    require(not torch._C._GLIBCXX_USE_CXX11_ABI, "original Torch ABI0 required")
    for name, digest in plan["headers"].items():
        runtime[name] = identity(package / "include" / name)
        require(runtime[name]["sha256"] == digest, "installed Torch header changed: " + name)
    return runtime


def value_sha(tensor):
    array = tensor.detach().numpy()
    require(array.ndim == 5 and array.shape[0] == 1 and array.flags.c_contiguous, "contiguous singleton-batch value hash required")
    digest = hashlib.sha256()
    for channel in range(array.shape[1]):
        digest.update(memoryview(array[0, channel]).cast("B"))
    return digest.hexdigest()
