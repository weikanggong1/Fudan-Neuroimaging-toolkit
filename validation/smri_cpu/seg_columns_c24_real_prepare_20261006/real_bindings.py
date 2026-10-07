"""Frozen CPU8 real-prefix/layer bindings; no image, compiler or model calls."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path, report):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_bindings(root, workspace, plan):
    root, workspace = Path(root), Path(workspace)
    observed = {}
    for relative, expected in plan["source_bindings"].items():
        actual = identity(root / "repo" / relative)
        require(actual == expected, "mature source changed: " + relative)
        observed["production/" + relative] = actual
    for name, expected in plan["worker_sources"].items():
        actual = identity(workspace / name)
        require(actual == expected, "new worker changed: " + name)
        observed["worker/" + name] = actual
    for name, expected in plan["assets"].items():
        actual = identity(root / expected["fnit_relative"])
        require(actual == {key: expected[key] for key in ("bytes", "sha256")}, "asset changed: " + name)
        observed["asset/" + name] = actual
    for name, expected in plan["dependencies"].items():
        actual = identity(root / expected["fnit_relative"])
        require(actual == {key: expected[key] for key in ("bytes", "sha256")}, "dependency changed: " + name)
        observed["dependency/" + name] = actual
    for name, entry in plan["mature_AST_bindings"].items():
        tree = ast.parse((root / "repo" / entry["source"]).read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry["class"])
        function = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == entry["method"])
        require(hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest() == entry["AST_sha256"], "mature sequence AST changed: " + name)
    observed["PLAN"] = identity(workspace / "PLAN.json")
    return observed


def check_environment(plan):
    require(sorted(os.sched_getaffinity(0)) == plan["affinity"], "same eight physical CPU cores required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    require(all(os.environ.get(name) == "8" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")), "same eight-thread CPU environment required")
    require(not any(name in os.environ for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "OPENBLAS_CORETYPE")), "loader overrides forbidden")


def flags(torch):
    return {"CUDA_initialized": torch.cuda.is_initialized(), "oneDNN": bool(torch.backends.mkldnn.enabled),
            "matmul_TF32": bool(torch.backends.cuda.matmul.allow_tf32), "cuDNN_TF32": bool(torch.backends.cudnn.allow_tf32),
            "CPU_autocast": bool(torch.is_autocast_enabled("cpu")), "grad_enabled": torch.is_grad_enabled(),
            "default_dtype": str(torch.get_default_dtype())}


def check_runtime(torch, plan):
    require(torch.__version__ == "2.5.1" and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.get_default_dtype() == torch.float32, "Torch2.5.1/ABI0/default FP32 required")
    prefix = Path(torch.__file__).resolve().parent
    observed = {"libtorch_cpu": identity(prefix / "lib/libtorch_cpu.so")}
    require(observed["libtorch_cpu"]["sha256"] == plan["libtorch_cpu_sha256"], "Torch library changed")
    for relative, sha in plan["headers"].items():
        observed[relative] = identity(prefix / "include" / relative)
        require(observed[relative]["sha256"] == sha, "Torch header changed")
    from fnit.synthseg_parc.cpu_columns import _Columns
    provider = _Columns(None, plan["provider_sha256"], allow_compute=False)
    provider._provider_still_matches()
    observed["provider"] = identity(provider.provider)
    return observed


def value_sha(tensor):
    require(tensor.device.type == "cpu" and tensor.is_contiguous(), "hash requires a contiguous CPU tensor")
    array = tensor.detach().numpy()
    buffer = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(buffer), 8 * 1024**2):
        digest.update(buffer[start:start + 8 * 1024**2])
    return digest.hexdigest()


def finite_tensor(tensor, np):
    array = tensor.detach().numpy()
    for channel in range(array.shape[1]):
        for depth in range(0, array.shape[2], 8):
            if not bool(np.isfinite(array[:, channel, depth:depth + 8]).all()):
                return False
    return True


def capture_bindings(run, plan):
    run = Path(run)
    meta_path = run / "capture/report.json"
    report = json.loads(meta_path.read_text())
    require(report["valid_capture"] and report["status"] == "two_prefix_features_saved_no_conv1", "valid original/flipped capture required")
    require(report["PLAN"] == plan["PLAN_identity"], "capture and layer PLAN differ")
    require(report["conv0_calls"] == 2 and report["conv1_calls"] == report["model_forward_calls"] == 0, "capture scope changed")
    observed = {"capture_report": identity(meta_path)}
    for name in ("original", "flipped"):
        path = run / "capture" / (name + ".private.npy")
        observed[name] = identity(path)
        require(observed[name] == report["features"][name]["file"], "capture array changed: " + name)
        require(report["features"][name]["shape"] == plan["layer_input_shape"] and report["features"][name]["dtype"] == "torch.float32", "capture geometry changed")
    return report, observed


def load_helper(root, plan):
    prototype = root / plan["dependencies"]["C24_prototype"]["fnit_relative"]
    specification = importlib.util.spec_from_file_location("fnit_private_C24_real_prototype", prototype)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.ColumnsC24(root / plan["dependencies"]["C24_binary"]["fnit_relative"],
                             plan["provider_sha256"], allow_compute=True, allow_bounded_contracts=False)
