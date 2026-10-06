"""One source-bound local guard/cache/mock exception suite; no MRI/compiler."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    leaf = Path(__file__).resolve().parent
    repository = leaf.parents[2]
    sys.path.insert(0, str(repository / "src"))
    names = ["src/fnit/synthseg_parc/cpu_columns.py", "src/fnit/synthseg_parc/_cpu_columns_build.py",
             "src/fnit/synthseg_parc/_columns_reuse.cpp", "src/fnit/synthseg_parc/cpu_conv.py",
             "src/fnit/synthseg_parc/segment.py", "tests/test_synthseg_cpu_columns.py"]
    def sources():
        return {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in names}
    before = sources()
    import torch
    import pytest
    import fnit
    from fnit.synthseg_parc import cpu_columns
    assert Path(fnit.__file__).resolve() == repository / "src/fnit/__init__.py"
    assert Path(cpu_columns.__file__).resolve() == repository / "src/fnit/synthseg_parc/cpu_columns.py"
    assert torch.__version__ == "2.5.1" and not torch.cuda.is_initialized()
    def flags():
        return {"CUDA_initialized": torch.cuda.is_initialized(), "oneDNN": torch.backends.mkldnn.enabled,
                "matmul_TF32": torch.backends.cuda.matmul.allow_tf32, "cuDNN_TF32": torch.backends.cudnn.allow_tf32,
                "CPU_autocast": torch.is_autocast_enabled("cpu"), "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads()}
    initial_flags = flags()
    tick = time.monotonic()
    code = pytest.main([str(repository / "tests/test_synthseg_cpu_columns.py"), "-q"])
    report = {"schema": "fnit_cpu_columns_local_guard_contract_receipt/v1", "exit_code": int(code),
              "scope": "Local source-bound unit guards/cache/mock compilation and exceptions only; no numerical performance/precision or MRI benchmark",
              "python": sys.version.split()[0], "Torch": torch.__version__,
              "source_before": before, "source_after": sources(),
              "flags_before": initial_flags, "flags_after": flags(),
              "wall_seconds": time.monotonic() - tick,
              "real_Cpp_compile_or_dlopen_calls": 0, "real_copy_SGEMM_convolution_calls": 0,
              "MRI_or_native_or_GPU_calls": 0,
              "actual_accepted_parameter_positive_guard": "SHA qualification mocked; actual F32/metadata and negative raw parameter gate tested; real positive whole pending",
              "new_clean_Conda_install_or_compiled_artifact_assessed": False}
    report["source_unchanged"] = report["source_before"] == report["source_after"]
    report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
    report["valid_local_contract_receipt"] = code == 0 and report["source_unchanged"] and report["flags_unchanged"]
    (leaf / "LOCAL_GUARDS.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["valid_local_contract_receipt"]
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
