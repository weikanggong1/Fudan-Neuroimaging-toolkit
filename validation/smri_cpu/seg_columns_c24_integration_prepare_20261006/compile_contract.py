"""C24 production cache compile/reuse contract: metadata only, no tensor math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from controller_safety import child_subprocess_lock_wrapper


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "plan", "bindings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--approved-compile-contract", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    bindings = json.loads(args.bindings.read_text())
    assert bindings["public_PLAN_sha256"] == sha(args.plan)
    assert sha(__file__) == plan["validation_sources"]["compile_contract.py"]
    assert sha(Path(__file__).with_name("controller_safety.py")) == plan["validation_sources"]["controller_safety.py"]
    lock_fd = int(os.environ["FNIT_VALIDATION_LOCK_FD"])
    assert os.fstat(lock_fd).st_ino == (args.root / plan["compile_contract"]["common_lock"]).stat().st_ino
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert sorted(os.sched_getaffinity(0)) == plan["compile_contract"]["affinity"]
    resource.setrlimit(resource.RLIMIT_AS, (8_000_000_000, 8_000_000_000))
    def sources():
        return {name: sha(args.source / "fnit/synthseg_parc" / name)
                for name in plan["production_sources"]["candidate"]}
    def old_cache():
        return {entry["fnit_relative"]: {"bytes": (args.root / entry["fnit_relative"]).stat().st_size,
                                        "sha256": sha(args.root / entry["fnit_relative"])}
                for entry in bindings["C72_cache_files"]}
    old_expected = {entry["fnit_relative"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                    for entry in bindings["C72_cache_files"]}
    before = sources()
    assert before == plan["production_sources"]["candidate"] and old_cache() == old_expected
    sys.path.insert(0, str(args.source))
    import torch
    from fnit.synthseg_parc import _cpu_columns_build as mature
    from fnit.synthseg_parc import _cpu_columns_c24_build as builds
    from fnit.synthseg_parc.cpu_columns_c24 import _ColumnsC24
    def flags():
        return {"CUDA_initialized": torch.cuda.is_initialized(), "intra": torch.get_num_threads(),
                "interop": torch.get_num_interop_threads(), "oneDNN": torch.backends.mkldnn.enabled,
                "matmul_TF32": torch.backends.cuda.matmul.allow_tf32,
                "cuDNN_TF32": torch.backends.cudnn.allow_tf32,
                "CPU_autocast": torch.is_autocast_enabled("cpu"),
                "CUDA_autocast": torch.is_autocast_enabled("cuda"),
                "default_dtype": str(torch.get_default_dtype()), "grad": torch.is_grad_enabled()}
    initial = flags()
    assert not initial["CUDA_initialized"]
    report = {"schema": "fnit_C24_production_compile_contract/v1", "mode": args.mode,
              "compile_calls": 0, "copy_calls": 0, "SGEMM_calls": 0, "MRI_calls": 0,
              "model_forward_calls": 0, "GPU_calls": 0, "thread_setter_calls": 0,
              "status": "started", "flags_before": initial, "source_before": before,
              "validation_helper_before": sha(Path(__file__).with_name("controller_safety.py"))}
    original_popen = mature.subprocess.Popen
    mature.subprocess.Popen = child_subprocess_lock_wrapper(original_popen, lock_fd)
    original_compile = mature._compile
    def observed_compile(command):
        report["compile_calls"] += 1
        assert [Path(value).name for value in command if value.endswith(".cpp")] == ["_columns_c24.cpp"]
        assert tuple(command[1:1 + len(mature._FLAGS)]) == mature._FLAGS
        return original_compile(command)
    mature._compile = observed_compile
    started = time.monotonic()
    error = None
    try:
        artifact = builds.build_artifact(torch)
        manifest = Path(artifact["library"]).with_suffix(".json")
        record = json.loads(manifest.read_text())
        assert record["identity"]["compiler"] == bindings["compiler_identity"]
        assert record["identity"]["accepted_C72_build_key"] == bindings["C72_cache_key"]
        helper = _ColumnsC24(artifact["library"], artifact["provider_sha256"], allow_compute=False)
        helper._provider_still_matches()
        report["loaded_ABI_description"] = helper.library.fnit_columns_c24_abi_description()
        assert report["loaded_ABI_description"] == 10404
        report["loaded_provider_sha256"] = helper.provider_sha256
        report["loaded_provider_path"] = str(helper.provider)
        assert not helper.allow_compute and not helper.allow_bounded_contracts
        assert report["compile_calls"] == (1 if args.mode == "cold" else 0)
        report.update(artifact=artifact, cache_manifest={"bytes": manifest.stat().st_size, "sha256": sha(manifest)},
                      library={"bytes": Path(artifact["library"]).stat().st_size, "sha256": sha(artifact["library"])},
                      status="metadata_compile_or_reuse_passed")
    except BaseException as caught:
        error = caught
        report.update(status="failed", error_type=type(caught).__name__, error=str(caught))
    finally:
        mature._compile = original_compile
        mature.subprocess.Popen = original_popen
        report["validation_helper_after"] = sha(Path(__file__).with_name("controller_safety.py"))
        report["validation_helper_unchanged"] = report["validation_helper_before"] == report["validation_helper_after"] == plan["validation_sources"]["controller_safety.py"]
        report["flags_after"] = flags()
        report["source_after"] = sources()
        report["C72_cache_after"] = old_cache()
        report["RSS_maximum_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["compiler_maxRSS_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        report["worker_seconds"] = time.monotonic() - started
        report["valid_compile_contract"] = (error is None and report["validation_helper_unchanged"] and initial == report["flags_after"]
            and before == report["source_after"] and report["C72_cache_after"] == old_expected
            and max(report["RSS_maximum_bytes"], report["compiler_maxRSS_bytes"]) <= 4_000_000_000)
        (args.output / "COMPILE_CONTRACT.json").write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error
    assert report["valid_compile_contract"]


if __name__ == "__main__":
    main()
