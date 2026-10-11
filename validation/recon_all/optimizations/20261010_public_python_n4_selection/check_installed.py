"""离线安装的公开N4选择门；不执行MRI、不初始化CUDA、不代替隔离验收。"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.installed.resolve()))
    import torch
    from fnit.recon_all import (native_free, batch, python_profile, thread_budget,
                               input_stage_parallel, input_n4_chain)
    report = {"scope": "offline installed API/CLI routing, no MRI execution or clean-OS isolation",
              "wheel_bytes": args.wheel.stat().st_size, "wheel_sha256": digest(args.wheel),
              "snapshot_manifest_sha256": digest(args.snapshot / "SOURCE_MANIFEST.json"),
              "snapshot_git_commit": json.loads((args.snapshot / "SOURCE_MANIFEST.json").read_text())["git_commit"],
              "modules": {}}
    for module in (native_free, batch, python_profile, thread_budget,
                   input_stage_parallel, input_n4_chain):
        installed = Path(module.__file__).resolve()
        relative = installed.relative_to(args.installed.resolve())
        source = args.snapshot / "source/src" / relative
        assert installed.read_bytes() == source.read_bytes(), module.__name__
        report["modules"][module.__name__] = {"relative_path": str(relative),
            "source_sha256": digest(source), "installed_sha256": digest(installed), "identical": True}
    for function in (native_free.run_recon_all_python, batch.run_recon_all_python_batch):
        signature = inspect.signature(function)
        assert signature.parameters["input_execution"].default == "serial"
        assert signature.parameters["n4_backend"].default == "native"
        assert signature.parameters["n4_execution"].default == "in-process"
    default = python_profile.python_profile_options(device="cuda:2", threads=4)
    explicit = python_profile.python_profile_options(device="cuda:2", threads=4,
                                                    n4_backend="torch", n4_execution="isolated")
    assert default["n4_backend"] == "simpleitk"
    assert explicit["n4_backend"] == "torch" and explicit["n4_execution"] == "isolated"
    assert {k: v for k, v in default.items() if not k.startswith("n4_")} == {
        k: v for k, v in explicit.items() if not k.startswith("n4_")}
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.installed.resolve())
    help_process = subprocess.run([sys.executable, "-m", "fnit.recon_all.native_free", "--help"],
        cwd="/tmp", env=environment, capture_output=True, text=True, check=True)
    for flag in ("--n4-backend", "--n4-execution", "--input-execution", "--backend"):
        assert flag in help_process.stdout
    assert not torch.cuda.is_initialized()
    report.update(status="passed", default_profile=default, explicit_torch_profile=explicit,
                  unchanged_non_n4_options=True, cli_help_exit_code=help_process.returncode,
                  cli_help_sha256=hashlib.sha256(help_process.stdout.encode()).hexdigest(),
                  cuda_initialized=False, full_workflow_measured=False,
                  overall_metric_equivalence="not_assessed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "wheel_bytes", "wheel_sha256", "cuda_initialized")}))


if __name__ == "__main__":
    main()
