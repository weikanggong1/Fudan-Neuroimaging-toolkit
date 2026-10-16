"""把两例拓扑 C++ 预检结果编排为一份尚未运行的单卡 GPU 租约。

输入：冻结 Python GA 原租约、两份通过的混合预检、同一 Conda 环境的
许可证路径、v4 租约调度器、全新私有作业目录。输出为 run_two_cases.sh、
job.private.json 和 run_lease.sh；不启动 GPU、不创建整例输出。

命令行所有路径均用具名参数。原程序只允许 Conda 独立源码构建的
mris_fix_topology_fnit；下游仍是 FNIT Python/Torch 算法。需等目标 GPU
没有其他计算进程时，才由 登录节点 tmux 执行生成的 run_lease.sh。失败
应保留原回执，新的原始 T1 整例使用新空目录，不在旧目录补跑。

原软件对应 mris_fix_topology -ga -seed 1234；本脚本没有官方独立 CLI。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_job(*, baseline_job: Path, preflight_sub06: Path,
                preflight_sub07: Path, preflight_script: Path,
                lease_runner: Path, conda_prefix: Path, fs_license: Path,
                output_dir: Path, run_id: str, lock_path: Path,
                shared_gpu_explicit: bool = False,
                cpu_affinity: list[int] | None = None) -> dict:
    """返回两例原始 T1 整例的未执行租约配置；所有路径为绝对路径。

    输入预检必须绑定相同源码、程序和目标 GPU；输出目录必须不存在。
    出错不改变预检或整例数据；成功只创建私有作业配置与启动脚本。
    """
    baseline_job = Path(baseline_job).resolve()
    preflight_paths = [Path(preflight_sub06).resolve(), Path(preflight_sub07).resolve()]
    preflight_script = Path(preflight_script).resolve()
    lease_runner = Path(lease_runner).resolve()
    prefix = Path(conda_prefix).resolve()
    license_path = Path(fs_license).resolve()
    output = Path(output_dir).resolve()
    lock = Path(lock_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("run_id must contain only letters, digits, hyphen or underscore")
    old = json.loads(baseline_job.read_text())
    reports = [json.loads((path / "provenance.json").read_text()) for path in preflight_paths]
    if any(row["status"] != "prepared_not_run" for row in reports):
        raise ValueError("both hybrid preflights must have passed without running")
    common_keys = ("commit", "benchmark_sha256")
    if any(row["source"][key] != reports[0]["source"][key]
           for row in reports[1:] for key in common_keys):
        raise ValueError("the two cases do not use the same frozen source")
    for row in reports:
        if row["baseline_job_sha256"] != _sha(baseline_job):
            raise ValueError("preflight baseline job has changed")
        if row["native_program"] != reports[0]["native_program"]:
            raise ValueError("the two cases selected different topology programs")
        if row["gpu_selection"]["target_uuid"] != reports[0]["gpu_selection"]["target_uuid"]:
            raise ValueError("the two cases target different GPUs")
        if row["fs_license_sha256"] != _sha(license_path):
            raise ValueError("FS_LICENSE differs from the completed preflight")
        if Path(row["run_output_root"]).exists():
            raise FileExistsError(row["run_output_root"])
    targets = [Path(row["run_output_root"]) for row in reports]
    if (targets[0].parent != targets[1].parent or targets[0].name != "ds000114_sub-06"
            or targets[1].name != "ds000114_sub-07" or targets[0].parent.exists()):
        raise ValueError("two outputs must be new sub-06/sub-07 children of one empty run root")
    gpu = reports[0]["gpu_selection"]
    if gpu["target_physical_index"] is None:
        raise ValueError("lease requires an explicit physical GPU index")
    if not old.get("require_idle_gpu") or old.get("threads") != 4:
        raise ValueError("baseline lease must enforce idle GPU and four total threads")
    if cpu_affinity is not None:
        if (len(cpu_affinity) != 4 or len(set(cpu_affinity)) != 4
                or not set(cpu_affinity).issubset(os.sched_getaffinity(0))):
            raise ValueError("four distinct CPU affinity cores must be allowed by this host")
    if shared_gpu_explicit and "shared_gpu_lease" not in lease_runner.name:
        raise ValueError("shared GPU job requires the audited shared GPU lease runner")
    if not all(path.is_file() for path in (preflight_script, lease_runner, license_path)):
        raise FileNotFoundError("preflight script, lease runner or license is missing")
    for path, row in zip(preflight_paths, reports):
        if not (path / "launch.sh").is_file():
            raise FileNotFoundError(path / "launch.sh")
        if (row["native_program"]["binary_sha256"]
                != _sha(Path(row["native_program"]["binary"]))):
            raise ValueError("topology binary changed after preflight")
    receipt = targets[0].parent.parent / f"lease_receipt_{run_id}"
    if receipt.exists() or lock.with_suffix(lock.suffix + ".unresolved.json").exists():
        raise FileExistsError("receipt or unresolved GPU lock already exists")
    case_script = output / "run_two_cases.sh"
    job_path = output / "job.private.json"
    lease_script = output / "run_lease.sh"
    case_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "umask 077",
                  *("bash " + shlex.quote(str(path / "launch.sh")) for path in preflight_paths)]
    case_text = "\n".join(case_lines) + "\n"
    job = dict(old)
    job.update(run_id=run_id,
               scope="two raw-T1 empty-directory hybrid runs; only topology GA uses FNIT Conda C++; 138 and metrics separately assessed",
               expected_gpu_uuid=gpu["target_uuid"],
               cuda_visible_devices=str(gpu["target_physical_index"]),
               argv=["/bin/bash", str(case_script)],
               receipt_dir=str(receipt), new_output_paths=[str(targets[0].parent)],
               require_idle_gpu=not shared_gpu_explicit,
               shared_gpu_explicit=shared_gpu_explicit)
    if cpu_affinity is not None:
        job["cpu_affinity"] = cpu_affinity
    job["env"] = {key: value for key, value in old["env"].items()
                  if key not in {"FNIT_RECON_AUDIT_SOURCE_DIR", "FNIT_RECON_FORBID_ORIGINAL_STAGES"}}
    job["env"]["FS_LICENSE"] = str(license_path)
    job["unset_env"] = [name for name in old.get("unset_env", []) if name != "FS_LICENSE"]
    expected = {row["path"]: row["sha256"] for row in old["expected_files"]}

    def add(path: Path, checksum: str | None = None) -> None:
        path = Path(path).resolve()
        value = checksum or _sha(path)
        existing = expected.get(str(path))
        if existing is not None and existing != value:
            raise ValueError(f"expected file hash conflict: {path}")
        expected[str(path)] = value

    add(baseline_job)
    add(preflight_script)
    add(lease_runner)
    add(license_path)
    add(Path(reports[0]["native_program"]["binary"]),
        reports[0]["native_program"]["binary_sha256"])
    for path, row in zip(preflight_paths, reports):
        add(path / "provenance.json")
        add(path / "launch.sh")
        base = Path(row["baseline_benchmark"])
        add(base, row["baseline_benchmark_sha256"])
        benchmark = json.loads(base.read_text())
        for kind, root in (("weights", Path(benchmark["cli_command"][benchmark["cli_command"].index("--weights-dir") + 1])),
                           ("assets", Path(benchmark["cli_command"][benchmark["cli_command"].index("--assets-dir") + 1]))):
            for relative, checksum in benchmark["resource_sha256"][kind].items():
                add(root / relative, checksum)
    output.mkdir(mode=0o700)
    case_script.write_text(case_text)
    case_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    add(case_script)
    lease_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join([
        str(prefix / "bin/python"), str(lease_runner), "--job", str(job_path),
        "--lock", str(lock)]) + "\n")
    lease_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    add(lease_script)
    job["expected_files"] = [{"path": path, "sha256": checksum}
                             for path, checksum in sorted(expected.items())]
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    return {"status": "prepared_not_run", "run_id": run_id,
            "job": str(job_path), "job_sha256": _sha(job_path),
            "runner_launch": str(lease_script), "runner_launch_sha256": _sha(lease_script),
            "target_gpu_uuid": gpu["target_uuid"], "target_physical_index": gpu["target_physical_index"],
            "expected_file_count": len(job["expected_files"]),
            "new_output_paths": job["new_output_paths"],
            "receipt_dir": str(receipt), "lock": str(lock),
            "require_idle_gpu": not shared_gpu_explicit,
            "shared_gpu_explicit": shared_gpu_explicit,
            "cpu_affinity": job["cpu_affinity"],
            "execution": "not_run"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-job", type=Path, required=True)
    parser.add_argument("--preflight-sub06", type=Path, required=True)
    parser.add_argument("--preflight-sub07", type=Path, required=True)
    parser.add_argument("--preflight-script", type=Path, required=True)
    parser.add_argument("--lease-runner", type=Path, required=True)
    parser.add_argument("--conda-prefix", type=Path, required=True)
    parser.add_argument("--fs-license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--shared-gpu-explicit", action="store_true")
    parser.add_argument("--cpu-affinity", type=int, nargs=4)
    args = parser.parse_args()
    result = prepare_job(baseline_job=args.baseline_job,
                         preflight_sub06=args.preflight_sub06,
                         preflight_sub07=args.preflight_sub07,
                         preflight_script=args.preflight_script,
                         lease_runner=args.lease_runner,
                         conda_prefix=args.conda_prefix, fs_license=args.fs_license,
                         output_dir=args.output_dir, run_id=args.run_id,
                         lock_path=args.lock_path,
                         shared_gpu_explicit=args.shared_gpu_explicit,
                         cpu_affinity=args.cpu_affinity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
