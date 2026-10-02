"""等待私有原生安装完成，再核验资源并顺序运行两例安装产物整例。

--round 为服务器本轮目录；--python 为主页Conda的Python；
--lock 为同主机共用锁；--resource-script 为已有capture_serial_resources.py。
输入原生status.json及installed-native-optimizations.json、baseline expected
清单和两例candidate配置；输出coordinator/candidate_ready_validation.json、
candidate_resources.json及candidate_queue_v1/。原生failed明确停止；
资源或程序SHA不符停止，不创建生产输出。不读官方被试目录。
完整计时由既有monitor/launcher记录；等待、资源核验、比较不计重建墙钟。
这是开发验证调度器，无独立官方等价CLI；运行语义见RUN_VALIDATION.md。
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--resource-script", type=Path, required=True)
    args = parser.parse_args()
    root = args.round.resolve()
    coordinator = root / "coordinator"
    ready_path = coordinator / "candidate_ready_validation.json"
    if ready_path.exists():
        raise FileExistsError(ready_path)
    tick = time.monotonic()
    installation = root / "private_install_v1"
    while True:
        try:
            status = json.loads((installation / "status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {}
        if status.get("status") == "passed":
            break
        if status.get("status") == "failed":
            raise RuntimeError("private native installation failed; preserve reports")
        if time.monotonic() - tick > 6*3600:
            raise TimeoutError("native dependency unavailable after six hours")
        time.sleep(20)
    manifest_path = installation / "installed-native-optimizations.json"
    manifest = json.loads(manifest_path.read_text())
    final_check = json.loads((root / "candidate_install_8d750e2/installed-source-check.json").read_text())
    if final_check["status"] != "passed" or final_check["compared_recon_all_py_files"] != 171:
        raise ValueError("final installed wheel/source checks failed")
    native = installation / "native_bundle/bin"
    expected = json.loads((coordinator / "expected_resources.json").read_text())
    native_expected = {}
    for name, value in manifest["programs"].items():
        path = native / name
        if path.resolve() != Path(value["path"]).resolve() or sha(path) != value["sha256"]:
            raise ValueError("native program manifest mismatch " + name)
        native_expected[name] = {"size_bytes": path.stat().st_size, "sha256": value["sha256"]}
        if name in expected["binaries"] and not value.get("rebuilt", False):
            if expected["binaries"][name]["sha256"] != value["sha256"]:
                raise ValueError("unchanged baseline program differs " + name)
    if len(native_expected) != 15:
        raise ValueError("expected fourteen existing programs and dedicated white optimizer")
    expected["binaries"] = native_expected
    expected_path = coordinator / "candidate_expected_resources.json"
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    config = json.loads((coordinator / "capture_baseline_resources.json").read_text())
    config.update(expected_manifest=str(expected_path),
                  code_commit="8d750e25d4d067a43edb788a96b2086a1c031ba0",
                  source_archive_sha256="45cb2a8260883f58c5640af82b0903ed9d6f3339936730385dc198434d7e88c9")
    config["roots"]["binaries"] = str(native)
    config_path = coordinator / "capture_candidate_resources.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "status": "validating", "dependency_wait_seconds": time.monotonic() - tick,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__),
        "installed_manifest_sha256": sha(manifest_path),
        "wheel_sha256": final_check["wheel_sha256"],
        "installed_recon_source_matches": 171,
        "native_binaries": native_expected,
        "isolation": "not_verified",
    }
    ready_path.write_text(json.dumps(report, indent=2) + "\n")
    env = dict(os.environ)
    for name in ["OMP_NUM_THREADS","NUMBA_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
        env[name] = "4"
    with args.lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        env["PYTHONPATH"] = str(root / "candidate_install_8d750e2/installed")
        verify = ("from fnit.recon_all.native_runtime_selection import select_native_optimizations;"
                  "import json; r=select_native_optimizations(" + repr(str(native)) + ",threads=4);"
                  "assert r['em_backend']=='cpu_cached';"
                  "assert r['white_binary'].endswith('/mris_place_surface_white_fast');"
                  "assert r['pial_binary'].endswith('/mris_place_surface');"
                  "print(json.dumps(r))")
        report["actual_native_selection"] = json.loads(subprocess.check_output(
            [str(args.python), "-c", verify], env=env, text=True))
        report["resource_script_sha256"] = sha(args.resource_script)
        subprocess.run([str(args.python), str(args.resource_script), "--config", str(config_path),
                        "--output", str(coordinator / "candidate_resources.json")],
                       env=env, check=True)
    report["status"] = "passed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    ready_path.write_text(json.dumps(report, indent=2) + "\n")
    queue = [str(args.python), str(coordinator / "run_whole_queue.py"), "--configs",
             str(coordinator / "candidate_sub01_8d750e2.json"),
             str(coordinator / "candidate_sub02_8d750e2.json"),
             "--lock", str(args.lock), "--output", str(coordinator / "candidate_queue_v1"),
             "--monitor", str(coordinator / "run_monitored.py"),
             "--launcher", str(coordinator / "execute_whole_case.py")]
    print(json.dumps({"ready": True, "queue": queue}), flush=True)
    subprocess.run(queue, env=env, check=True)

if __name__ == "__main__":
    main()
