"""私有benchmark参考包SHA验证和独立RPATH部署，不修改共享Conda环境。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patchelf", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.archive) != args.expected_sha256:
        raise ValueError("reference archive SHA mismatch")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    prefix = args.output / "reference_prefix"
    prefix.mkdir(mode=0o700)
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError("invalid reference archive member")
        archive.extractall(prefix)
    manifest = json.loads((prefix / "deployment.json").read_text())
    for name, data in manifest["libraries"].items():
        if sha(prefix / "lib" / name) != data["sha256"]:
            raise ValueError("reference library SHA mismatch")
    environment = os.environ.copy()
    # 仅此参考子进程：排除SSH继承的其他Conda库；不是全局精度/缓存设置。
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = str(prefix / "lib")
    report = {"status": "running", "scope": "benchmark only; fixed source compiled mains and declared Conda libraries",
              "archive_sha256": args.expected_sha256, "source_commit": manifest["source_commit"],
              "programs": {}, "libraries": manifest["libraries"],
              "patchelf_sha256": sha(args.patchelf), "script_sha256": sha(Path(__file__)),
              "child_library_policy": "private reference/lib only plus system base libraries; no LD_PRELOAD"}
    for name, data in manifest["programs"].items():
        binary = prefix / "bin" / name
        if sha(binary) != data["before_relocation_sha256"]:
            raise ValueError("reference program SHA mismatch")
        subprocess.run([str(args.patchelf), "--force-rpath", "--set-rpath", "$ORIGIN/../lib", str(binary)], check=True)
        dependencies = subprocess.check_output(["ldd", str(binary)], env=environment, text=True)
        if "not found" in dependencies:
            raise ValueError("reference dependency missing")
        for line in dependencies.splitlines():
            if "=> /" in line:
                path = line.split("=>", 1)[1].split()[0]
                resolved = Path(path).resolve()
                in_private = resolved.is_relative_to((prefix / "lib").resolve())
                in_system = any(resolved.is_relative_to(Path(base).resolve())
                                for base in ["/lib", "/lib64", "/usr/lib", "/usr/lib64"])
                if not in_private and not in_system:
                    raise ValueError("reference dependency escaped private prefix/system base")
        (args.output / (name + ".ldd")).write_text(dependencies)
        launch = subprocess.run([str(binary), "--help"], env=environment, capture_output=True, text=True, timeout=15)
        (args.output / (name + ".launch.log")).write_text(launch.stdout + launch.stderr)
        report["programs"][name] = {**data, "after_relocation_sha256": sha(binary),
                                    "help_returncode": launch.returncode, "ldd_verified": True}
    report["status"] = "complete"
    (args.output / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REFERENCE_RUNTIME_COMPLETE")


if __name__ == "__main__":
    main()
