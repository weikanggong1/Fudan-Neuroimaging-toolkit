"""新进程调用独立target的实际CLI help，记录CUDA初始化前后及wheel扩展清单。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="已经独立安装的FNIT target目录")
    parser.add_argument("--output", type=Path, required=True, help="私有JSON输出文件")
    args = parser.parse_args()
    tick = time.perf_counter()
    import torch
    initialized_before = torch.cuda.is_initialized()
    assert initialized_before is False
    dist = importlib.metadata.distribution("fudan-neuroimaging-toolkit")
    assert Path(dist.locate_file("")).resolve() == args.target.resolve()
    entry = next(e for e in dist.entry_points if e.group == "console_scripts" and e.name == "fnit-recon-all")
    function = entry.load()
    old_arguments = sys.argv
    sys.argv = ["fnit-recon-all", "--help"]
    try:
        try:
            returned = function()
        except SystemExit as error:
            returned = error.code
        assert returned in (0, None)
    finally:
        sys.argv = old_arguments
    initialized_after = torch.cuda.is_initialized()
    assert initialized_after is False
    run_root = args.target.parent
    binding_raw = (run_root/"SOURCE_BINDING.private.json").read_bytes()
    check = json.loads((run_root/"CHECK.private.json").read_bytes())
    wheel = run_root/"wheel"/check["wheel"]["name"]
    data = wheel.read_bytes()
    assert hashlib.sha256(data).hexdigest() == check["wheel"]["sha256"]
    with zipfile.ZipFile(wheel) as archive:
        extensions = [n for n in archive.namelist() if n.endswith((".so", ".dll", ".dylib", ".pyd"))]
    assert len(extensions) == 1 and extensions[0].startswith("fnit/msm/_fastpd_native.")
    result = {"status":"complete", "scope":"CLI help and wheel content only; noMRI/runtimeOSisolation",
              "source_binding_sha256":hashlib.sha256(binding_raw).hexdigest(),
              "wheel_sha256":hashlib.sha256(data).hexdigest(),
              "entrypoint":entry.value, "CLI_help_exit_code":returned,
              "cuda_initialized_before":initialized_before,"cuda_initialized_after":initialized_after,
              "wheel_compiled_extensions":extensions,
              "extension_use_in_recon_all_help":False,
              "MSM_scope":"separate project feature; declared setup compiler only; noFSstage binary built",
              "audit_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "wall_seconds_including_torch_import_help_wheel_read":time.perf_counter()-tick}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
