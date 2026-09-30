"""验证已初始化 CUDA 的 API 错误路径；此检查不执行或代表完整重建。"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from fnit.recon_all.native_free import run_recon_all_python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "subject", "weights", "assets", "native-bin", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.subject.mkdir(parents=True, exist_ok=False)
    retained = torch.empty(1024 * 1024, device=args.device, dtype=torch.float32)
    torch.cuda.synchronize(args.device)
    calls = []
    names = ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved")
    originals = {name: getattr(torch.cuda, name) for name in names}

    def tracked(name):
        def invoke(device=None):
            calls.append({"function": name, "device": str(device)})
            assert device is not None and torch.device(device) == torch.device(args.device)
            return originals[name](device)
        return invoke

    # 主调度真实执行前置检查；在首个计算阶段注入预期错误以检查失败报告。
    with patch("fnit.recon_all.input_talairach_chain.run_input_talairach_chain",
               side_effect=RuntimeError("expected CUDA reporting regression probe")), \
            patch.object(torch.cuda, names[0], tracked(names[0])), \
            patch.object(torch.cuda, names[1], tracked(names[1])), \
            patch.object(torch.cuda, names[2], tracked(names[2])):
        try:
            run_recon_all_python(
                t1=args.t1, subject_dir=args.subject, weights_dir=args.weights,
                assets_dir=args.assets, native_bin_dir=args.native_bin,
                device=args.device, threads=4)
        except RuntimeError as error:
            assert str(error) == "expected CUDA reporting regression probe"
        else:
            raise AssertionError("expected failure was not raised")
    report = json.loads((args.subject / "fnit-native-free-run.json").read_text())
    assert report["status"] == "failed"
    assert report["gpu_peak_allocated_bytes"] >= retained.numel() * retained.element_size()
    assert {row["function"] for row in calls} == set(names)
    source = Path(__import__("fnit.recon_all.native_free", fromlist=["__file__"]).__file__)
    result = {"pass": True, "scope": __doc__, "device": args.device, "calls": calls,
              "cuda_initialized_before_api": True,
              "gpu_peak_allocated_bytes": report["gpu_peak_allocated_bytes"],
              "gpu_peak_reserved_bytes": report["gpu_peak_reserved_bytes"],
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
