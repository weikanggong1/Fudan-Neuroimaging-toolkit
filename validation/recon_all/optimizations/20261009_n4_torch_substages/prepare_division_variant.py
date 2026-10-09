"""从冻结 N4 工作区生成独立 device-divisor 变体；不修改生产或已测 v1。

输入为含 src/fnit/recon_all 的工作区，输出必须是不存在的新目录。
复制本任务的两个模块、测试和 benchmark，再将三个 FP32 CPU scalar
除数改为同设备零维 Tensor。写出原始/候选 SHA；本脚本本身不运行算法。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"frozen source anchor changed: {before!r}")
    return text.replace(before, after, 1)


def prepare(*, source_workspace: Path, output_workspace: Path) -> dict:
    """输入工作区→新变体；路径错误或源码锚点改变抛异常，绝不覆盖现有目录。"""
    if output_workspace.exists():
        raise FileExistsError(output_workspace)
    relative_paths = [
        "src/fnit/recon_all/n4_bspline_torch.py",
        "src/fnit/recon_all/n4_itk_torch_experimental.py",
        "tests/recon_all/test_n4_bspline_torch.py",
        "tests/recon_all/test_n4_complete_torch.py",
        "validation/recon_all/optimizations/20261009_n4_torch_substages/benchmark_full.py",
        "validation/recon_all/optimizations/20261009_n4_torch_substages/benchmark_fit.py",
    ]
    originals = {relative: (source_workspace / relative).read_bytes() for relative in relative_paths}
    fitting = originals[relative_paths[0]].decode()
    fitting = replace_once(fitting, "    a = u.abs()\n", "    a = u.abs()\n    divisor = torch.scalar_tensor(6.0, dtype=torch.float32, device=u.device)\n")
    if fitting.count(" / 6.0\n") != 2:
        raise ValueError("expected two cubic division anchors")
    fitting = fitting.replace(" / 6.0\n", " / divisor\n")
    complete = originals[relative_paths[1]].decode()
    complete = replace_once(complete,
        "            u = (torch.arange(size, dtype=torch.float32, device=self.device) * float(spans)) / float(size - 1)\n",
        "            denominator = torch.scalar_tensor(size - 1, dtype=torch.float32, device=self.device)\n            u = (torch.arange(size, dtype=torch.float32, device=self.device) * float(spans)) / denominator\n")
    complete = replace_once(complete,
        "        self.distance = torch.minimum(self.n, 512.0 - self.n)\n",
        "        self.distance = torch.minimum(self.n, 512.0 - self.n)\n        self.histogram_divisor = torch.scalar_tensor(199.0, dtype=torch.float32, device=self.device)\n")
    complete = replace_once(complete, "        slope = (maximum - minimum) / 199.0\n",
        "        slope = (maximum - minimum) / self.histogram_divisor\n")
    outputs = dict(originals)
    outputs[relative_paths[0]] = fitting.encode()
    outputs[relative_paths[1]] = complete.encode()
    output_workspace.mkdir(parents=True)
    for relative, data in outputs.items():
        destination = output_workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    manifest = {
        "variant": "device_fp32_divisor_same_n4_formula",
        "purpose": "diagnose CUDA CPU-scalar reciprocal transformation, not accepted production backend",
        "source_workspace": str(source_workspace.resolve()),
        "output_workspace": str(output_workspace.resolve()),
        "gpu_tested": False,
        "files": {relative: {
            "source_sha256": hashlib.sha256(originals[relative]).hexdigest(),
            "variant_sha256": hashlib.sha256(outputs[relative]).hexdigest(),
        } for relative in relative_paths},
    }
    (output_workspace / "division_variant_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output-workspace", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(source_workspace=arguments.source_workspace, output_workspace=arguments.output_workspace)


if __name__ == "__main__":
    main()
