"""Register a conventional sphere through sulc and smoothwm without FreeSurfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .mris_register_sulc_run import run_register_sulc
from .mris_register_smoothwm_run import run_register_smoothwm


def run_register_sphere(sphere: str | Path, smoothwm: str | Path,
                        sulc: str | Path, atlas: str | Path,
                        output: str | Path, *, overlap_device: str = "cpu",
                        averaging_device: str = "cpu",
                        require_unfolded: bool = False,
                        overlap_diagnostics: bool = False) -> dict:
    """Read ordered sphere/smoothwm meshes, per-vertex sulc and atlas TIFF.

    Write the registered sphere to output. Return input/output paths and
    hashes, temporary sulc-seed hash, device, nested sulc/smoothwm pass
    trajectories and timings, and total seconds; remove the temporary seed.
    averaging_device 默认 cpu，仅选择有序梯度平均的执行设备；CUDA 仍
    使用 float32，主体目标函数与步长决策留在 CPU，完整耗时含传输。
    overlap_device 单独控制末尾清理。输入是 surface RAS/mm、对应
    顶点顺序的 sulc 与半球 TIFF 图谱；网格不对应或不收敛时抛异常。
    require_unfolded 默认 False 保留通用参考入口；True 要求最终清理
    后 FP32 和双精度径向翻折均为零，失败在写输出前抛带诊断的
    SphereOverlapQualityError。overlap_diagnostics 默认 False，开启
    或严格检查时在 smoothwm_pass 内返回停止原因、末态计数和负面积
    mm²。严格检查不改变优化轨迹和源停止规则，不判定指标等效。
    """
    started = time.perf_counter()
    output = Path(output)
    input_hashes = {name: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                    for name, value in (("sphere", sphere), ("smoothwm", smoothwm),
                                        ("sulc", sulc), ("atlas", atlas))}
    with TemporaryDirectory(prefix="fnit-sphere-reg-", dir=output.parent) as temp:
        seed = Path(temp) / "sulc_seed"
        sulc_report = run_register_sulc(sphere, smoothwm, sulc, atlas, seed,
                                      averaging_device=averaging_device)
        seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
        smoothwm_report = run_register_smoothwm(
            sphere, smoothwm, seed, atlas, output,
            seed_iteration=sulc_report["last_iteration"],
            overlap_device=overlap_device, averaging_device=averaging_device,
            require_unfolded=require_unfolded,
            overlap_diagnostics=overlap_diagnostics)
    sulc_report["output"] = "(temporary sulc seed)"
    smoothwm_report["sulc_seed"] = "(temporary sulc seed)"
    return {"sphere": str(sphere), "smoothwm": str(smoothwm),
            "sulc": str(sulc), "atlas": str(atlas), "output": str(output),
            "overlap_device": overlap_device, "averaging_device": averaging_device,
            "require_unfolded": require_unfolded,
            "input_sha256": input_hashes,
            "sulc_seed_sha256": seed_hash,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sulc_pass": sulc_report,
            "smoothwm_pass": smoothwm_report,
            "total_seconds_including_io": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "sulc", "atlas", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--overlap-device", default="cpu")
    parser.add_argument("--averaging-device", default="cpu")
    parser.add_argument("--require-unfolded", action="store_true",
                        help="最终球面有翻折时失败，不写输出")
    parser.add_argument("--overlap-diagnostics", action="store_true",
                        help="记录实际停止原因与最终负面积")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_register_sphere(
        args.sphere, args.smoothwm, args.sulc, args.atlas, args.output,
        overlap_device=args.overlap_device, averaging_device=args.averaging_device,
        require_unfolded=args.require_unfolded,
        overlap_diagnostics=args.overlap_diagnostics)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
