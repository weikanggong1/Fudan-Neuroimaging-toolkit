"""运行固定 recon-all N4 Torch 实验 API；不会切换生产默认。"""
import argparse
import hashlib
import json
from pathlib import Path

from fnit.recon_all import n4_bspline_torch, n4_itk_torch_experimental


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="3D T1，NIfTI/MGH/MGZ")
    parser.add_argument("--output", type=Path, required=True, help="原网格 uint8 N4 输出")
    parser.add_argument("--report", type=Path, required=True, help="机器可读 JSON")
    parser.add_argument("--device", default="cuda:0", help="显式目标 GPU 或 cpu")
    parser.add_argument("--profile", action="store_true", help="同步剖析；增加计时开销")
    args = parser.parse_args()
    report = n4_itk_torch_experimental.correct_volume(input_path=args.input,
        output_path=args.output, device=args.device, profile=args.profile)
    paths = (Path(__file__), Path(n4_bspline_torch.__file__), Path(n4_itk_torch_experimental.__file__))
    report["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
