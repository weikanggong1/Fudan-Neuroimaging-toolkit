"""完整标准WM/aseg的纯Python/Numba文件入口，无FreeSurfer程序依赖。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .edit_wm_aseg_core_python import write_wm_asegedit_profile


def main(argv: list[str] | None = None) -> None:
    """具名CLI：四个同网格输入、新WM输出及可选JSON；参数见中文专页。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wm-file", "brain-file", "aseg-file", "entowm-file", "output-file"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--no-fill-seg-wm", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = write_wm_asegedit_profile(wm_file=args.wm_file, brain_file=args.brain_file,
        aseg_file=args.aseg_file, entowm_file=args.entowm_file, output_file=args.output_file,
        fill_seg_wm=not args.no_fill_seg_wm)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
