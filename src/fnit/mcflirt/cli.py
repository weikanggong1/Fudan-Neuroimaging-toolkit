"""单被试 MCFLIRT 命令行。"""
import argparse


def add_arguments(parser):
    parser.add_argument("-in", "--in", dest="input", required=True, help="原始四维 BOLD NIfTI")
    parser.add_argument("-out", "--out", dest="output", required=True, help="校正图输出路径或前缀")
    parser.add_argument("-reffile", dest="reference", help="同网格的三维参考图；默认中间帧")
    parser.add_argument("-mats", action="store_true", help="输出每帧 input→reference 的 MAT_####")
    parser.add_argument("-plots", action="store_true", help="输出六列弧度/mm 的 .par")
    parser.add_argument("-rmsrel", action="store_true", help="输出相邻帧80mm球体RMS位移")
    parser.add_argument("-rmsabs", action="store_true", help="输出各帧相对identity的80mm球体RMS位移")
    parser.add_argument("-stages", type=int, choices=(1, 2, 3), default=3, help="8/4/4 mm 阶段数")
    interpolation = parser.add_mutually_exclusive_group()
    interpolation.add_argument("-spline_final", action="store_true", help="最终使用三阶 B-spline")
    interpolation.add_argument("-trilinear_final", action="store_true", help="最终使用三线性（原 MCFLIRT 默认）")
    parser.add_argument("--device", default=None, help="cpu、cuda 或 cuda:1")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser


def run_from_args(args):
    from .core import TorchMCFLIRT
    return TorchMCFLIRT(device=args.device).run(
        args.input, args.reference, output=args.output, mats=args.mats, plots=args.plots,
        stages=args.stages, interpolation="spline" if args.spline_final else "linear",
        overwrite=args.overwrite, rmsrel=args.rmsrel, rmsabs=args.rmsabs)


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(prog="fnit mcflirt", allow_abbrev=False))
    args = parser.parse_args(argv)
    try:
        run_from_args(args)
    except (FileNotFoundError, FileExistsError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
