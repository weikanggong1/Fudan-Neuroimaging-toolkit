"""旧 fnit-nuclei 命令的参数兼容入口，统一运行 PyTorch segment_subregions。"""

import argparse

import torch

from .pipeline import segment_subregions
from .setup import prepare_subregion_atlases, save_subregion_root


_STRUCTURES = {"brainstem": "brainstem", "thalamus": "thalamus",
               "hippo-left": "hippo-amygdala-left", "hippo-right": "hippo-amygdala-right",
               "hippo-amygdala-left": "hippo-amygdala-left",
               "hippo-amygdala-right": "hippo-amygdala-right"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="准备并校验统一亚区图谱")
    setup.add_argument("--atlas-root", required=True)
    setup.add_argument("--asset-dir")
    setup.add_argument("--device", default="cpu", help="图谱准备设备，默认 CPU")
    run = subparsers.add_parser("run", help="从原始 T1 或已有 norm/aseg/wmparc 运行统一亚区分割")
    images = run.add_mutually_exclusive_group(required=True)
    images.add_argument("--t1", help="原始 T1；未给 aseg 时自动运行 SynthSeg")
    images.add_argument("--norm", help="旧强度图参数名，等同 --t1")
    run.add_argument("--aseg", help="可选粗分割，映射到 coarse_segmentation")
    run.add_argument("--wmparc", help="可选白质分区")
    run.add_argument("--atlas-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--structure", action="append", choices=tuple(_STRUCTURES),
                     help="可重复；旧 hippo-left/right 自动映射，默认全部四项结构")
    run.add_argument("--device", default="cuda:0", help="cpu 或 CUDA 设备，默认 cuda:0")
    run.add_argument("--optimization", choices=("fast", "balanced"), default="fast")
    run.add_argument("--threads", type=int, default=4, help="PyTorch CPU 线程数，必须至少 1")
    args = parser.parse_args(argv)
    if args.command == "setup":
        root = prepare_subregion_atlases(args.atlas_root, asset_dir=args.asset_dir, device=args.device)
        save_subregion_root(root)
        print(root)
        return
    if args.threads < 1:
        parser.error("--threads 必须至少为 1")
    torch.set_num_threads(args.threads)
    result = segment_subregions(
        args.t1 or args.norm, args.atlas_root,
        coarse_segmentation=args.aseg, wmparc=args.wmparc,
        structures=tuple(_STRUCTURES[name] for name in args.structure) if args.structure else "all",
        device=args.device, optimization=args.optimization,
        output_dir=args.output_dir, save_highres=True,
    )
    print(result.labels.get_filename() or args.output_dir)


if __name__ == "__main__":
    main()
