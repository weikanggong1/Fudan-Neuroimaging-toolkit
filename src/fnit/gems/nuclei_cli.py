"""丘脑、海马及杏仁核图谱下载与横断面分割命令。"""

import argparse

from .nuclei import prepare_nuclei_atlas, segment_nuclei


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="下载并校验两个图谱")
    setup.add_argument("--atlas-root", required=True)
    setup.add_argument("--asset-dir")
    run = subparsers.add_parser("run", help="分割已有 norm、aseg、wmparc 的单受试者")
    run.add_argument("--norm", required=True)
    run.add_argument("--aseg", required=True)
    run.add_argument("--wmparc", required=True)
    run.add_argument("--atlas-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--structure", action="append", choices=("thalamus", "hippo-left", "hippo-right"))
    run.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "setup":
        print(prepare_nuclei_atlas(atlas_root=args.atlas_root, asset_dir=args.asset_dir))
    else:
        result = segment_nuclei(
            norm=args.norm, aseg=args.aseg, wmparc=args.wmparc,
            atlas_root=args.atlas_root, output_dir=args.output_dir,
            structures=tuple(args.structure or ("thalamus", "hippo-left", "hippo-right")),
            threads=args.threads,
        )
        for name, files in result.items():
            print(name, files["labels"])


if __name__ == "__main__":
    main()
