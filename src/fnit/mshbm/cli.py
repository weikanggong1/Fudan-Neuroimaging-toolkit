"""Command-line runner for standalone fsLR32k cortical MS-HBM."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .core import load_assets, parcellate, profiles_from_timeseries, _profile
from .output import save_results
from .volume import parcellate_volume


def read_cortex(path, mask):
    """Read T×59412 or T×64984 NPY, or fsLR32k CIFTI dtseries."""
    path = Path(path)
    if path.name.endswith(".npy"):
        series = np.load(path, mmap_mode="r", allow_pickle=False)
        if series.ndim != 2:
            raise ValueError(f"Not a 2-D time series: {path}")
        if series.shape[1] in (59412, 64984):
            pass
        elif series.shape[0] in (59412, 64984):
            series = series.T
        else:
            raise ValueError(f"Expected fsLR32k cortical vertices in {path}")
        if series.shape[1] == 64984:
            series = series[:, mask]
        return np.asarray(series, dtype=np.float32)
    if not path.name.endswith(".dtseries.nii"):
        raise ValueError("Input must be .npy or .dtseries.nii")
    image = nib.load(str(path))
    axis = image.header.get_axis(1)
    full = np.zeros((image.shape[0], 64984), dtype=np.float32)
    for name, part, brain_model in axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            offset = 0
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            offset = 32492
        else:
            continue
        if brain_model.nvertices[name] != 32492:
            raise ValueError("CIFTI cortex is not fsLR32k")
        full[:, offset + brain_model.vertex] = np.asarray(image.dataobj[:, part], dtype=np.float32)
    return full[:, mask]


def main(argv=None):
    parser = argparse.ArgumentParser(description="CBIG Kong2019 single-subject MS-HBM, 17 networks")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--volume", help="4D MNI BOLD NIfTI; requires anatomical surfaces and a cortical mask")
    inputs.add_argument("--timeseries", nargs="+",
                        help="One or more T×59412 fsLR32k cortical .npy/.dtseries.nii files")
    parser.add_argument("--censor", nargs="*", help="Optional 0/1 text file per input")
    parser.add_argument("--left-surface", help="Left fsLR32k midthickness in the volume scanner-RAS space")
    parser.add_argument("--right-surface", help="Right fsLR32k midthickness in the volume scanner-RAS space")
    parser.add_argument("--cortical-mask", help="3D cortical mask on the volume voxel grid")
    parser.add_argument("--device", default="cuda:0", help="Volume sampling/mapping device; MS-HBM remains CPU")
    parser.add_argument("--frame-chunk", type=int, default=8, help="GPU volume frames per chunk")
    parser.add_argument("--max-distance-mm", type=float, default=3.0, help="Maximum voxel-to-midthickness mapping distance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets", help="Alternate HCP_40 fsLR32k asset .npz")
    parser.add_argument("--w", type=float, default=200.0, help="CBIG spatial-prior weight")
    parser.add_argument("--c", type=float, default=50.0, help="CBIG MRF weight")
    args = parser.parse_args(argv)
    if args.censor is not None and len(args.censor) != (1 if args.volume else len(args.timeseries)):
        parser.error("--censor requires one path per --timeseries file")
    assets = load_assets(args.assets)
    mask = assets["cortex_mask"]
    if args.volume:
        if not all((args.left_surface, args.right_surface, args.cortical_mask)):
            parser.error("--volume requires --left-surface, --right-surface and --cortical-mask")
        censor = np.loadtxt(args.censor[0]) if args.censor else None
        labels, history = parcellate_volume(
            args.volume, args.left_surface, args.right_surface, args.cortical_mask,
            args.output_dir, assets=assets, censor=censor, w=args.w, c=args.c,
            device=args.device, frame_chunk=args.frame_chunk,
            max_distance_mm=args.max_distance_mm,
        )
        profiles = [None, None]
    else:
        profiles = []
        sessions = []
        for i, path in enumerate(args.timeseries):
            series = read_cortex(path, mask)
            censor = np.loadtxt(args.censor[i]) if args.censor is not None else None
            sessions.append(series if censor is None else series[np.asarray(censor).astype(bool)])
            if len(args.timeseries) == 1:
                profiles = profiles_from_timeseries(series, assets, censor)
            else:
                if censor is not None:
                    if censor.shape != (len(series),) or not np.isin(censor, [0, 1]).all():
                        raise ValueError("Invalid censor vector")
                    series = series[censor.astype(bool)]
                if not np.isfinite(series).all():
                    raise ValueError("Time series contains nonfinite values")
                positions = np.searchsorted(np.flatnonzero(mask), assets["seed_vertices"])
                profiles.append(_profile(series, positions))
        labels, history = parcellate(profiles, assets, w=args.w, c=args.c)
        save_results(args.output_dir, labels, np.concatenate(sessions), mask)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps({
        "method": "CBIG Kong2019 MS-HBM Python CPU port",
        "source_commit": "b69b822a15e2a94f1e439606552fc44b6858cf3c",
        "timeseries": [str(Path(p).resolve()) for p in (args.timeseries or [])],
        "volume": str(Path(args.volume).resolve()) if args.volume else None,
        "projection": ({"left_surface": args.left_surface, "right_surface": args.right_surface,
                        "cortical_mask": args.cortical_mask, "device": args.device,
                        "frame_chunk": args.frame_chunk,
                        "max_distance_mm": args.max_distance_mm,
                        "method": "trilinear midthickness sampling + masked nearest vertex"}
                       if args.volume else None),
        "censor": args.censor, "w": args.w, "c": args.c,
        "sessions": len(profiles), "history": history,
        "network_count": 17, "cortical_vertices": int(mask.sum()),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
