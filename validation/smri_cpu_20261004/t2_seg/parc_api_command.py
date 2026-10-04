"""Time existing SynthSegPlus API options absent from the shared FNIT CLI."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--keep-geometry", action="store_true")
    args = parser.parse_args()
    import torch
    from fnit import SynthSegPlus
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    model = SynthSegPlus(weights=args.weights, parc_weights=args.weights,
                         device=args.device)
    result = model(args.input, fast=args.fast, keep_geometry=args.keep_geometry,
                   volumes=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.combined.save(args.output)
    result.write_volumes_csv(args.input, args.csv)


if __name__ == "__main__":
    main()
