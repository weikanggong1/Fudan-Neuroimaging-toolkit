"""Build a reusable packed cache from existing real-data subject NPY files."""

import argparse
import json
import resource
from pathlib import Path

import numpy as np

from fnit.bwas.cache import build_packed_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, required=True)
    parser.add_argument("--subject-block-size", type=int, default=16)
    args = parser.parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < args.subjects+128:
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.subjects+128, hard))
    matrices = [np.load(args.source_dir / f"subject-{index}.npy", mmap_mode="r")
                for index in range(args.subjects)]
    seconds = build_packed_cache(matrices, args.packed_dir, args.subject_block_size)
    bytes_written = sum(path.stat().st_size for path in args.packed_dir.glob("*.npy"))
    print(json.dumps({"subjects": args.subjects, "voxels": matrices[0].shape[0],
                      "subject_block_size": args.subject_block_size,
                      "seconds": seconds, "packed_bytes": bytes_written}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
