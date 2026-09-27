#!/usr/bin/env python3
"""Run FNIT MS-HBM inference from CBIG's saved binary session profiles."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from fnit.mshbm import load_assets, parcellate


def _normalized_profile(path: str, cortex_mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(loadmat(path)["profile_mat"])[cortex_mask].astype(np.float32)
    binary -= binary.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", binary, binary, optimize=True))
    binary /= np.maximum(norm[:, None], 1e-20)
    return binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", required=True,
                        help="CBIG profile_sessN.mat; repeat once per session")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--assets", help="Optional FNIT MS-HBM asset .npz")
    parser.add_argument("--w", type=float, default=200.0)
    parser.add_argument("--c", type=float, default=50.0)
    args = parser.parse_args()

    assets = load_assets(args.assets)
    profiles = [_normalized_profile(path, assets["cortex_mask"]) for path in args.profile]
    labels, history = parcellate(profiles=profiles, assets=assets, w=args.w, c=args.c)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, labels=labels,
                        history_json=np.asarray(json.dumps(history)))


if __name__ == "__main__":
    main()
