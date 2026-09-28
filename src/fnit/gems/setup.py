"""Download the small BrainstemSS atlas and prepare PyTorch alpha resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from ..recon_all.assets import configured_dir, download_asset
from ..weights import cache_dir
from .atlas import GEMSAtlas
from .smoothing import smooth_atlas_alphas


LABEL_CLASSES = [10, 12, 0, 11, 1, 0, 2, 0, 9, 2, 8, 1, 0, 2, 0, 7, 1, 6, 5, 4, 3]
ATLAS_FILES = ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt")


def prepare_brainstem_atlas(
    output_root: str | Path,
    *,
    asset_dir: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    """Create ``output_root/brainstem`` from three verified FreeSurfer data files.

    ``output_root`` is the later ``segment_subregions(atlas_root=...)`` argument.
    ``asset_dir`` stores downloaded assets; existing hash-valid files are reused.
    ``device`` runs atlas-prior preparation on CPU or a chosen CUDA device.
    """
    source_root = Path(asset_dir or configured_dir() or cache_dir() / "recon_all_assets")
    target = Path(output_root) / "brainstem"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ATLAS_FILES:
        source = download_asset(f"average/BrainstemSS/atlas/{filename}", source_root)
        shutil.copy2(source, target / filename)

    atlas = GEMSAtlas.from_freesurfer(target / "AtlasMesh.gz",
                                      target / "compressionLookupTable.txt")
    classes = np.asarray(LABEL_CLASSES, dtype=np.int64)
    for sigma in (2, 1):
        np.save(target / f"sigma{sigma}.npy",
                smooth_atlas_alphas(atlas, classes, sigma, device=device))
    config = {
        "include_label_ids": [173, 174, 175, 178],
        "alignment_target_label_ids": [16],
        "label_classes": LABEL_CLASSES,
        "segmentation_fit": "brainstem",
        "segmentation_fit_iterations": 40,
        "working_resolution_mm": 0.5,
        "mask_to_atlas": True,
        "block_size": 12,
        "em_iterations": 25,
        "deform_lr": 0.12,
        "deform_em_interval": 10,
        "fit_alpha_files": ["sigma2.npy", "sigma1.npy", None],
        "fit_stage_iterations": [20, 20, 20],
    }
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory that will contain the brainstem atlas pack")
    parser.add_argument("--asset-dir", type=Path,
                        help="Verified download cache; defaults to configured recon-all assets")
    parser.add_argument("--device", default="cpu", help="CPU or CUDA device for alpha smoothing")
    args = parser.parse_args(argv)
    print(prepare_brainstem_atlas(args.output_root, asset_dir=args.asset_dir,
                                  device=args.device))


if __name__ == "__main__":
    main()
