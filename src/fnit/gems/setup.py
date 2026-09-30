"""Download the small BrainstemSS atlas and prepare PyTorch alpha resolutions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from ..recon_all.assets import ASSET_FILES, configured_dir, download_asset
from ..weights import cache_dir, verify_file
from .atlas import GEMSAtlas
from .smoothing import smooth_atlas_alphas


LABEL_CLASSES = [10, 12, 0, 11, 1, 0, 2, 0, 9, 2, 8, 1, 0, 2, 0, 7, 1, 6, 5, 4, 3]
ATLAS_FILES = ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt")


def configured_subregion_root() -> Path | None:
    config = cache_dir() / "subregion_atlases.json"
    if not config.is_file():
        return None
    root = Path(json.loads(config.read_text())["directory"])
    if not root.is_absolute():
        raise ValueError(f"Subregion atlas path must be absolute: {config}")
    return root


def save_subregion_root(root: str | Path) -> None:
    config = cache_dir() / "subregion_atlases.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.with_name(config.name + ".part")
    temporary.write_text(json.dumps({"directory": str(Path(root).resolve())}) + "\n")
    temporary.replace(config)


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
        key = f"average/BrainstemSS/atlas/{filename}"
        if not verify_file(target / filename, *ASSET_FILES[key][:2]):
            source = download_asset(key, source_root)
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


def prepare_subregion_atlases(output_root: str | Path | None = None, *,
                              asset_dir: str | Path | None = None,
                              device: str = "cpu") -> Path:
    """Prepare three verified official atlas families and their PyTorch recipes."""
    root = Path(output_root or configured_subregion_root() or cache_dir() / "subregion_atlases")
    root.mkdir(parents=True, exist_ok=True)
    prepare_brainstem_atlas(root, asset_dir=asset_dir, device=device)
    source_root = Path(asset_dir or configured_dir() or cache_dir() / "recon_all_assets")
    from .recipes.thalamus import ThalamusRecipe
    from .recipes.hippo_amygdala import HippoAmygdalaRecipe

    for pack, official in (("thalamus", "ThalamicNuclei"),
                           ("hippo-amygdala-left", "HippoSF")):
        target = root / pack
        target.mkdir(parents=True, exist_ok=True)
        for filename in ATLAS_FILES:
            key = f"average/{official}/atlas/{filename}"
            if not verify_file(target / filename, *ASSET_FILES[key][:2]):
                source = download_asset(key, source_root)
                if source.resolve() != (target / filename).resolve():
                    shutil.copy2(source, target / filename)
        recipe = (ThalamusRecipe(pack, target) if pack == "thalamus" else
                  HippoAmygdalaRecipe("left", target))
        # These alphas depend on the subject affine and working resolution.
        # Remove earlier population-grid caches; recipes now smooth in place.
        for pattern in ("seg-sigma*-first.npy", "image-sigma*.npy"):
            for path in target.glob(pattern):
                path.unlink()
        config = {"atlas_family": official, "working_resolution_mm": recipe.resolution_mm,
                  "segmentation_schedule": recipe.seg_schedule,
                  "intensity_schedule": recipe.image_schedule,
                  "alpha_smoothing": "transformed_reference_mesh"}
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = root / "hippo-amygdala-left"
    target = root / "hippo-amygdala-right"
    target.mkdir(exist_ok=True)
    for pattern in ("seg-sigma*-first.npy", "image-sigma*.npy"):
        for path in target.glob(pattern):
            path.unlink()
    for path in source.iterdir():
        if path.name == "config.json" or not path.is_file():
            continue
        destination = target / path.name
        destination.unlink(missing_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
    (target / "config.json").write_text(json.dumps({
        "atlas_family": "HippoSF", "side": "right",
        "working_resolution_mm": 0.33333,
        "alpha_smoothing": "transformed_reference_mesh",
        "segmentation_schedule": HippoAmygdalaRecipe.seg_schedule,
        "intensity_schedule": HippoAmygdalaRecipe.image_schedule}, indent=2) + "\n")
    return root


def subregion_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare all verified GEMS subregion atlases")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    root = prepare_subregion_atlases(args.output_root, asset_dir=args.asset_dir,
                                     device=args.device)
    save_subregion_root(root)
    print(root)


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
