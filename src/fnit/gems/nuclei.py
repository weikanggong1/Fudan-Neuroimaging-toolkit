"""同一受试者的丘脑核团、海马亚区和杏仁核分割。"""

from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from ..recon_all.assets import configured_dir, download_asset
from ..weights import cache_dir


_ATLASES = {"thalamus": "ThalamicNuclei", "hippo-left": "HippoSF", "hippo-right": "HippoSF"}
_ATLAS_FILES = ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt")


def prepare_nuclei_atlas(atlas_root: str | Path, *, asset_dir: str | Path | None = None) -> Path:
    """下载并校验约 30 MB 的丘脑及海马/杏仁核图谱；返回 atlas_root。"""
    root = Path(atlas_root)
    cache = Path(asset_dir or configured_dir() or cache_dir() / "recon_all_assets")
    lookup = download_asset("FreeSurferColorLUT.txt", cache)
    root.mkdir(parents=True, exist_ok=True)
    if lookup.resolve() != (root / lookup.name).resolve():
        shutil.copy2(lookup, root / lookup.name)
    for name in ("ThalamicNuclei", "HippoSF"):
        target = root / "average" / name / "atlas"
        target.mkdir(parents=True, exist_ok=True)
        for filename in _ATLAS_FILES:
            source = download_asset(f"average/{name}/atlas/{filename}", cache)
            if source.resolve() != (target / filename).resolve():
                shutil.copy2(source, target / filename)
    return root


def segment_nuclei(
    norm: str | Path,
    aseg: str | Path,
    wmparc: str | Path,
    atlas_root: str | Path,
    output_dir: str | Path,
    *,
    structures: tuple[str, ...] = ("thalamus", "hippo-left", "hippo-right"),
    threads: int = 4,
) -> dict[str, dict[str, Path]]:
    """以相同网格的 norm/aseg/wmparc 运行三个横断面图谱，返回输出文件路径。"""
    if not structures or any(name not in _ATLASES for name in structures):
        raise ValueError(f"structures 必须从 {tuple(_ATLASES)} 中选择，且不能为空")
    if threads < 1:
        raise ValueError("threads 必须大于零")
    images = [nib.load(str(path)) for path in (norm, aseg, wmparc)]
    if any(image.ndim != 3 for image in images):
        raise ValueError("norm、aseg、wmparc 必须均为三维图像")
    if any(image.shape != images[0].shape or not np.allclose(image.affine, images[0].affine, atol=1e-5, rtol=0)
           for image in images[1:]):
        raise ValueError("norm、aseg、wmparc 必须在相同体素网格上")
    root, output = Path(atlas_root), Path(output_dir)
    if not (root / "FreeSurferColorLUT.txt").is_file():
        raise FileNotFoundError(root / "FreeSurferColorLUT.txt")
    for name in set(_ATLASES[item] for item in structures):
        for filename in _ATLAS_FILES:
            path = root / "average" / name / "atlas" / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    from .native_samseg.subregions import run_cross_sectional, set_thread_count

    set_thread_count(threads)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in structures:
        directory = output / name
        parameters = dict(
            atlas_root=str(root), outDir=str(directory),
            inputImageFileNames=[str(norm)], inputSegFileName=str(aseg),
        )
        if name == "thalamus":
            run_cross_sectional("thalamus", parameters)
            stem = directory / "ThalamicNuclei"
            results[name] = {"labels": Path(f"{stem}.FSvoxelSpace.mgz"),
                             "high_resolution_labels": Path(f"{stem}.mgz"),
                             "volumes": Path(f"{stem}.volumes.txt")}
        else:
            side = name.split("-")[1]
            parameters.update(side=side, wmParcFileName=str(wmparc))
            run_cross_sectional("hippo-amygdala", parameters)
            stem = directory / f"{side[0]}h.hippoAmygLabels"
            results[name] = {"labels": Path(f"{stem}.FSvoxelSpace.mgz"),
                             "high_resolution_labels": Path(f"{stem}.mgz"),
                             "hippocampal_volumes": directory / f"{side[0]}h.hippoSfVolumes.txt",
                             "amygdala_volumes": directory / f"{side[0]}h.amygNucVolumes.txt"}
        for path in results[name].values():
            if not path.is_file():
                raise FileNotFoundError(path)
    return results
