"""旧核团 Python 接口的薄兼容层；计算统一由 segment_subregions 完成。"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .pipeline import segment_subregions
from .recipes.thalamus import _NUCLEI
from .setup import ATLAS_FILES, prepare_subregion_atlases


_STRUCTURES = {"thalamus": "thalamus", "hippo-left": "hippo-amygdala-left",
               "hippo-right": "hippo-amygdala-right"}
_HIPPO_BODY = {"subiculum-body", "CA1-body", "presubiculum-body", "molecular_layer_HP-body",
               "CA3-body", "GC-ML-DG-body", "CA4-body", "fimbria"}
_HIPPO_HEAD = {"subiculum-head", "presubiculum-head", "CA1-head", "parasubiculum",
               "molecular_layer_HP-head", "GC-ML-DG-head", "CA4-head", "CA3-head", "HATA"}
_HIPPO_WHOLE = _HIPPO_BODY | _HIPPO_HEAD | {"Hippocampal_tail"}


def prepare_nuclei_atlas(atlas_root: str | Path, *, asset_dir: str | Path | None = None) -> Path:
    """保留 atlas_root 参数，准备已校验的统一四结构图谱布局。"""
    return prepare_subregion_atlases(output_root=atlas_root, asset_dir=asset_dir)


def _write_volumes(path, volumes):
    with path.open("w", encoding="utf-8") as stream:
        for name, volume in volumes.items():
            stream.write(f"{name} {volume:.6f}\n")


def segment_nuclei(
    norm: str | Path,
    aseg: str | Path,
    wmparc: str | Path,
    atlas_root: str | Path,
    output_dir: str | Path,
    *,
    structures: tuple[str, ...] = ("thalamus", "hippo-left", "hippo-right"),
    threads: int = 4,
    device: str | None = None,
    optimization: str = "fast",
) -> dict[str, dict[str, Path]]:
    """保留旧参数和嵌套路径返回值，使用一次统一 PyTorch 拟合。

    各结构标签写成 NIfTI；右侧兼容标签恢复旧 ID，统一输出保持原 ID。
    volumes 文本使用工作网格后验积分软体积（mm³），不是硬标签计数。
    """
    if not structures or any(name not in _STRUCTURES for name in structures):
        raise ValueError(f"structures 必须从 {tuple(_STRUCTURES)} 中选择，且不能为空")
    if threads < 1:
        raise ValueError("threads 必须大于零")
    root, output = Path(atlas_root), Path(output_dir)
    selected = tuple(_STRUCTURES[name] for name in structures)
    if any(not (root / name / filename).is_file() for name in selected for filename in ATLAS_FILES):
        prepare_nuclei_atlas(root, asset_dir=root if (root / "average").is_dir() else None)
    torch.set_num_threads(threads)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = segment_subregions(
        t1=norm, atlas_root=root, coarse_segmentation=aseg, wmparc=wmparc,
        structures=selected, device=device, optimization=optimization,
        output_dir=output, save_highres=True,
    )
    native = np.asanyarray(result.labels.dataobj)
    outputs = {}
    for name, canonical in zip(structures, selected):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        ids = [identifier for identifier, value in result.label_metadata.items() if value.source == canonical]
        side = name.rsplit("-", 1)[-1]
        offset = 10000 if name == "hippo-right" else 0
        stem = "ThalamicNuclei" if name == "thalamus" else f"{side[0]}h.hippoAmygLabels"
        files = {"labels": directory / f"{stem}.FSvoxelSpace.nii.gz",
                 "high_resolution_labels": directory / f"{stem}.nii.gz"}
        labels = np.where(np.isin(native, ids), native - offset, 0).astype(np.int32)
        nib.save(nib.Nifti1Image(labels, result.labels.affine, result.labels.header), files["labels"])
        fine = result.structure_results[canonical].highres_labels
        fine_data = np.asanyarray(fine.dataobj)
        labels = np.where(np.isin(fine_data, ids), fine_data - offset, 0).astype(np.int32)
        nib.save(nib.Nifti1Image(labels, fine.affine, fine.header), files["high_resolution_labels"])
        volumes = {result.label_table[identifier]: result.volumes[identifier]["soft_volume_mm3"]
                   for identifier in ids}
        if name == "thalamus":
            volumes = {label: volume for label, volume in volumes.items()
                       if label.removeprefix("Left-").removeprefix("Right-") in _NUCLEI}
            for hemisphere in ("Left", "Right"):
                volumes[f"{hemisphere}-Whole_thalamus"] = sum(
                    volume for label, volume in volumes.items() if label.startswith(hemisphere + "-"))
            files["volumes"] = directory / "ThalamicNuclei.volumes.txt"
            _write_volumes(files["volumes"], volumes)
        else:
            prefix = side.capitalize() + "-"
            for parent, key, suffix in (("hippocampus", "hippocampal_volumes", "hippoSfVolumes"),
                                        ("amygdala", "amygdala_volumes", "amygNucVolumes")):
                values = {result.label_table[identifier].removeprefix(prefix): result.volumes[identifier]["soft_volume_mm3"]
                          for identifier in ids if result.label_metadata[identifier].parent == parent}
                if parent == "hippocampus":
                    values = {label: volume for label, volume in values.items()
                              if label in _HIPPO_WHOLE or label == "hippocampal-fissure"}
                    values["Whole_hippocampus"] = sum(volume for label, volume in values.items()
                                                       if label in _HIPPO_WHOLE)
                    values["Whole_hippocampal_body"] = sum(volume for label, volume in values.items()
                                                            if label in _HIPPO_BODY)
                    values["Whole_hippocampal_head"] = sum(volume for label, volume in values.items()
                                                            if label in _HIPPO_HEAD)
                else:
                    values["Whole_amygdala"] = sum(values.values())
                files[key] = directory / f"{side[0]}h.{suffix}.txt"
                _write_volumes(files[key], values)
        outputs[name] = files
    return outputs
