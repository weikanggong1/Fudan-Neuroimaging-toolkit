"""由 fsaverage 注释和已完成的 recon-all 表面生成原生皮层 atlas。"""

from pathlib import Path
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import torch

from .anatomy import combine_cortical_subcortical
from .atlas_surface import resample_annotation_to_native, surface_annotation_to_volume
from .freesurfer_subject import ConnectomeNode, FreeSurferSubject


def _native_labels_to_t1(
    subject: FreeSurferSubject, mapped: tuple[torch.Tensor, torch.Tensor], device: str,
) -> nib.Nifti1Image:
    """将双半球原生顶点标签 `[Vh]` 投影为 T1 int32 NIfTI。

    ``subject`` 提供 ribbon、pial/white 表面，``device`` 是 Torch 设备。
    原版对应 ``map_surface_label_to_volume.py``；真实体素核对见
    ``validation/connectome/ds004666/atlas_native_aparc_20260929.md``。
    """
    surfaces = []
    for hemi in ("lh", "rh"):
        surfaces.append(tuple(torch.as_tensor(nib.freesurfer.read_geometry(
            str(subject.subject_dir / f"surf/{hemi}.{kind}"))[0], device=device)
            for kind in ("pial", "white")))
    ribbon = nib.load(str(subject.subject_dir / "mri/ribbon.mgz"))
    cortical = surface_annotation_to_volume(
        ribbon=torch.as_tensor(ribbon.get_fdata(dtype=np.float32), device=device),
        vox2ras_tkr=torch.as_tensor(ribbon.header.get_vox2ras_tkr(), device=device),
        lh_pial=surfaces[0][0], lh_white=surfaces[0][1], lh_labels=mapped[0],
        rh_pial=surfaces[1][0], rh_white=surfaces[1][1], rh_labels=mapped[1],
    )
    return nib.Nifti1Image(cortical.cpu().numpy(), ribbon.affine)


def fsaverage_annotation_to_t1(
    subject_dir: str | Path,
    fsaverage_dir: str | Path,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    names: tuple[str, ...],
    *,
    device: str = "cuda:0",
) -> tuple[nib.Nifti1Image, tuple[ConnectomeNode, ...]]:
    """把 fsaverage 双半球连续整数标签映射到受试者 T1 ribbon。

    ``left_labels``、``right_labels`` 分别对应 fsaverage sphere.reg
    顶点，均以 0 为背景，非零标签在两半球之间必须互不重叠且形成 1..K。
    ``names`` 为 K 个节点名；``device`` 是球面最近邻与 ribbon 投影的
    PyTorch 设备。返回 T1 网格 int32 NIfTI 与 K 行节点表。
    参考步骤是 FreeSurfer ``mri_surf2surf --sval-annot``，再运行原
    UKB ``map_surface_label_to_volume.py``；FNIT 只读 recon-all 图像与表面。
    """
    subject = FreeSurferSubject(Path(subject_dir))
    fsaverage = Path(fsaverage_dir)
    source_labels = (np.asarray(left_labels, dtype=np.int32),
                     np.asarray(right_labels, dtype=np.int32))
    present = sorted(set(np.unique(source_labels[0]).tolist()) |
                     set(np.unique(source_labels[1]).tolist()))
    left_nonzero = np.unique(source_labels[0])
    right_nonzero = np.unique(source_labels[1])
    if present != list(range(len(names) + 1)) or \
            np.intersect1d(left_nonzero[left_nonzero > 0],
                           right_nonzero[right_nonzero > 0]).size:
        raise ValueError("two hemispheres must define disjoint contiguous labels 1..K")
    mapped = []
    for hemi, labels in zip(("lh", "rh"), source_labels):
        source_sphere, _ = nib.freesurfer.read_geometry(str(fsaverage / f"surf/{hemi}.sphere.reg"))
        native_sphere, _ = nib.freesurfer.read_geometry(str(subject.subject_dir / f"surf/{hemi}.sphere.reg"))
        if len(labels) != len(source_sphere):
            raise ValueError(f"{hemi} source annotation/sphere vertex counts differ")
        mapped.append(resample_annotation_to_native(
            fsaverage_sphere_reg=torch.as_tensor(source_sphere, device=device),
            native_sphere_reg=torch.as_tensor(native_sphere, device=device),
            fsaverage_labels=torch.as_tensor(labels, device=device),
        ))
    left_ids = set(left_nonzero.tolist())
    nodes = tuple(ConnectomeNode(
        index=i, original_label=i, hemisphere="L" if i in left_ids else "R",
        name=name,
    ) for i, name in enumerate(names, 1))
    return _native_labels_to_t1(subject, tuple(mapped), device), nodes


def native_annotation_to_t1(
    subject_dir: str | Path, annotation: str, *, device: str = "cuda:0",
) -> tuple[nib.Nifti1Image, tuple[ConnectomeNode, ...]]:
    """把 recon-all 的原生 aparc 注释按原 UKB 编号投影到 T1 ribbon。

    ``subject_dir`` 是完成的 recon-all 目录；``annotation`` 为
    ``aparc`` 或 ``aparc.a2009s``，内部读取双半球
    ``label/{lh,rh}.{annotation}.annot``。返回 T1 网格 int32 NIfTI
    ``[X,Y,Z]``（背景 0，节点 1..K）和 K 行 ``ConnectomeNode``。
    原版对应 ``convert_native_annot.py`` 后接
    ``map_surface_label_to_volume.py``；缺失的编号按原脚本删去。
    """
    if annotation not in ("aparc", "aparc.a2009s"):
        raise ValueError("annotation must be aparc or aparc.a2009s")
    subject = FreeSurferSubject(Path(subject_dir))
    labels = [nib.freesurfer.read_annot(str(
        subject.subject_dir / f"label/{hemi}.{annotation}.annot"))
        for hemi in ("lh", "rh")]
    # 原 convert_native_annot.py 用左半球出现的编号过滤两个半球的 LUT。
    present = set(np.unique(labels[0][0]).tolist())
    mapped = []
    nodes = []
    for hemisphere, (values, _, names) in zip(("L", "R"), labels):
        output = np.zeros(values.shape, dtype=np.int32)
        for original in range(1, len(names)):
            if original not in present:
                continue
            index = len(nodes) + 1
            output[values == original] = index
            nodes.append(ConnectomeNode(
                index=index, original_label=original, hemisphere=hemisphere,
                name=f"{'left' if hemisphere == 'L' else 'right'}_{names[original].decode('utf-8')}",
            ))
        mapped.append(torch.as_tensor(output, device=device))
    return _native_labels_to_t1(subject, tuple(mapped), device), tuple(nodes)


def schaefer_to_t1(
    subject_dir: str | Path,
    fsaverage_dir: str | Path,
    left_annot: str | Path,
    right_annot: str | Path,
    *,
    device: str = "cuda:0",
) -> tuple[nib.Nifti1Image, tuple[ConnectomeNode, ...]]:
    """从原 UKB 双半球 Schaefer 注释生成受试者 T1 ribbon atlas。

    ``subject_dir`` 是已完成的 recon-all 目录，``fsaverage_dir`` 提供
    双半球 sphere.reg，``left_annot``/``right_annot`` 是 fsaverage 上
    的 Schaefer 注释，``device`` 指定 GPU/CPU。右半球非零标签在左
    半球标签后连续编号。返回 int32 T1 NIfTI 和逐行对应 1..K 的
    ``ConnectomeNode``。原版依次执行 ``convert_schaefer_annot.py``、
    双半球 ``mri_surf2surf --sval-annot`` 和
    ``map_surface_label_to_volume.py``。
    """
    left, _, left_names = nib.freesurfer.read_annot(str(left_annot))
    right, _, right_names = nib.freesurfer.read_annot(str(right_annot))
    n_left = len(left_names) - 1
    if len(right_names) != len(left_names) or n_left < 1:
        raise ValueError("Schaefer left/right annotation tables must have equal positive size")
    right = np.where(right > 0, right + n_left, 0)
    names = tuple(label.decode("utf-8") for label in (*left_names[1:], *right_names[1:]))
    return fsaverage_annotation_to_t1(
        subject_dir=subject_dir, fsaverage_dir=fsaverage_dir,
        left_labels=left, right_labels=right, names=names, device=device,
    )


def glasser_to_t1(
    subject_dir: str | Path,
    fsaverage_dir: str | Path,
    atlas_templates_dir: str | Path,
    *,
    device: str = "cuda:0",
    workbench_command: str | Path = "wb_command",
) -> tuple[nib.Nifti1Image, tuple[ConnectomeNode, ...]]:
    """将原 UKB Glasser CIFTI 映射到 recon-all 受试者 T1 ribbon。

    ``atlas_templates_dir`` 含 Glasser 32k dlabel，邻接 ``surfaces``
    目录含 32k fsLR 与 164k fsaverage 球面；``fsaverage_dir`` 含
    双半球 sphere.reg。``workbench_command`` 是允许使用的 Workbench
    可执行文件。Workbench 做 32k→164k BARYCENTRIC 标签重采样；
    PyTorch 做 164k→原生顶点和 T1 ribbon 投影。返回 int32 T1 NIfTI
    和按 Glasser 原编号 1..360 排列的节点表。原版对应双半球
    ``wb_command -cifti-separate``、``-label-resample``、
    ``mri_surf2surf --sval-annot`` 和 ``map_surface_label_to_volume.py``。
    """
    templates = Path(atlas_templates_dir)
    surfaces = templates.parent / "surfaces"
    dlabel = templates / (
        "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_"
        "Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    )
    labels = []
    names = None
    with tempfile.TemporaryDirectory(prefix="fnit-glasser-") as scratch:
        for hemi, side, cortex in (("lh", "L", "CORTEX_LEFT"),
                                   ("rh", "R", "CORTEX_RIGHT")):
            label_32k = Path(scratch) / f"{hemi}.32k.label.gii"
            label_164k = Path(scratch) / f"{hemi}.164k.label.gii"
            subprocess.run([str(workbench_command), "-cifti-separate", str(dlabel),
                            "COLUMN", "-label", cortex, str(label_32k)],
                           check=True, capture_output=True)
            subprocess.run([
                str(workbench_command), "-label-resample", str(label_32k),
                str(surfaces / f"{side}.sphere.32k_fs_LR.surf.gii"),
                str(surfaces / f"fs_{side}-to-fs_LR_fsaverage.{side}_LR."
                     f"spherical_std.164k_fs_{side}.surf.gii"),
                "BARYCENTRIC", str(label_164k),
            ], check=True, capture_output=True)
            image = nib.load(str(label_164k))
            labels.append(np.asarray(image.darrays[0].data, dtype=np.int32))
            if names is None:
                lut = image.labeltable.get_labels_as_dict()
                names = tuple(lut[i] for i in range(1, len(lut)))
    return fsaverage_annotation_to_t1(
        subject_dir=subject_dir, fsaverage_dir=fsaverage_dir,
        left_labels=labels[0], right_labels=labels[1], names=names,
        device=device,
    )


def combine_cortical_tian(
    cortical_t1: nib.spatialimages.SpatialImage,
    cortical_nodes: tuple[ConnectomeNode, ...],
    tian_t1: nib.spatialimages.SpatialImage,
    tian_names: tuple[str, ...],
) -> tuple[nib.Nifti1Image, tuple[ConnectomeNode, ...]]:
    """合并同一 T1 网格的连续皮层标签与 Tian 标签。

    ``cortical_t1`` 标签为 0..K，``cortical_nodes`` 定义 1..K；
    ``tian_t1`` 标签为 0..S，``tian_names`` 按 1..S 排列。
    皮层非零体素优先；仅在皮层背景处将 Tian 标签偏移 K。
    返回 int32 NIfTI 和 K+S 行节点表。原版等价步骤为
    ``python scripts/python/combine_volumetric_atlases.py ...``。
    """
    if cortical_t1.shape != tian_t1.shape or not np.allclose(
        cortical_t1.affine, tian_t1.affine, atol=1e-5
    ):
        raise ValueError("cortical and Tian atlases must share one T1 voxel grid")
    k = len(cortical_nodes)
    if tuple(node.index for node in cortical_nodes) != tuple(range(1, k + 1)):
        raise ValueError("cortical_nodes must define consecutive labels 1..K")
    cortical = np.asarray(cortical_t1.dataobj)
    tian = np.asarray(tian_t1.dataobj)
    if (not np.all(np.isfinite(cortical)) or not np.all(np.isfinite(tian)) or
            not np.all(cortical == np.round(cortical)) or
            not np.all(tian == np.round(tian)) or
            cortical.min() < 0 or tian.min() < 0 or
            cortical.max() > k or tian.max() > len(tian_names)):
        raise ValueError("atlas labels must be integers inside their node tables")
    combined = combine_cortical_subcortical(
        torch.as_tensor(cortical.astype(np.int32)),
        torch.as_tensor(tian.astype(np.int32)),
        cortical_max_label=k,
    ).numpy()
    nodes = cortical_nodes + tuple(ConnectomeNode(
        index=k + i, original_label=i,
        hemisphere="R" if name.endswith("-rh") else "L" if name.endswith("-lh") else "",
        name=name,
    ) for i, name in enumerate(tian_names, 1))
    return nib.Nifti1Image(combined, cortical_t1.affine), nodes
