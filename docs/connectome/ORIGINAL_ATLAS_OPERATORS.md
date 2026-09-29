# 原 UKB 流程 atlas 算子：输入、输出与调用

本页对应 [UKB-connectomics 总流程](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/UKB_connectivity_mapping_pipeline.sh)中的皮层 FreeSurfer annotation 投影、Tian 皮层下 atlas 逆形变和最近邻采样。FreeSurfer `recon-all` 在外部提供分割、表面与 annotation；FNIT 算子只依赖 PyTorch、nibabel 及包内 `TorchApplyWarp`，不会在运行时启动 FSL、FreeSurfer 或 MRtrix。CUDA 默认允许 TF32；几何坐标采用 float64，标签采用 int32，没有使用 float16/bfloat16。完整的同输入结果、计时口径与公开示例图见[验证报告](../../validation/connectome/atlas_original_parity/README.md)。

## `surface_annotation_to_volume`

**功能。** 按[原始 Python 脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/python/map_surface_label_to_volume.py)的规则，把 FreeSurfer `lh/rh.aparc.annot` 映射到 `ribbon.mgz`：对每个标签为 3（左皮层）或 42（右皮层）的体素，以 `vox2ras_tkr` 取得 FreeSurfer surface RAS 坐标，在该半球 pial 与 white 表面顶点的并集中找欧氏距离最近者，再取对应 annotation 索引。空间分桶只缩小候选集合；不能保证最近点时对全部顶点搜索。输出其余体素为零。

| 参数 | 输入类型和含义 |
|---|---|
| `ribbon` | `[X,Y,Z]` 整数 PyTorch 张量，体素标签 3/42；其设备决定计算设备。 |
| `vox2ras_tkr` | `4×4` 体素索引到 FreeSurfer surface RAS 的矩阵；不能用普通 NIfTI scanner RAS affine 替代。 |
| `lh_pial`, `lh_white`, `rh_pial`, `rh_white` | 每个 `[V,3]`，单位 mm，同半球 pial/white 顶点顺序一致。 |
| `lh_labels`, `rh_labels` | 每个 `[V]`，`nibabel.freesurfer.read_annot` 返回的 annotation 索引，与相应顶点顺序相同。 |
| `bin_width_mm` | 空间候选分桶宽度，默认 3 mm；不会改变最近点的数学定义。 |
| `query_batch` | 一次处理的 ribbon 体素数，默认 8192；可按显存缩小。 |

输出是与 `ribbon` 同形状、同设备的 `torch.int32` 张量，背景零；保存 NIfTI 时使用 `ribbon.mgz` 的 **scanner RAS** affine。皮层 atlas 是 T1 原生网格，尚未变换到 DWI 网格。

```python
from fnit.connectome.atlas_surface import surface_annotation_to_volume

# 以下变量均已由 nibabel 读取为上表所述的张量，且位于同一 CUDA 设备。
# ribbon: FreeSurfer ribbon.mgz 的整数体素；vox2ras_tkr: 该 MGZ header 的 tkr 矩阵。
# 四个 surface 张量: 对应半球 pial/white 顶点；两个 labels 张量: annot 顶点索引。
parcels_t1 = surface_annotation_to_volume(
    ribbon=ribbon,
    vox2ras_tkr=vox2ras_tkr,
    lh_pial=lh_pial,
    lh_white=lh_white,
    lh_labels=lh_labels,
    rh_pial=rh_pial,
    rh_white=rh_white,
    rh_labels=rh_labels,
    bin_width_mm=3.0,
    query_batch=8192,
)
# parcels_t1: [X,Y,Z] int32，仍位于 T1/ribbon 网格。
```

原软件命令必须依照原仓库的目录布局执行；五个位置参数依次是主目录、FreeSurfer subjects 目录、受试者标识、影像实例、atlas 名称：

```bash
python scripts/python/map_surface_label_to_volume.py \
  "$MAIN_DIR" "$SUBJECTS_DIR" "$SUBJECT_ID" "$INSTANCE" "aparc"
```

该命令输出 `data/temporary/subjects/${SUBJECT_ID}_${INSTANCE}/atlases/native.aparc.nii.gz`。原脚本使用 SciPy KDTree；此命令仅作独立参考，FNIT 函数不调用它。

## `invert_fnirt_t1_warp`

**功能。** 将 FSL FNIRT 的 T1→MNI **cubic coefficient** 形变反向求解为 native T1 网格上的 MNI→T1 relative pull field。PyTorch 先把系数展开为 float32 前向场，再按 FSL `invwarp` 的四面体逐行走步求逆；越出前向场视野的顶点使用零非线性位移，未定义值由六邻域均值扩边。各扫描行在 GPU 上并行，几何求解用 float64，保存为 float32。该函数仅支持 NIfTI intent 2007 的 FNIRT cubic coefficient 文件；FSL DCT/quadratic coefficient 或其他位移字段不是该函数输入。算法依据 [FSL `invwarp.cc`](https://git.fmrib.ox.ac.uk/fsl/fnirt/-/blob/master/invwarp.cc) 和同仓库 `tetrahedron.h`。FSL 可选的 Jacobian 拓扑约束尚未实现；本次真实 coefficient 的 FSL 默认约束与 `--noconstraint` 输出任一分量最多相差 7.63×10⁻⁶ mm，因此没有影响此次标签。对需要拓扑修正的其他形变不能直接推断同精度。

| 参数 | 输入类型和含义 |
|---|---|
| `forward_coefficients` | FSL `fnirt --cout` 的 T1→MNI coefficient NIfTI，路径或 nibabel 图像，intent 2007。 |
| `native_t1_reference` | 原生 T1 3D NIfTI，路径或 nibabel 图像；决定输出网格、affine、qform/sform。 |
| `device` | PyTorch 设备，默认 `"cuda:0"`；无 GPU 时显式设为 `"cpu"`。 |
| `max_mirror_steps` | 每个 T1 体素最多镜像移动四面体的次数，默认 1000，与 FSL 源码上限相同；超过上限的未定义点经六邻域均值扩边。 |

返回 `nibabel.Nifti1Image`，形状 `[X,Y,Z,3]`、float32、FSL intent 2006，三个末轴分量为 FSL scaled-mm relative displacement。输出供包内 `TorchApplyWarp` 最近邻采样 Tian atlas；它本身不包含 atlas 标签。

```python
from fnit.applywarp import TorchApplyWarp
from fnit.connectome.atlas_tian import invert_fnirt_t1_warp

# forward_coefficients_path: 同一 T1 经 FSL FNIRT 产生的 intent-2007 coefficient 文件。
# native_t1_path: 原生 T1 参考图像；tian_mni_path: MNI 网格的 Tian S1 标签图。
# inverse: T1 网格上的 MNI→T1 relative pull field nibabel 图像。
inverse = invert_fnirt_t1_warp(
    forward_coefficients=forward_coefficients_path,
    native_t1_reference=native_t1_path,
    device="cuda:0",
    max_mirror_steps=1000,
)
# warped.image: T1 网格的 Tian 整数标签 NIfTI；warped.valid_mask 和 warped.qc
# 分别给出有效采样掩膜和变换诊断。
warped = TorchApplyWarp(device="cuda:0")(
    input=tian_mni_path,
    reference=native_t1_path,
    warp=inverse,
    premat=None,
    postmat=None,
    interpolation="nearest",
    warp_convention="relative",
    output_dtype="int32",
)
```

同输入原软件命令如下。`$T1_IMAGE` 是 3D 原生 T1，`$FORWARD_COEF` 是该 T1 的 FNIRT coefficient，`$TIAN_MNI` 是 MNI Tian S1 标签图；`$INVERSE_FIELD` 为反向 4D 形变，`$TIAN_T1` 为 T1 网格标签图。

```bash
invwarp --ref="$T1_IMAGE" --warp="$FORWARD_COEF" --out="$INVERSE_FIELD"
applywarp --ref="$T1_IMAGE" --in="$TIAN_MNI" \
  --warp="$INVERSE_FIELD" --interp=nn --out="$TIAN_T1"
```

若已经有 FSL `invwarp` 输出，可把它直接作为 `TorchApplyWarp` 的 `warp` 参数；同一实际输入上此单独采样算子与 FSL `applywarp --interp=nn` 的 Tian S1 标签逐体素一致。FNIT 现有 `TorchFNIRT` 的 T1→MNI 前向实现，但与 FSL 前向配准尚未达到数值一致，不能把它生成的 coefficient 作为原 UKB 形变的逐值替身；本页同输入验证固定的是 FSL 参考 coefficient。另可用 FNIT PyTorch SynthMorph 自动生成 Tian 标签，其与 FNIRT 的配对差异见[当前真实 T1 报告](../../validation/connectome/ds004666/atlas_synthmorph_20260929.md)。完整原 UKB 提供的 T1 存档不含预生成的 `T1_to_MNI_warp_coef`，本次参考 warp 是在同一 T1 上新生成并固定的，不能标作 UKB 存档结果。

## 范围与安装

两函数在包内 `fnit.connectome.atlas_surface` 与 `fnit.connectome.atlas_tian`；项目主页 `environment.yml` 提供 PyTorch、nibabel 和绘图依赖，wheel 通过 `pyproject.toml` 声明 PyTorch/nibabel。官方软件只用于单独参考 benchmark。当前完成 `aparc` 皮层映射，以及 Tian S1 逆场和最终标签的同输入对照；Tian S2–S4、TorchFNIRT 与 FSL 前向输出的一致性及最终七套 connectome 仍需逐项验证。
