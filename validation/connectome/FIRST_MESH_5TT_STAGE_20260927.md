# FIRST 网格、五组织图与灰白质界面：真实数据逐阶段核对

## 范围与状态

本阶段使用一份真实 UKB T1 的官方 FreeSurfer `aparc+aseg.mgz` 和 FSL FIRST 生成的 14 个原始 `*_first.vtk`。FNIT 从 VTK 开始以 PyTorch 计算部分体积图（PVE）、五组织图（5TT）和灰白质界面（GMWMI）；运行时不调用 FSL、FreeSurfer 或 MRtrix。参考使用 [UKB 原流程](https://github.com/sina-mansour/UKB-connectomics/blob/ec73ab75c868060e3ecacd19735b7a347ad226a5/scripts/bash/probabilistic_tractography_native_space.sh)固定的 MRtrix3 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`；栅格化对照 [MRtrix 源码](https://github.com/MRtrix3/mrtrix3/blob/eeab681d3e0cb004cf1d1d31579d3892197ef5b6/src/surface/algo/mesh2image.cpp)。完整聚合数值见 [公开基准 JSON](first_mesh_5tt.public.json)。真实受试者影像、网格、逐体素位置和文件哈希保留在私有服务器。

**状态：14 个结构均已生成，PVE 非零体素总数一致，尚未达到逐值完全一致。** 14 张 PVE 合计 51,321 个非零体素，两实现数量相同；其中 52 个体素的差异超过 `1e-6`，最大差异约 `0.001`，等于每个边界体素 1000 个子采样点中的一个。由此传递到五组织图的 92 个元素和 GMWMI 的 228 个体素，最大差异分别约 `0.001`、`0.0005`。因此这里不能宣称整个 UKB dMRI→connectome 流程已经与原软件逐值相同。

FSL `run_first_all` 在 14 个真实 VTK 网格生成后，其末尾边界校正/合并步骤失败，未得到完整官方 `*_firstseg.nii.gz`。MRtrix 此处的 **`-first DIR` 分支只读取该文件的图像几何头**，故参考运行使用同网格 NIfTI 作为几何模板；14 个用于 PVE 的 VTK 均为真实 FIRST 输出。这使本次 `VTK→PVE→5TT` 同输入比较成立，但不能替代对 FIRST 完整分割流程的验证。FreeSurfer `recon-all` 与 FSL FIRST 仍是外部输入阶段。

## 函数、输入与输出

| 函数 | 输入 | 输出 | 对应原软件步骤 |
| --- | --- | --- | --- |
| `read_first_vtk` | `path`：ASCII FIRST `*_first.vtk` 文件 | `vertices`：`float64 [N,3]`，FIRST 毫米坐标；`faces`：`int64 [M,3]`，三角面顶点索引 | MRtrix `meshconvert` 读取 VTK |
| `first_vertices_to_voxel` | `vertices`；`template_affine`：NIfTI 体素到扫描仪 RAS 的 `4×4` 矩阵；`template_shape`：`(X,Y,Z)` | `float64 [N,3]`，目标 NIfTI 体素坐标 | `meshconvert -transform first2real`，随后 `mesh2voxel` 的坐标转换 |
| `mesh_to_pve` | `vertices`：目标体素坐标；`faces`：三角面；`shape`：目标网格 | `float32 [X,Y,Z]`，体素 PVE，边界取值为 `0.001` 的整数倍 | `mesh2voxel -template` 的三角面相交、内部填充与 `10×10×10` 边界采样 |
| `first_vtk_to_pve` | `vtk_path`、`template_affine`、`template_shape`、`device`：PyTorch CPU/GPU 设备 | 目标设备上的 `float32 [X,Y,Z]` PVE；自动处理 NIfTI 与 MRtrix 轴手性 | 上述 VTK→PVE 的便捷入口 |
| `freesurfer_first_five_tissue` | `five_tissue`：FreeSurfer 基础 `float32 [X,Y,Z,5]`；二选一 `first_pve`：按下文固定顺序排列的 `[X,Y,Z,14]`，或 `first_labels`：整数 `[X,Y,Z]` | `float32 [X,Y,Z,5]`，ACT 顺序为皮层灰质、皮层下灰质、白质、脑脊液、病灶 | `5ttgen freesurfer -first DIR` 的 FIRST 融合。整数 `first_labels` 备用分支只有单元测试，尚无本次真实数据基准 |

基础五组织图来自现有 `freesurfer_five_tissue`；GMWMI 来自现有 `gmwmi_from_five_tissue`。14 通道固定顺序为 `L_Accu, R_Accu, L_Caud, R_Caud, L_Pall, R_Pall, L_Puta, R_Puta, L_Thal, R_Thal, L_Amyg, R_Amyg, L_Hipp, R_Hipp`，对应 `-sgm_amyg_hipp`。输出 `5TT` 的末维恰好为 5，GMWMI 为 `float32 [X,Y,Z]`。

## 完整具名参数用法

以下是调用示意；`<...>` 由本地实际 BIDS 衍生文件路径替换。`template` 应与 `aparc+aseg` 及 FIRST 网格的目标体素网格一致。

```python
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.connectome.anatomy import freesurfer_five_tissue, gmwmi_from_five_tissue
from fnit.connectome.first_5tt import freesurfer_first_five_tissue
from fnit.connectome.first_mesh_pve import first_vtk_to_pve

names = ("L_Accu", "R_Accu", "L_Caud", "R_Caud", "L_Pall", "R_Pall",
         "L_Puta", "R_Puta", "L_Thal", "R_Thal", "L_Amyg", "R_Amyg",
         "L_Hipp", "R_Hipp")
template = nib.load("<FreeSurfer/mri/aparc+aseg.mgz>")  # 输入：真实 FreeSurfer 标签图
segmentation = torch.as_tensor(
    np.asanyarray(template.dataobj).astype(np.int32),  # 输入：整数 FreeSurfer 区域标签
    device="cuda:0",                              # 输入：计算设备
)
base = freesurfer_five_tissue(
    segmentation=segmentation,  # 输入：整数 [X,Y,Z]；输出：基础 [X,Y,Z,5]
)
first_maps = [
    first_vtk_to_pve(
        vtk_path=Path("<FIRST目录>") / f"<T1前缀>-{name}_first.vtk",  # 输入：原始 FIRST ASCII VTK
        template_affine=torch.as_tensor(template.affine),             # 输入：输出网格 RAS 仿射 [4,4]
        template_shape=tuple(int(x) for x in template.shape[:3]),     # 输入：输出网格 (X,Y,Z)
        device=torch.device("cuda:0"),                                # 输入：GPU；输出为该设备上的 PVE
    )
    for name in names
]
five = freesurfer_first_five_tissue(
    five_tissue=base,                                  # 输入：基础 FreeSurfer 5TT
    first_pve=torch.stack(first_maps, dim=-1),         # 输入：14 个 PVE 的固定顺序堆叠
    first_labels=None,                                 # 输入：目录模式不使用整数 FIRST 分割
)
gmwmi = gmwmi_from_five_tissue(
    five_tissue=five,  # 输入：FIRST 融合后的 5TT；输出：[X,Y,Z] 界面强度
)
```

## 原软件等价命令

```bash
run_first_all -i <T1.nii.gz> -o <FIRST/T1> -d -s 14
5ttgen freesurfer <FreeSurfer/mri/aparc+aseg.mgz> <5tt.mif> \
  -first <FIRST目录> -nocrop -sgm_amyg_hipp -nthreads 4
5tt2gmwmi <5tt.mif> <gmwmi.mif>
```

`run_first_all` 命令展示 FIRST 上游的原软件操作；本次数据的合并阶段未成功。参考中 `5ttgen` 使用固定 `eeab681` 版本，FNIT 调用的 VTK 与其 `-first` 输入目录的 14 个网格相同。

## 基准复跑参数

`tools/benchmark_connectome_first_mesh_5tt.py` 的每个参数均具名传入：

| 参数 | 含义 |
| --- | --- |
| `--mesh-dir` | 14 个原始 FIRST VTK 所在目录；只读取网格，不调用 FIRST |
| `--mesh-prefix` | VTK 文件名中 `-L_Accu_first.vtk` 等结构名前的 T1 前缀 |
| `--aparc-aseg` | 同一 T1 的 FreeSurfer 整数标签图 |
| `--reference-pve-dir` | 独立 MRtrix 参考生成的 14 张 PVE，用于比较，不参与 FNIT 候选计算 |
| `--reference-5tt` | 独立 MRtrix 参考五组织图，只用于比较 |
| `--reference-gmwmi` | 独立 MRtrix 参考灰白质界面图，只用于比较 |
| `--device` | PyTorch 设备，例如 `cuda:0`；默认启用 TF32 |
| `--output` | 聚合指标 JSON 的写出位置，建议放在私有工作目录 |

```bash
python tools/benchmark_connectome_first_mesh_5tt.py \
  --mesh-dir "<FIRST目录>" --mesh-prefix "<T1前缀>" \
  --aparc-aseg "<aparc+aseg.mgz>" \
  --reference-pve-dir "<MRtrix-PVE目录>" \
  --reference-5tt "<MRtrix-5tt.nii.gz>" \
  --reference-gmwmi "<MRtrix-gmwmi.nii.gz>" \
  --device cuda:0 --output "<私有报告.json>"
```

## 同输入基准

| 结果 | 固定版本 MRtrix 参考 | FNIT 最终版 | 差异 |
| --- | ---: | ---: | ---: |
| 14 网格三角面总数 | 19,652 | 19,652 | 0 |
| 14 张 PVE 非零体素合计 | 51,321 | 51,321 | 0 |
| PVE 绝对误差 `>1e-6` | — | — | 52 体素；最大 `0.00100005`；前景并集平均 `1.02309e-6` |
| 5TT 绝对误差 `>1e-6` | — | — | 92 元素；最大 `0.00100011`；全图平均 `1.10746e-9` |
| GMWMI 绝对误差 `>1e-6` | — | — | 228 体素；最大 `0.00050005`；全图平均 `3.69995e-9` |
| 计算时间 | `5ttgen`：48.08 秒，head CPU、4 线程 | VTK→PVE→5TT→GMWMI：3.906 秒，H100 GPU | 硬件与涵盖步骤不同，不计算直接加速比 |
| 资源峰值 | `5ttgen` 最大 RSS：663,420 kB | PyTorch 已分配 GPU 显存：3.094 GiB | 指标定义不同 |

FNIT 包括基准文件读取和数值比较的总墙钟为 26.664 秒；其 GPU 核心计算时间为 3.906 秒。官方计时只包围 `5ttgen`，不包括后续 `5tt2gmwmi` 与 NIfTI 转换。FNIT CPU 相同核心计算另测为 32.211 秒。所有数值只代表当前真实输入与固定版本，不能扩展为跨受试者一致性结论。

下面的 [公开 ds004666 解剖示例图](../../docs/connectome/figures/ds004666_t1_b0_mask_atlas.png) 仅说明本工具包的 T1/b0/atlas 输出形态；它不是本次受隐私保护的 UKB FIRST 配对图像，也不代表本阶段的逐体素误差分布。
