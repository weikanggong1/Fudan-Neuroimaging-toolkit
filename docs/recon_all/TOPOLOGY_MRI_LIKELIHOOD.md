# 拓扑缺陷的 MRI 似然子项

## 1. 功能简介

`joint_gray_white_pdf()` 根据有序顶点的灰白强度估计联合 PDF；
`patch_mri_likelihood()` 在当前候选几何的法向两侧各采样 0.5 mm，
计算完整面、顶点 MRI 似然及面积加权诊断。实现使用 NumPy/Numba，
不运行 FreeSurfer 程序，复用已经通过完整同输入验证的
[`topology_edge_mri_score` MRI 采样](TOPOLOGY_MRI_SCORE.md)。

这两个函数承担拓扑优化的 MRI 子项。几何重连、候选拟合、体积
unMRI、法向与曲率子项、遗传搜索和最终质量检查仍由上层负责。
本页真实测试覆盖完整初始统计和首个重连候选的 MRI 子项，尚不代表
全部候选 fitness 或完整 GA 已通过。

## 2. Python 调用、输入与输出

```python
import numpy as np
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.topology_mri_likelihood import (
    joint_gray_white_pdf,
    patch_mri_likelihood,
)

# 初始统计生成全局 PDF；后续重连候选沿用这份自产 PDF。
joint_pdf = joint_gray_white_pdf(
    white_values=white_vertex_values,       # float32(N,)，原 MRI 0…255 强度
    gray_values=gray_vertex_values,         # float32(N,)，原 MRI 0…255 强度
    excluded_vertices=marked_or_rip,        # bool(N,)，排除全部 marked 或 rip 顶点
)
topology_ras_to_voxel = source_topology_ras_to_voxel(
    volume_header=brain_image.header,       # 当前 uint8 MRI 的 nibabel MGH 头
)
likelihood = patch_mri_likelihood(
    original_vertices=original_ras_vertices,  # float32(N,3)，当前候选 surface RAS/mm
    faces=ordered_faces,                      # int32(F,3)，同序面顶点索引
    vertex_normals=current_vertex_normals,    # float32(N,3)，当前无量纲法向
    face_normals=current_face_normals,        # float32(F,3)，当前无量纲面法向
    vertex_areas_mm2=current_vertex_areas,     # float32(N,)，源 metric 面积字段
    face_areas_mm2=current_signed_face_areas,  # float32(F,)，保留源有向面积
    patch_vertices=ordered_patch_vertices,    # int32(K,)，完整有序且唯一的顶点索引
    patch_faces=ordered_patch_faces,          # int32(M,)，完整有序且唯一的面索引
    volume=np.asarray(brain_image.dataobj),   # 3D uint8，读取用 nibabel
    ras_to_voxel=topology_ras_to_voxel,        # float32(4,4)，MRI tkregister RAS→体素
    gray_white_pdf=joint_pdf,                 # float32(256,256,1)，白行/灰列的自产 PDF
)
```

所有参数均须显式提供，没有设备、平滑或阈值开关。PDF 按原顶点顺序
计数，空 bin 赋 0.1，再按白行、灰列顺序以 float32 归一化。
灰白输入是原 MRI 强度，不是组织标签。法向与面积须在当前坐标状态
计算，不能复用之前 white、sphere 或另一轮候选的缓存。

MRI 采样使用当前 MRI 自身的 tkregister RAS 逆矩阵。拓扑内部这条
路径不使用 placement 的 surface-footer→scanner RAS 桥；成熟
placement 函数保持原语义。面中心保留 float32 求和和除法边界，
后续采样与累计按固定源码的 float/double 规则执行。

输出结构：

| 函数/字段 | 类型、结构与含义 |
| --- | --- |
| `joint_gray_white_pdf()` 返回值 | `float32(256,256,1)` 正值联合概率表 |
| `face_ll`、`vertex_ll` | Python `float`，源 float32 MRI 子项 |
| `mri_ll` | Python `float`，上面两个子项的 double 相加值 |
| `area_weighted_face_ll`、`area_weighted_vertex_ll` | Python `float`，源面积加权诊断，不代替主 fitness |
| `area_weighted_face_defined`、`area_weighted_vertex_defined` | `bool`，对应诊断是否有限；不是最终网格验收状态 |
| `area_weighted_face_bits`、`area_weighted_vertex_bits` | `int`，对应源 float32 位模式，保留 NaN/Inf 诊断 |
| `scope` | 字符串，明确仅 MRI 子项，未实现完整 fitness/GA |

函数不修改输入、不移动网格、不写标准输出文件。允许空 patch，对应
子项为 0。非法形状、索引越界或重复、非有限字段、非正 PDF、全部
顶点排除或未排除强度越界抛 `ValueError`。按原顺序 double 累计的选中
有向面积总和为 0 时保留源 IEEE NaN/Inf 诊断，`*_defined=False`，不把
它改成 0，也不拒绝主项有限的中间候选。JSON 调用者将未定义诊断写成
null 并保留 `*_bits`。采样产生非有限主似然时抛 `FloatingPointError`。

源拓扑初始状态允许有向负面面积，必须保留其符号。面积加权似然
可以为正，不能解释成物理脑区面积或用于放行最终网格。最终表面、
球面翻折、自相交等质量门独立执行。

## 3. 命令行调用

这两个内部函数没有独立生产 CLI。同输入回归脚本读取隔离 benchmark
检查点和同源自产 MRI，在新目录生成候选 PDF、标量及数值报告：

```bash
python tools/benchmark_recon_topology_mri_likelihood.py \
  --capture benchmark/reference/statistics.bin \
  --brain benchmark/input/brain.mgz \
  --output benchmark/mri-likelihood-new \
  --threads 4 \
  --capture-role initial-statistics

# 重连候选使用上一步 FNIT 自产全局 PDF，不按新的 marked 状态重建。
python tools/benchmark_recon_topology_mri_likelihood.py \
  --capture benchmark/reference/patch_statistics.bin \
  --brain benchmark/input/brain.mgz \
  --output benchmark/patch-mri-likelihood-new \
  --threads 4 \
  --capture-role retessellated-patch \
  --joint-pdf-input benchmark/mri-likelihood-new/candidate.npz
```

`capture` 是固定源码观察器的 `FNITFIT1` 检查点，仅用于算子诊断；
`brain` 必须为同输入的三维 uint8 MRI；`output` 必须不存在；
`threads` 是成熟线程预算作用域的正整数，默认 4，CPU affinity
由运行者固定。`capture_role` 默认 initial-statistics；候选必须显式指定
retessellated-patch 并提供 `joint_pdf_input`，后者为初始自产 PDF 的 NPZ。
输出包含 `statistics_fields.npz`、`candidate.npz`
和 `report.private.json`。解析器 `load_statistics_capture()` 复制
全部数组，截断、尾部或非法 CSR 报错。

候选计算使用冻结几何、自产 PDF 和 MRI，不接收参考标量；保存候选
后才比较参考 PDF/标量。检查点与 NPZ 不应回灌生产或公开发布。
未定义诊断按原 IEEE 类别及位模式比较，误差 max/P99 写 null，不填0。
严格回归失败会先保存报告再抛异常。

## 4. 原软件调用

对应 `mris_fix_topology` 内部的
`mrisComputeGrayWhiteJointPDF()`、`mrisDefectFaceMRILogLikelihood()`
与 `mrisDefectVertexMRILogLikelihood()`，没有独立原软件命令。
本轮固定源码 `d932c45b7941662ea380a05efef580568b98d41a` 独立
Conda 构建参考观察器，运行以下完整入口直到初始统计和首次边排序
检查点，然后有意停止；这不是完整 GA 运行：

```bash
# 只在隔离 benchmark 中运行；原程序不进入 FNIT 生产调用。
mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix \
  -orig orig.nofix -out diagnostic.orig.premesh \
  -ga -seed 1234 -threads 4 subject lh
```

## 5. 最新真实数据的精度与耗时

公开 `ds000114/sub-07/ses-test` 左半球，输入来自 `79a41cdd` 原始
T1 连续生产链。共 110,534 顶点、221,116 面；完整初始统计选择
108,538 顶点和 216,722 面。权重、上游、源码、参考程序和输入
哈希由[机器可读收据](../../validation/recon_all/optimizations/20261010_topology_source_mri_likelihood/README.md)绑定。

| 比较项 | 当前结果 |
| --- | --- |
| 全部 65,536 个联合 PDF 值 | 不同数 0，max/P99=0 |
| MRI 变换矩阵 | 不同数 0 |
| `face_ll` | −9.599752426147461；参考差 0 |
| `vertex_ll` | −6.82485818862915；参考差 0 |
| 加权面似然 | +6.5949201583862305；参考差 0 |
| 加权顶点似然 | −6.870598316192627；参考差 0 |
| 自有完整子项重复 | 字段完全一致 |
| 两次固定源码参考完整检查点 | SHA-256 相同，初始边/采样/准备检查点保持原值 |

同主机四核预算、CPU NumPy/Numba，未使用 GPU：

| 本次计时范围 | 秒 |
| --- | ---: |
| PDF 首次 JIT 计算 | 0.483241 |
| 完整面/顶点 MRI 似然首次 JIT | 0.987338 |
| 同输入热调用 | 0.117249 |
| 读检查点/MRI、写 NPZ/报告的脚本墙钟 | 3.635384 |
| 子解释器启动、导入和完整脚本 | 9.564424 |
| 参考入口两次至检查点（含准备与全部初始统计） | 7.614243 / 5.089190 |

参考入口包含更多步骤，不能将这两行计算为 MRI 子项速度比。
这些是共享节点阶段实测，不代表整例耗时、十分钟目标或总体指标
等效。图形、分割 Dice 和最终 ROI 指标属于完整拓扑生产链验收，
本子项不生成新几何，因而本页没有新的脑表面图。

### 首个完整候选的接口修复与回归

完整 GA 接入前发现成熟子函数的零面积校验会阻断真实中间候选。
原首次重连候选的 99 个顶点、138 个面面积全为零：两 MRI 主项仍有限，
源面积诊断确实计算 0/0，未参与 fitness。本次使用原顺序 IEEE 除法保留
这些诊断，不改变候选、主似然或最终网格检查。

实际新源 SHA `e2867b8f3faf939cd8d7e057a62c5dd0de71e9c9b7783a4692709681e9b2207a`：
完整初始 PDF、矩阵和四个字段仍无差；候选 `face_ll=-7.894798278808594`、
`vertex_ll=-7.215606212615967` 与两次固定参考相同，两 NaN 诊断均为
`0xffc00000`。11 项契约通过。四核同输入候选 API 首次/热调用为
0.061067/0.060614 秒，含读检查点/MRI并写候选 NPZ 的段时间1.447977秒；
完整初始 MRI 热调用0.115265秒。计时边界、原 SHA、脚本及版本见
[本轮零面积回归](../../validation/recon_all/optimizations/20261010_topology_zero_area_mri/README.md)。
这些是子项时间，未验证完整 GA、整例速度或指标等效。

## 6. 更新与 benchmark 记录

2026-10-10：新增完整联合 PDF、面/顶点 MRI 似然和固定源码统计
观察器；7 项本地合同通过（含负有向面积、MRI 矩阵求和规则）。
真实 v3 首次调用被错误的“每个面面积必须为正”校验拒绝。完整输入
有 86,280 个选中负面面积，所选总有向面积仍非零；修改后 v4 完整
同输入严格通过。v3 失败原收据、源码哈希和日志哈希均保留，未修改
原参考、数据、最终质量门或验收阈值。

v5 进一步把零分母检查放入原顺序 double 累计内核，保留有向顶点面积，
避免 NumPy 平衡归约与源顺序抵消不同导致错误拒绝。新增两项合同后
9/9 通过，完整真实 PDF 与四个标量仍全部差 0；v4 原源码/结果绑定
继续保留，不重标为 v5。最新子项及计时绑定 v5 源 SHA
`c3932110db5cff03b1f9b8539b0a7d24768531450e04a5a6a42ac22dabebd585`。

部署 v1/v2 在准备阶段失败，没有启动有效算法验证；最终报告只绑定
实际测试的 v3/v4/v5 版本。完整 unMRI 距离/体积反馈、动态候选 fitness
及 GA 控制器仍单独开发，不能以本页精确子项替代这些步骤。

## 7. 原实现与参考文献

- [固定源码 `mrisurf_defect.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)：MRI face/vertex 子项约 1276–1419，完整初始统计约 15368–15435，联合 PDF 约 3470–3536。
- [FreeSurfer `mris_fix_topology`](https://surfer.nmr.mgh.harvard.edu/fswiki/mris_fix_topology)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery: constructing geometrically accurate and topologically correct models of the human cerebral cortex. *IEEE Transactions on Medical Imaging*, 2001.
