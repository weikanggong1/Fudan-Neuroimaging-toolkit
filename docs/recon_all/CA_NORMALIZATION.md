# GCA 强度归一化与坐标精度

## 功能简介

`run_ca_normalize` 在完整 EM 注册之后，利用固定 T1 GCA 选择三轮控制点并校正强度，生成 `norm.mgz` 和六帧 `ctrl_pts.mgz`。它使用已有 NumPy/SciPy/Numba 算法、nibabel 读写，不调用 FreeSurfer 程序，不改变其他阶段的 PyTorch GPU、TF32 或低精度策略。

```mermaid
flowchart LR
    N[自产 nu 与 brainmask] --> M[掩膜闭运算与强度峰缩放]
    L[完整 EM 自产 voxel LTA] --> C[prior 到 source 坐标]
    G[固定已校验 GCA] --> C
    M --> P[三轮控制点选择与偏置场]
    C --> P
    P --> O[norm uint8 与 ctrl_pts 六帧 float32]
```

## Python 调用及输入输出

```python
from fnit.recon_all.ca_normalize_python import run_ca_normalize

normalization_report = run_ca_normalize(
    nu_path=subject_mri_directory / "nu.mgz",           # 256³、1 mm conformed uint8 T1
    mask_path=subject_mri_directory / "brainmask.mgz",  # 与 nu 同网格；0和1不参与
    gca_path=verified_assets_directory / "average/RB_all_2020-01-02.gca", # 固定单通道图谱
    lta_path=subject_mri_directory / "transforms/talairach.lta", # source voxel→atlas voxel
    norm_path=subject_mri_directory / "norm.mgz",       # 同网格 uint8 强度输出
    ctrl_path=subject_mri_directory / "ctrl_pts.mgz",   # 同网格六帧 float32 控制点输出
)
```

六个路径参数必传，无隐藏默认值。`nu_path` 和 `mask_path` 必须是同一 conformed 网格，affine 的既有检查为绝对误差 1e-4、相对误差 0；nu 必须为 uint8。GCA 固定 128³ prior、256³ atlas 和当前单通道 T1 profile，不是任意多通道图谱。LTA 是 type=0 的单个 4×4 affine，携带两侧几何；不能把 scanner RAS 或 mm 变换直接当 voxel LTA。输出 norm 保留输入网格/dtype；ctrl_pts 前三帧为各轮控制点标签，后三帧为相应图谱均值。返回字典包含 atlas/image peak、setup、三轮 selection/bias、写出和总耗时，以及各标签控制点数量。缺文件、错误 dtype/geometry/LTA/affine 等明确报错。

内部坐标接口：

```python
import numpy as np
from fnit.recon_all.ca_normalize_python import prior_to_source_coordinates

source_voxel_indices = prior_to_source_coordinates(
    prior=atlas_prior_voxel_coordinates,        # N×3 整数，prior 网格 x/y/z
    voxel_lta=source_to_atlas_voxel_affine,      # 4×4 source→atlas affine
    prior_spacing=2.0,                          # prior 到 1 mm atlas 的缩放
    source_shape=(256, 256, 256),               # 可选；None只做坐标与nint
)  # N×3 int32 源体素索引，不涉及world/RAS或重采样
```

`prior`、`voxel_lta`、`prior_spacing` 必传；`source_shape` 默认为 None，只计算整数坐标。固定 GCA 调用传 256³：原生先在浮点坐标检查闭区间 `[0, size-1]`，有效时才更新整数 source 坐标；越界样本保留 `GCAfindAllSamples` 初始 atlas 坐标，不能裁到边界或使用 NumPy 负索引。逆矩阵复用已有 VNL FP32 affine 实现，spacing 只乘前三列、平移不缩放；每项乘积和累计均舍入 FP32。已产生的 FP32 坐标先提升到 double，再按正负半整数远离零的 nint 规则取整，避免 `.5` 在 FP32 中提前被舍入。非有限、奇异或非 affine 矩阵、错误坐标/shape/索引范围明确失败。此内部接口只声明固定 1 mm GCA profile，不用于任意 atlas 几何。

## 命令行及原软件调用

当前为 pipeline 内部 callable，没有新增生产独立 CLI。真实冻结阶段复现命令与所有参数见 [任务3脚本说明](../../validation/recon_all/accuracy_20261003/task_03/README.md)。原软件对应命令仅在隔离 benchmark 目录执行：

```bash
mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz \
  nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz
```

坐标累计与 nint 是原软件内部步骤，没有单独 CLI。生产完整 EM 入口、图谱许可/大小/SHA 检查沿用项目既有安装与资源清单，本修复无新增依赖。

## 成熟子函数 bug 与当前验证

2026-10-03 定位到旧 `atlas_samples` 用 FP64 inverse/matmul 后才取整，与原生 FP32 inverse/逐项累计不同。在两例冻结完整官方输入上，sub01 原本零差；sub02 多出一个右 WM 控制点，在三轮标签和均值帧形成 6 个不同元素，norm 有 11 个体素差 1。进程内 FP32 坐标诊断使两例 norm/ctrl 全部零差、网格/dtype不变。这是一般坐标数值契约修复，不按某个被试或体素改写输出；生产的平滑、控制点过滤、三轮偏置场、峰值缩放均保留。

| 同输入阶段（旧两例） | 修复前 norm / ctrl 差异 | 坐标诊断 norm / ctrl 差异 | 坐标诊断含I/O秒 |
|---|---:|---:|---:|
| sub01 | 0 / 0 | 0 / 0 | 28.01 |
| sub02 | 11 / 6 | 0 / 0 | 25.55 |

[完整坐标诊断](../../validation/recon_all/accuracy_20261003/task_03/coordinates_fp32_completed.json)绑定输入、源码及输出哈希。生产 helper 还按原生完整链保留越界初始坐标；三项独立契约测试覆盖平移/spacing、负半整数、FP32 `.5` 提升和浮点边界。生产 helper 两例 ABBA（每例 baseline→candidate→candidate→baseline）已完成，四次候选 norm/ctrl 均零差，网格/dtype相同；含加载/计算/写出的均值 sub01 基线26.2825秒、候选25.9953秒（观察-1.09%），sub02 基线25.1052秒、候选26.5025秒（观察+5.57%）。sub02两次候选都慢于两次基线，分步骤差异分布在setup、selection、bias及写出；尚无隔离证据把成本归到坐标算子，也没有预先约定的速度容差，速度稳定未判通过。[完整ABBA报告](../../validation/recon_all/accuracy_20261003/task_03/production_fix_abba_completed.json)及[CSV](../../validation/recon_all/accuracy_20261003/task_03/production_fix_abba_completed.csv)保留八次原始计时；不把诊断耗时替代配对回归。目前没有脑图或新十例整例改善声明，138 项与全链由协调者统一评估，整体等效 `not_assessed`。

完整八组 nu/mask 输入诊断证明：两例有效 mask 支持集都相同；sub01 两个 nu 体素使完整 EM LTA 和控制点改变，norm 扩展为 490692 个不同体素；这是上游传播。sub02 当前自产输入有 norm 41 个不同体素，其中同官方输入的 11 个残差由本次坐标诊断消除，剩余上游传播不能由同输入修复的结果推断已经解决。[八组实测 CSV](../../validation/recon_all/accuracy_20261003/task_03/cross_completed.csv)、[同输入尾段六项零差报告](../../validation/recon_all/accuracy_20261003/task_03/tail_completed.json)与[原始两例审计](../../validation/recon_all/accuracy_20261003/task_03/audit.json)分别记录诊断层级。

## 版本与 benchmark 记录

- 2026-09-27：早期 CA 同输入单例验证见 `validation/recon_all/python_gpu_port/native_cpp_conda_20260927/ca_normalize_same_input.json`；不能代替本轮两例或十例。
- 基线 `816e5610`：sub02 冻结官方输入仍有 norm 11 / ctrl 6 残差；本轮完整八组诊断另存，不改写历史结果。
- 2026-10-03：一般 FP32 prior→source 坐标修复；诊断两例零差。生产 ABBA 已完成同输入精度回归，sub02耗时增加观察仍需处理；源码 SHA 和全部读写计时见任务3当前报告。

## 原实现与参考文献

- [FreeSurfer 固定 GCA 坐标与采样源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/gca.cpp)：`GCAfindAllSamples`、`GCAcomputeSampleCoords`、`GCApriorToSourceVoxel`。
- [固定矩阵源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/matrix.cpp)：`MatrixMultiply` 的 FP32 累计。
- [固定 nint 源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/utils.cpp)；[mri_ca_normalize](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_ca_normalize/mri_ca_normalize.cpp)。
- Fischl et al. Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain. Neuron 33, 341–355 (2002).


## 2026-10-11 共享 GCA 预处理修复与回归

现场同输入独立原版 GCA 检查发现，真实输入的强度直方图过宽时，源实现禁用峰值缩放；共享 `estimate_image_white_matter_peak` 缺少这条分支。已补齐有序 FP32 直方图平滑和稳健 Gaussian 宽度检查，CA normalization 沿用同一个子函数。固定宽度阈值来自源算法，不是本次数据拟定的精度门槛；控制点选择、三轮归一化、偏置场和写出保持原实现。

此前两个旧输入的零差异仍是历史证据，不代表本次新输入已通过。两例完整 GCA 同输入矩阵检查已通过；两例原始 T1→filled 回归按冻结源码运行。固定资源 `native/bin` 缺少 CA 参考程序，随后现场找到现有 Conda 环境内的固定 `d932c45` 完整源码和链接库。在独立新 benchmark 目录保留编译/链接 flags 与库顺序增编参考，构建 EXIT 0，共享 source/build/prefix 的输入哈希前后一致；生产链不读取该程序或其输出。不能把旧原版结果与不同阶段输入直接当成同输入比较。已有公开原版整例参考只用于生产完成后的只读比较，并先核对阶段输入和来源。独立 callable benchmark 可显式传入原版程序，但不会获取它或传参考输出给生产者：

```bash
python tools/benchmark_recon_ca_normalize_python.py \
  --nu subject/mri/nu.mgz --mask subject/mri/brainmask.mgz \
  --atlas resources/average/RB_all_2020-01-02.gca \
  --lta subject/mri/transforms/talairach.lta \
  --binary reference-environment/bin/mri_ca_normalize \
  --output validation/new-ca-normalization-comparison
```

所有参数为必填路径：前四项为同一自产链的 nu、brainmask、GCA 和 type0 LTA；`binary` 为隔离原版参考；`output` 为不存在的新目录。分别输出 Python/原版的 norm（uint8同网格）和三轮 label/mean control frames（六帧float32），报告不同元素、最大/P99差、geometry/dtype和完整文件 API 时间。公开 JSON 只含哈希和 typed 指标；私有日志不进入仓库。该参考未实际执行时必须标记未验证。

重复性检查使用两份已完成的同输入回执，逐数组比较 Python 与原版各自两次输出；输入、候选源码与参考程序 SHA 必须一致：

```bash
python tools/compare_recon_ca_repeats_python.py \
  --first validation/ca-run1 --second validation/ca-run2 \
  --output validation/ca-repeatability.public.json
```

`first`、`second` 是包含 `metrics.public.json`、`python/`、`native-reference/` 的两次隔离 benchmark 目录；`output` 为新的 JSON 路径。影像只读，输出报告数组差异、几何、dtype、有限值状态和文件 SHA，不以 MGZ 压缩文件字节相等替代数组复现。

共享禁用分支的新保护直接返回 atlas 峰值，即 scale=1；即使局部 uint8 区域最大值低于 atlas 峰值，也不再把 atlas 值当局部 bin 索引。合同检查覆盖 GCA/CA 共用函数与恒等缩放，不把该合成合同称为原版 benchmark。两例原有成功回执仍绑定冻结源码；新保护未触发其已通过的索引范围。

2026-10-11 最新本机同输入真实回归：两例各运行两次，四次 Python/原版 norm 与六帧 ctrl_pts 均零差异，shape/affine/dtype相同；Python及原版各自重复数组也全部零差。

| 输入 | Python两次文件API/s | 原版两次含spawn/IO/s | norm/ctrl不同元素 | 两后端重复差异 |
|---|---|---|---|---|
| 公共例A | 34.946 / 32.353 | 50.406 / 48.442 | 0 / 0 | 0 |
| 公共例B | 32.281 / 31.814 | 48.491 / 49.085 | 0 / 0 | 0 |

候选冻结源码24bd8dd1，原版固定d932c45，CPU四核；Python计时排除解释器和顶部导入，原版包括spawn，不作为严格同边界内核加速比。[完整同输入与重复回执](../../benchmark/recon_volume_python_20261011/complete_ca_and_repeats.public.json)和[隔离构建指纹](../../benchmark/recon_volume_python_20261011/ca_reference_build.public.json)保留source/program/compiler/链接库SHA。该证据范围为两例同输入CA，不替代完整整例等效。
