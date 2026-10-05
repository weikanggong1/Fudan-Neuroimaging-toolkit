# MS-HBM CPU 精度与速度验证（2026-10-04）

## 1. 功能简介

`fnit.mshbm` 用固定 HCP_40 prior，将完整 fsLR32k 皮层时序划分为单被试 17 网络。连接 profile 和推断在 CPU 上计算；体积输入的三线性采样与标签回写可使用 PyTorch CPU 或 CUDA。CPU 推断没有随机种子参数。

本轮只将验证后的 float32 profile 提前转为 float64，随后沿用原有推断。外部 profile 仍为 float32；内部方向与 posterior 原本就是 float64。这样避免在每次混合 dtype 矩阵乘法中重新转换完整 profile。所有参数、收敛阈值与迭代上限保持原值。

## 2. Python 调用、输入与输出

详细参数、输出结构与体积空间要求见[功能说明](README.md)。完整真实主样例包含 490 帧、59,412 个皮层顶点、1,483 个 seed；最终标签保留双侧全部 64,984 个顶点。单 run 按前后 245 帧组成两个 pseudo-session。

```python
from fnit.mshbm import load_assets, profiles_from_timeseries, parcellate
from fnit.mshbm.cli import read_cortex

network_assets = load_assets(path=None)  # 使用随包 HCP_40 fsLR32k 先验
cortical_timeseries = read_cortex(
    path="complete_run.dtseries.nii",  # 完整处理后的 CIFTI；仅取双侧 cortex
    mask=network_assets["cortex_mask"],  # 固定 59,412 顶点 mask
)
session_profiles = profiles_from_timeseries(
    series=cortical_timeseries,  # time×59412；保留完整时间轴
    assets=network_assets,  # prior、seed、mask 和邻接图
    censor=None,  # 主速度样例不剔除帧
)
network_labels, convergence_history = parcellate(
    profiles=session_profiles,  # 两个 59412×1483 float32 profile
    assets=network_assets,
    w=200.0,  # 空间 prior 权重
    c=50.0,  # mesh MRF 权重
    max_outer=50,  # 外层上限
    max_em=101,  # EM 上限
    max_m=300,  # M-step 上限
    max_lambda=101,  # posterior 更新上限
)
```

返回 `network_labels` 为 `(64984,) uint8`，0 为背景或无效顶点，1–17 保留 HCP_40 网络身份；`convergence_history` 包含外层与 EM 次数、κ 和目标值。NPY 的四种布局使用同一完整真实时序做入口控制。两个 245 帧文件的多文件控制保留原 run 的全部帧，不能作为两个独立采集 run 的证据。

## 3. 命令行调用

```bash
# CPU surface 推断；环境中固定 BLAS/OpenMP 线程数并由协调器设置亲和性。
fnit-mshbm --timeseries complete_run.dtseries.nii \
  --output-dir network_results --w 200 --c 50

# 完整体积 API 的 CPU 调用。表面坐标与 BOLD 必须位于同一 scanner-RAS 空间。
fnit-mshbm --volume complete_mni_2mm_bold.nii.gz \
  --left-surface left_mni.surf.gii --right-surface right_mni.surf.gii \
  --cortical-mask cortical_mask.nii.gz --output-dir volume_network_results \
  --device cpu --frame-chunk 8 --max-distance-mm 3 --w 200 --c 50
```

`--censor` 为每个输入对应的 0/1 文本向量，1 保留；`--assets` 可指定同规格 NPZ。全部参数和输出文件见[功能说明](README.md)。本轮主计时无 censor；附加 censor 功能验证另列其来源和保留帧数。

## 4. 原软件调用

独立参考使用固定 CBIG commit `b69b822a15e2a94f1e439606552fc44b6858cf3c` 的干净 Git 对象与 MATLAB R2018b。现有磁盘四个函数已修改，因而只复用其合法依赖，参考主函数从原 Git 对象私有导出。导出保留原函数 SHA 与磁盘差异；无关原软件源码不随 FNIT 发布。

```matlab
reference_parameters.project_dir = '/private/reference_result';
reference_parameters.lh_fMRI_list = {'/private/complete_run.dtseries.nii'};
reference_parameters.censor_list = 'NONE';  % 原版无 censor 的标记；保留完整 490 帧
reference_parameters.target_mesh = 'fs_LR_32k';
reference_parameters.group_prior = '/private/CBIG-clean/lib/group_priors/HCP_40/Params_Final.mat';
reference_parameters.w = '200';
reference_parameters.c = '50';
reference_parameters.overwrite_flag = 0;
[left_labels, right_labels] = CBIG_MSHBM_parcellation_single_subject(reference_parameters);
```

CBIG 没有原生 MNI 体积入口。完整体积的投影与标签回写使用独立 SciPy 插值与 KD-tree 检查，不能称为官方 CBIG 体积速度比较。MATLAB 只用于独立 benchmark，FNIT 运行时不调用它。

## 5. 最新精度和时间

### 完整输入下的冻结 FNIT 与优化 FNIT

固定基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。实际导入的 `core.py` SHA 分别为 `8cd2f6fb706a13c60be5070c575cdb77898e2d309cf3f958c4fdbfee355a27ea` 和 `eabb4d62c810fc180d71f41c0e783bcf2dfd9c1bb6ae42fcc699e44859a9aa9e`。输入 CIFTI SHA 为 `ebf2f4ac854e5e521819b5182233442a6b18271c81e62c4609f03cc0cbb4dcdd`。

| 相同物理核预算 | 冻结版完整函数链 | 优化版完整函数链 | 观测加速 | 冻结版推断 | 优化版推断 | 最大 RSS：旧→新 |
|---|---:|---:|---:|---:|---:|---:|
| CPU1 | 85.646 s | 44.162 s | 1.94× | 80.709 s | 38.150 s | 2.617→2.870 GiB |
| CPU8 | 63.490 s | 12.642 s | 5.02× | 59.144 s | 8.314 s | 2.621→2.873 GiB |

两种预算下，64,984 个标签均为 0 个差异；两个完整 profile 的 SHA 相同，六个科学输出文件逐字节相同。外层/EM 记录均为 `(1,5)、(2,2)`。CPU1 全部历史值相同；CPU8 的 κ 最大绝对差为 `1.45974×10⁻¹⁰`、最大相对差为 `1.71817×10⁻¹³`，目标值差为 0。

完整函数链包括加载 prior、读取时序、profile、推断与保存；解释器启动和适配器校验时间另记。每个实现/预算为一次新进程观测，物理核固定为 CPU1 `[2]`、CPU8 `[2,14,18,22,26,30,34,38]`，同一 CPU 锁串行执行。诊断 line profile 不参与速度比较；其主要耗时位于重复的 `profile.T @ posterior` 与 `profile @ direction`。

聚合结果与分步时钟见[完整 CPU 对照](../../validation/fmri_cpu_20261004/task03_mshbm/cpu_hoist_control.public.json)和[实际源码与运行回执](../../validation/fmri_cpu_20261004/task03_mshbm/cpu_hoist_receipts.public.json)。

### 完整体积、参数和 CLI

下表全部使用同一完整 490 帧输入与固定 CPU3 核组。每项旧新完整标签差为 0，科学输出逐字节相同，外层/EM 次数相同；κ 最大相对尾差不超过 `3.08×10⁻¹³`，目标值相对差不超过 `2.18×10⁻¹⁶`。

| 功能控制 | CPU 线程 | 冻结版函数链 | 优化版函数链 | 观测加速 |
|---|---:|---:|---:|---:|
| 完整体积公开 API | 1 | 259.848 s | 200.667 s | 1.29× |
| 完整体积分步调用 | 8 | 109.429 s | 53.057 s | 2.06× |
| 完整体积公开 API | 8 | 116.644 s | 51.447 s | 2.27× |
| 同 run 两个 245 帧文件 | 8 | 62.156 s | 33.813 s | 1.84× |
| `w=100,c=25` | 8 | 90.000 s | 21.452 s | 4.20× |
| CLI 参数入口，完整转置 NPY | 8 | 62.933 s | 13.135 s | 4.79× |

优化版完整体积 CLI 参数入口另实测 77.323 s，使用默认 `frame_chunk=8`、`max_distance_mm=3`。它包含参数解析、全部采样、推断和保存；七个科学输出文件与完整公开 API 逐字节相同。完整 surface CLI、多文件同 run split 与原 surface API 的六个科学文件也逐字节相同。CLI 的 provenance 含不同输出/输入路径，文件本身不要求逐字节一致，科学输出和收敛字段单独核查，见[跨入口控制](../../validation/fmri_cpu_20261004/task03_mshbm/cross_api_controls.public.json)。

四种 NPY 布局均读回 `490×59412`，与完整 CIFTI 的 29,111,880 个值逐值相同。完整体积分步调用的旧新投影时序和两个 profile SHA 也相同。详见[完整功能矩阵](../../validation/fmri_cpu_20261004/task03_mshbm/full_api_matrix.public.json)及[各入口分步回执](../../validation/fmri_cpu_20261004/task03_mshbm/full_api_matrix_receipts.public.json)。

独立 SciPy 检查全部 29,111,880 个采样值，相关性 `0.999999999999454`，原始 BOLD 强度 MAE `0.00240023`、最大绝对差 `0.0634766`；这反映 float32 坐标/插值舍入。独立 KD-tree 回写的完整 MNI 标签体素差为 0、affine 差为 0。完整 surface/volume 网络均值与独立矩阵代数差为 0，17×17 Pearson 矩阵最大绝对差为 `1.33×10⁻¹⁵`。这些是完整数学核查，不能当作 CBIG 原生体积算法的速度比较，详见[独立核查聚合](../../validation/fmri_cpu_20261004/task03_mshbm/independent_oracles.public.json)。既有测试为 7 项通过、1 项 CUDA 小样本单元检查在 CPU 节点跳过；完整 GPU 验收使用下面的真实全部帧结果。

### 相同线程的官方完整链对照

固定 CBIG、冻结 FNIT 与优化 FNIT 在 nodecw10 完成全部六次完整运行，退出码均为 0。CPU1 使用 `[2]`，CPU8 使用同一 socket 的八个独立物理核 `[2,6,10,14,18,22,26,30]`，三种实现使用相同线程和亲和性，按同一锁串行执行。下表每格为一次完整新进程观测。

| 线程 | 实现 | 完整函数链 | 完整进程墙钟 | user / system CPU time | 最大 RSS |
|---|---|---:|---:|---:|---:|
| 1 | 官方 CBIG | 1741.111 s | 1918.470 s | 120.57 / 25.25 s | 7.379 GiB |
| 1 | 冻结 FNIT | 1164.583 s | 1204.470 s | 61.70 / 33.56 s | 2.618 GiB |
| 1 | 优化 FNIT | 603.380 s | 645.810 s | 48.58 / 3.52 s | 2.870 GiB |
| 8 | 官方 CBIG | 1199.323 s | 1361.020 s | 219.56 / 91.71 s | 7.398 GiB |
| 8 | 冻结 FNIT | 773.121 s | 817.330 s | 70.80 / 35.61 s | 2.620 GiB |
| 8 | 优化 FNIT | 181.457 s | 222.580 s | 51.38 / 8.41 s | 2.872 GiB |

官方函数链含原 reader、profile、完整推断和保存；独立导出标签供比较的操作另计。FNIT 函数链含正常标签、网络时序、连接矩阵及 provenance 输出。完整进程另含解释器、导入、校验和报告，均不包含等待 CPU 锁。测量期间节点 load 约 2,415–2,539，因此这些是同线程完整运行的实际观测，不能据此推断稳定加速比，也不能与上面的 nodecw8 时间跨节点相除。

完整 64,984 顶点按原网络身份 1–17 逐值比较，不重排列标签。CPU1 的冻结及优化 FNIT 各有 1 个左半球顶点与 CBIG 不同，背景差为 0：双半球网络 4 和 12 的 Dice 分别为 `0.999817685` 和 `0.999881587`，其余网络为 1。CPU8 的冻结及优化 FNIT 与 CBIG 的全部标签相同，17 个网络 Dice 均为 1。新旧 FNIT 的全部标签在两种线程预算下均相同。

优化 FNIT 的分步骤如下；保存阶段包含全部正常输出。

| 线程 | 加载资产 | 读取时序 | profile | 完整推断 | 保存 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.414 s | 12.332 s | 51.527 s | 530.732 s | 7.558 s |
| 8 | 0.284 s | 13.188 s | 46.938 s | 113.352 s | 6.668 s |

原参考从固定 Git 导出 667 个文件，其中 139 个为 CIFTI/GIFTI/XML 依赖，逐文件大小和 SHA 通过。原 reader 实际读取全部 490 帧，独立比较 29,111,880 个皮层值差异为 0。参考使用与原影像 SHA 完全相同的私有 `complete_run.dtseries.nii` 文件名兼容链接；CBIG 和原 reader 均未修改。早期合法 hostid、startup、`NONE` 和 reader ABI 的失败记录保留。完整聚合与实际源码、进程、精度回执见[官方同线程 CPU 对照](../../validation/fmri_cpu_20261004/task03_mshbm/official_cbig_matched_cpu_v5.public.json)。额外 censor 与 `w=100,c=25` 官方控制均完成；其实际结果和精度差异另列。

### 相同真实输入的可选参数控制

[四项原版／FNIT 完整报告](../../validation/fmri_cpu_20261004/task03_mshbm/optional4_matched_cpu_v1.public.json)沿用 nodecw10 的同八物理核、同先验和完整 490 帧读入。每项原网络身份固定，核对全部 64,984 顶点。

| 可选配置 | profile 保留帧 | 原版函数链 / fresh | FNIT 函数链 / fresh | 固定标签差异 |
| --- | ---: | --- | --- | --- |
| 同 run DVARS P95 censor，w=200/c=50 | 465 | 1232.553 / 1434.240 s | 194.912 / 238.420 s | 0；全部网络 Dice=1 |
| 无 censor，w=100/c=25 | 490 | 1262.607 / 1532.220 s | 215.422 / 258.580 s | 79：左42、右37；背景0 |

`w=100/c=25` 的双半球网络最小 Dice 为 0.995934。该配置尚未达到逐标签一致；已准备冻结旧核心在相同完整输入下的独立一次 API，以区分已有实现差异和本轮优化差异，原版和现有四项结果保持原字节。

DVARS P95 是预先定义的删帧接口控制，不是推荐的运动质控阈值，也不加入无删帧默认主耗时。删帧控制读取完整 490 帧后才构建保留帧 profile。四项观察期间节点 load 约 2451–2526，各配置仅一次完整调用；表中函数链和 fresh process 边界同上方主结果。

### 完整 GPU 回归

用户直接授权后，完整 GPU API 按旧、新、新、旧顺序在同一 H100、同一实际 GPU 锁和固定八物理核运行，四次均返回 0。完整 API 时间为旧 `95.002/94.295 s`、新 `28.105/27.824 s`；完整函数链为旧 `95.688/94.936 s`、新 `28.800/28.468 s`；完整进程为旧 `99.576/98.633 s`、新 `32.619/31.723 s`。全部 490 帧和 64,984 顶点保留，七个科学输出逐值相同；外层/EM 次数相同，κ 相对尾差 `2.94×10⁻¹³` 以内。

四次 TF32 均开启，未使用 float16；Torch allocated 峰值 `183,072,768` 字节、reserved 峰值 `320,864,256` 字节，同期进程树显存峰值 `872,415,232` 字节（0.8125 GiB），均低于 `20,000,000,000` 字节。采样期间平均 GPU 利用率约 99.9%；本结果为繁忙共享 H100 上的完整 API 观测，不能推广为独立 GPU kernel 的速度结论。体积 GPU 函数源码 SHA 旧新相同，观测到的整链收益来自共享 CPU 推断优化。

GPU 首次尝试在创建 CUDA context 前重置峰值计数而退出；第二次适配器先初始化 context 后成功，失败证据保留。[GPU ABBA 聚合](../../validation/fmri_cpu_20261004/task03_mshbm/gpu_abba_control.public.json)和[实际源码回执](../../validation/fmri_cpu_20261004/task03_mshbm/gpu_abba_receipts.public.json)含完整输出、迭代、显存与同期负载证据。

## 6. 版本与 benchmark 记录

- 2026-10-04/05：完成真实完整 CPU 主样例、体积/参数/CLI 矩阵、独立全帧几何和网络核查、GPU ABBA；提前提升 profile dtype；保留参数和完整迭代预算。官方同节点完整六项对照与全部 reader 值核查通过；额外参数控制独立记录。
- 2026-10-01：完整 490 帧官方发布输入与 FNIT 上游输入的下游对照；两边的推断均为 FNIT。已有公开表面、体积与连接矩阵脑图见[功能说明中的历史发布对照](README.md#ukb-官方发布数据对照2026-10-01)。这些脑图提供输出示例，不是新 CPU 计时证据。
- 2026-09-27：旧 100 帧真实样例的 profile/标签 MATLAB 对照，保留原测量源码身份；不代替本轮 490 帧原版重跑。

## 7. 参考文献和原实现

- Kong et al., *Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion*, Cerebral Cortex (2019), [doi:10.1093/cercor/bhy123](https://doi.org/10.1093/cercor/bhy123)。
- [固定 CBIG Kong2019 MS-HBM](https://github.com/ThomasYeoLab/CBIG/tree/b69b822a15e2a94f1e439606552fc44b6858cf3c/stable_projects/brain_parcellation/Kong2019_MSHBM)，[CBIG MIT 许可](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/LICENSE.md)。
- [本轮完整功能矩阵与复现适配器](../../validation/fmri_cpu_20261004/task03_mshbm/README.md)。
