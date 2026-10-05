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

### 官方参考与 GPU 状态

nodecw8 的 MATLAB 正式尝试因合法授权 hostid 不匹配退出，失败记录保留。随后 nodecw10 的实际 MATLAB `disp(version); exit(0)` 返回 0，版本 `9.5.0.944444 (R2018b)`，启动耗时 114.553 s；该证据验证合法可执行性。

官方 CBIG、冻结 FNIT 与优化 FNIT 的完整 CPU1/CPU8 同节点配对使用 nodecw10，同 socket 八个独立物理核为 `[2,6,10,14,18,22,26,30]`。早期 node10 观测时节点负载约 2,500，冻结/优化 FNIT CPU1 链分别为 1147.331/617.421 s；这些历史时间不作为恢复后的配对结果，墙钟和 CPU time 分开报告。前两次官方尝试分别发现上游 startup 的 `clear` 清掉适配器工作区、无 censor 误传为 `{'NONE'}`；适配器已将 startup 放入 base workspace，并按原版语义传标量 `'NONE'`。第三次实际失败位于 CIFTI 读取：现有 default `ft_read_cifti` 的磁盘文件也已修改，字段不兼容原 CBIG 的 `dtseries` ABI。

固定 Git 参考现已导出 667 个文件，其中 139 个为原 CIFTI/GIFTI/XML vendor；每个文件的大小和 SHA 均通过。第四次实际原 reader 探针发现另一项输入约定：原 `ft_read_cifti` 根据文件名的点分段生成数据字段，多段 UKB 文件名不生成 CBIG 要求的 `dtseries` 字段。第五次参考使用指向相同原文件的私有 `complete_run.dtseries.nii` 链接；完整影像字节和 SHA 保持相同，原 reader 与 CBIG 函数均不修改。原 reader 已实际读入完整 490 帧，与独立 CIFTI 读取比较的全部 29,111,880 个皮层值差异为 0。完整原版／冻结／优化 FNIT CPU1、CPU8 六项配对已启动，尚无新的官方分区精度与速度结果。所有失败保留，不能将 nodecw8 的 FNIT 时间除以 nodecw10 的 MATLAB 时间。

用户直接授权后，完整 GPU API 按旧、新、新、旧顺序在同一 H100、同一实际 GPU 锁和固定八物理核运行，四次均返回 0。完整 API 时间为旧 `95.002/94.295 s`、新 `28.105/27.824 s`；完整函数链为旧 `95.688/94.936 s`、新 `28.800/28.468 s`；完整进程为旧 `99.576/98.633 s`、新 `32.619/31.723 s`。全部 490 帧和 64,984 顶点保留，七个科学输出逐值相同；外层/EM 次数相同，κ 相对尾差 `2.94×10⁻¹³` 以内。

四次 TF32 均开启，未使用 float16；Torch allocated 峰值 `183,072,768` 字节、reserved 峰值 `320,864,256` 字节，同期进程树显存峰值 `872,415,232` 字节（0.8125 GiB），均低于 `20,000,000,000` 字节。采样期间平均 GPU 利用率约 99.9%；本结果为繁忙共享 H100 上的完整 API 观测，不能推广为独立 GPU kernel 的速度结论。体积 GPU 函数源码 SHA 旧新相同，观测到的整链收益来自共享 CPU 推断优化。

GPU 首次尝试在创建 CUDA context 前重置峰值计数而退出；第二次适配器先初始化 context 后成功，失败证据保留。[GPU ABBA 聚合](../../validation/fmri_cpu_20261004/task03_mshbm/gpu_abba_control.public.json)和[实际源码回执](../../validation/fmri_cpu_20261004/task03_mshbm/gpu_abba_receipts.public.json)含完整输出、迭代、显存与同期负载证据。

## 6. 版本与 benchmark 记录

- 2026-10-04/05：完成真实完整 CPU 主样例、体积/参数/CLI 矩阵、独立全帧几何和网络核查、GPU ABBA；提前提升 profile dtype；保留参数和完整迭代预算。官方同节点原依赖/完整读入仍在处理，状态以本轮回执为准。
- 2026-10-01：完整 490 帧官方发布输入与 FNIT 上游输入的下游对照；两边的推断均为 FNIT。已有公开表面、体积与连接矩阵脑图见[功能说明中的历史发布对照](README.md#ukb-官方发布数据对照2026-10-01)。这些脑图提供输出示例，不是新 CPU 计时证据。
- 2026-09-27：旧 100 帧真实样例的 profile/标签 MATLAB 对照，保留原测量源码身份；不代替本轮 490 帧原版重跑。

## 7. 参考文献和原实现

- Kong et al., *Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion*, Cerebral Cortex (2019), [doi:10.1093/cercor/bhy123](https://doi.org/10.1093/cercor/bhy123)。
- [固定 CBIG Kong2019 MS-HBM](https://github.com/ThomasYeoLab/CBIG/tree/b69b822a15e2a94f1e439606552fc44b6858cf3c/stable_projects/brain_parcellation/Kong2019_MSHBM)，[CBIG MIT 许可](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/LICENSE.md)。
- [本轮完整功能矩阵与复现适配器](../../validation/fmri_cpu_20261004/task03_mshbm/README.md)。
