# ApplyWarp 显式 World API：真实 CPU 对照

## 1. 功能和范围

`apply_world()` / `run_world()` 按显式 RAS world pull 链，将完整三维影像或四维序列一次插值到参考网格：参考坐标先加 RAS 位移，应用 reference→source world 仿射，再应用每帧的 motion pull 矩阵。该接口与 [普通 FSL scaled-mm 入口](CPU_BENCHMARK_20261004.md) 的坐标、边界和插值协议不同。

匹配原版的条件为 `coordinate_precision="fmriprep"`、`boundary="grid-constant"`，固定 fMRIPrep 25.2.4 的 `resample_image()`。本轮修复成熟 `fnit._world_resampling` 的 CPU 查询：该协议的 nearest/linear 保留原 FP64 坐标，spline 每帧只做一次 SciPy cubic prefilter，再分块查询。几何和 motion 仍使用原 PyTorch 链；帧池不超过调用方的 Torch 线程预算，在坐标准备后启动。主页 Conda 和 Python 包已包含 SciPy，没有新增依赖。

CUDA、默认 `float64`、`periodic` 和普通 FSL 入口保留原采样算子。`float64` 与 `periodic` 使用固定旧 FNIT 同设备结果核对，不对它们计算原版速度比。`fmriprep` 的三维输出另外保留 target 原时间单位；默认 `float64` 三维仍清除时间单位，四维仍保留 source TR/时间单位。

## 2. Python 调用、输入和输出

```python
import nibabel as nib
import numpy as np
import torch
from fnit.applywarp import TorchApplyWarp, WorldTransformChain

torch.set_num_threads(8)  # 调用方设定 CPU 预算；比较双方使用相同预算和亲和性
source_bold_path = "/private/source_bold.nii.gz"  # 完整 (88,88,64,490) 原始 BOLD
reference_mni_image = nib.load("/private/MNI_reference.nii.gz")  # 输出 (91,109,91) 网格
forward_bbr_matrix = np.loadtxt("/private/boldref_to_T1w_world.txt")  # boldref→T1w，RAS mm
pull_ras_image = nib.load("/private/MNI_to_T1w_pull_ras.nii.gz")  # 参考网格上的 (X,Y,Z,3) 位移，RAS mm
frame_motion_pull = np.load("/private/boldref_to_orig_pull.npy", allow_pickle=False)  # (490,4,4)，每帧 world pull

world_chain = WorldTransformChain(
    reference=reference_mni_image,
    reference_to_source_world=np.linalg.inv(forward_bbr_matrix),  # T1w→boldref 的 pull 方向
    pre_affine_pull_ras=pull_ras_image,  # 先作用于 MNI reference world 点
    motion_pull_world=frame_motion_pull,  # 最后映射到每个原始帧
    coordinate_precision="fmriprep",  # 原版 25.2.4 的逐级坐标舍入协议
)
warper = TorchApplyWarp(device="cpu")
warped_bold_image = warper.apply_world(
    source_bold_path,
    world_chain,
    interpolation="spline",  # nearest、linear 或 cubic B-spline
    boundary="grid-constant",  # 零延伸；periodic 为另一种 FNIT 协议
    output_mask=None,  # 可选：同参考网格的真实掩膜；阈值 >0.5 后乘到所有输出帧
    batch_size=8,  # 坐标/采样的帧批大小，不减少帧数
    spatial_chunk_size=262144,  # 样条查询块大小，不裁剪输出网格
)
nib.save(warped_bold_image, "/private/warped_bold.nii")

# 含保存的同一入口；返回绝对输出 Path。
saved_bold_path = warper.run_world(
    source_bold_path, world_chain, "/private/warped_bold_run.nii",
    interpolation="spline", boundary="grid-constant", output_mask=None,
    batch_size=8, spatial_chunk_size=262144,
)
```

| 输入/参数 | 含义 |
|---|---|
| `input` | NIfTI 路径或 image；完整 3D 或 4D，影像有限，输出 float32。 |
| `reference` | `WorldTransformChain` 的三维目标网格。 |
| `reference_to_source_world` | 有限、可逆的 4×4 RAS world pull 仿射。 |
| `pre_affine_pull_ras` | 可选的 `(Xref,Yref,Zref,3)` RAS mm 位移，必须与 reference 同网格；先于仿射。 |
| `motion_pull_world` | 可选的 `(T,4,4)` world pull；帧数须与输入一致。 |
| `coordinate_precision` | `float64`（默认）或 `fmriprep`；后者按固定原版实现逐级舍入及 motion voxel 合成。 |
| `interpolation` | `spline`（World 类入口默认）、`linear`、`nearest`。 |
| `boundary` | `grid-constant`（默认）或 `periodic`；改变数值协议，不自动视作原版等价。 |
| `output_mask` | 可选 reference 网格掩膜，所有帧同一掩膜。 |
| `batch_size` | 默认 8，正整数；完整处理全部帧。 |
| `spatial_chunk_size` | 默认 262144，正整数；完整处理全部目标体素。CPU 原版协议的三种查询均分块，GPU 及其他协议沿用原样条查询块规则。 |

`apply_world()` 返回 NIfTI image；`run_world()` 计算并保存完整 NIfTI，返回 `Path`。输出空间、qform/sform 由 reference 定义；四维输出保留输入 TR 和时间单位。`fmriprep` 三维保留 reference 时间单位标志，不增加时间轴。此入口不返回普通 FSL plan 的静态 `valid_mask`。

## 3. 命令行复现

World 链没有独立 `fnit applywarp --world` CLI。正式单 run volume pipeline 会调用这个 Python 入口；本轮固定链对照使用独立工具：

```bash
python tools/benchmark_applywarp_world_cpu.py run \
  --manifest /private/verified_world_cases.json \
  --candidate-root /path/to/frozen_candidate \
  --baseline-root /path/to/frozen_baseline \
  --output-dir /private/world_cpu_results \
  --cpuset 35,39,43,47,51,55,59,63 --threads 1,8 \
  --single-observation --lock-file /private/applywarp.cpu-timing.lock

python tools/benchmark_applywarp_world_cpu.py export \
  --source /private/world_cpu_results/suite.private.json \
  --output world_cpu_benchmark.public.json
```

manifest 的最小结构如下；路径须由有权限的调用方在服务器现场填写，不将私有 manifest 随报告公开：

```json
{
  "singularity": "/path/to/singularity",
  "fmriprep_image": "/path/to/fmriprep-25.2.4.simg",
  "brain_mask": "/private/MNI_brain_mask.nii.gz",
  "cases": [{
    "id": "world_full490_spline_fmriprep",
    "input": "/private/source_bold.nii.gz",
    "reference": "/private/MNI_reference.nii.gz",
    "pull_ras": "/private/MNI_to_T1w_pull_ras.nii.gz",
    "bbr_forward_world": "/private/boldref_to_T1w_world.txt",
    "motion": "/private/boldref_to_orig_pull.npy",
    "output_mask": null,
    "interpolation": "spline",
    "boundary": "grid-constant",
    "coordinate_precision": "fmriprep",
    "entry": "run_world",
    "batch_size": 8,
    "spatial_chunk_size": 262144,
    "openblas_threads": 1,
    "complete_490": true,
    "official_match": true
  }]
}
```

`brain_mask` 只定义精度报告的脑内/脑外区域，不自动掩膜输出；`output_mask` 才改变输出。`complete_490=true` 检查完整帧数及 motion 形状。`official_match=false` 的扩展配置改用 `--baseline-root` 中的固定旧 FNIT 作为参照。

`official_concurrency` 可选，仅设置原版受支持的 `resample_image(nthreads=...)`；默认等于 CPU 预算。补测在 CPU 预算 8 下设为 1，保持同一八核亲和性和环境预算，但每次只并发一个原版帧任务。候选仍使用原 `batch_size=8`；该配置的实际用核和精度另列，避免混同原版 `nthreads=8` 的失败观察。

`openblas_threads` 可选，将双方 `OPENBLAS_NUM_THREADS` 限制为该值，其他线程预算保持不变。完整八帧并发的正确对照使用 1，避免帧池内再启八个矩阵库线程；仍使用同一总八核亲和性，并记录实际用核。该设置只属于隔离 benchmark，不修改生产 API 的全局线程环境。

公开 export 使用字段白名单，不导出私有原始影像、个体标识、输入路径或输出路径。

## 4. 原版对应

隔离参照端在固定 fMRIPrep 25.2.4 容器内调用：

```python
transforms = nt.TransformChain([
    nt.nonlinear.DenseFieldTransform(pull_ras_image, is_deltas=True),
    nt.Affine(np.linalg.inv(forward_bbr_matrix)),
    nt.linear.LinearTransformsMapping(frame_motion_pull),
])
official_image = resample_image(
    source_image, target_image, transforms,
    fieldmap=None, pe_info=None, jacobian=False,
    nthreads=8, output_dtype="f4", order=3,
    mode="grid-constant", cval=0, prefilter=True,
)
```

nearest、linear、spline 分别对应 `order=0/1/3`。三维 fixture 不添加 motion mapping。mask 变体在原版采样后执行与 FNIT 相同的掩膜乘法；它是声明的完整操作链，不是原版 `resample_image()` 的额外参数。容器只用于 benchmark 参照，生产 FNIT 不依赖 fMRIPrep 或 Nitransforms。

## 5. 真实数据、精度与耗时

主例为完整 `88×88×64×490` BOLD，目标 `91×109×91×490`；使用已有真实 pipeline 生成的 RAS 位移、BBR world 仿射和全部 490 个 motion pull。参数变体使用完整 490 帧原始序列的逐体素均值，产生完整 `88×88×64` 三维 fixture；它不是原 pipeline 保存的运动校正 boldref，也没有从序列挑选少数帧。

1/8 线程在同一 nodecw10、相同 CPU 亲和性上串行运行，共用 ApplyWarp 计时锁。每配置仅一次完整观测，包含加载、完整计算和 `.nii` 保存；入口框架导入/初始化排除，函数内部惰性导入计入，另记完整进程启动/import/container 成本。不开启 warmup，不把单次观测称作稳定速度比。逐值和位模式、全网格/真实 MNI 脑掩膜/掩膜外指标、header 和时间单位分别记录。候选 SciPy 为 1.17.1，固定官方容器为 1.15.2。

### 5.1 完整 490 帧的旧快照观察与优化原因

| CPU 预算 | 官方 API 读写 | all_v15 API 读写 | 官方完整进程 | all_v15 完整进程 | 脑内 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 295.432 s | 675.971 s | 308.419 s | 678.657 s | 1.493e-7 |
| 8 | 71.071 s | 164.907 s | 81.595 s | 166.805 s | 259.188（精度未通过） |

旧路径的 CPU cubic 使用八次三线性 `grid_sample` 与多份 FP64 中间数组，单线程 API 耗时约为原版的 2.3 倍。完整 490 输出均为有限 float32，空间、TR、单位、qform/sform、完整 header 与官方一致；但旧 8 CPU 配对精度未通过。逐帧复核表明，旧候选 1/8 CPU 输出整份文件相同，官方 8 CPU 在 69 帧中偏离官方 1 CPU：脑内 RMSE 259.188，全网格 max 28920.338。候选线程相关 bug 未成立。全网格及脑内/外数字见[旧快照完整 28 配置记录](world_cpu_v15_benchmark_20261004.public.json)和[完整跨线程诊断](world_cpu_thread_diagnosis_20261004.public.json)；失败配对不用于有效速度比。

### 5.2 成熟子函数的精度修复与完整真实 3D gate

旧 CPU nearest 在完整网格出现 3 个不同值，均在真实 MNI 脑掩膜外。逐点重建发现：原 FP64 查询与官方相同（最大差别 1.42e-14），归一化坐标转 float32 后三个查询恰好落到半体素，再由 PyTorch half-to-even 选点。原查询离半体素为 1.24e-7 至 2.69e-6；两种索引分别完整复现保存值。CPU 原版协议已改为直接 FP64 查询，保留其他协议的既有 nearest 规则。linear 同时移除该协议的 float32 normalized grid/采样累计；不声明 CUDA nearest/linear 为官方逐位等价。

三维旧 header 的唯一差别是时间单位：FNIT 为 mm/unknown，官方从 target 保留 mm/sec。问题在成熟 sampler 的 `header.set_xyzt_units(xyz=...)`，不是额外 mask 乘法桥接。新 `fmriprep` 三维分支保留 target 标志，四维和默认 `float64` 合同不变。

最终 CPU 分支以同一完整真实三维 fixture 完成三种插值与 mask ×1/8 CPU：8/8 配置全部 902,629 值逐位一致；全网格、脑内和脑外 max/MAE/RMSE 均为 0，affine、完整 header binary 和 extensions 全部与官方一致。[完整 3D gate](world_cpu_3d_gate_20261004.public.json)保留每配置计时、参数、源文件 SHA-256 和版本。它验证功能和数值，不替代完整 490 帧吞吐。

### 5.3 all_v18 完整正式对照

新冻结源码 all_v18 已完成 15 配置×1/8 CPU，共 30 次完整观察：完整 490 默认 spline、完整 3D 三种插值/mask、九种扩展，以及同一 3D 的 batch1/query262144 对比 batch8/query65536。三维 batch 参数只说明单帧调用的合法覆盖；多帧主例始终完整处理全部 490 帧。[完整正式记录](world_cpu_benchmark_20261004.public.json)绑定两端源码与容器 SHA-256。

| CPU 预算 | 官方 API 读写 | 新 FNIT API 读写 | 官方完整进程 | 新 FNIT 完整进程 | 新 FNIT 精度参照 |
|---:|---:|---:|---:|---:|---|
| 1 | 292.543 s | 243.894 s | 302.940 s | 246.836 s | 官方 1 CPU；脑内 RMSE 1.487e-9 |
| 8 | 73.166 s（官方精度失配） | 57.519 s | 83.289 s（失败观察） | 59.361 s | 使用相同输入的官方 1 CPU；脑内 RMSE 1.487e-9 |

新候选 1/8 CPU 的完整 NIfTI 文件 SHA-256 相同，全部 442,288,210 值逐位相同。对正确官方 1 CPU，全网格只有 113 个值不同，max 0.00048828125、RMSE 2.325e-8；脑内 111,956,670 值只有 10 个不同，max 1.52588e-5、RMSE 1.487e-9。两预算输出的空间、TR、时间单位、qform/sform、完整 header binary 与 extensions 均与正确参照一致。对旧 FNIT，完整主例的 API 观测由 675.971→243.894 s（1 CPU）、164.907→57.519 s（8 CPU）；单线程原版 292.543→243.894 s。这些是单次完整观测。

本病例的固定容器原版 `nthreads=8` 在 50 帧中偏离官方 1 CPU，全网格 RMSE 199.926、脑内 RMSE 213.921；该失败原值完整保留。双方旧/新、1/8 CPU 的所有输入 hash 一致，候选旧/新各自的跨线程像素和位模式均相同。8 CPU 环境预算不能使失配参照成为有效精度与速度配对。

为取得相同预算下的正确参照，额外在八核亲和性、环境线程预算 8 下使用原版受支持的 `nthreads=1`，完整执行全部 490 帧。官方输出整份 NIfTI SHA-256 与原版 1 CPU 相同，候选输出整份 SHA-256 与此前候选 1/8 CPU 相同，输入 hash 也全部相同：

| 八核预算的正确配对配置 | API 读写 | 完整进程 | 进程 CPU 时间 | CPU 时间/墙钟 |
|---|---:|---:|---:|---:|
| 原版，`nthreads=1`（串行帧并发） | 297.844 s | 309.081 s | 765.367 s | 2.476 核 |
| FNIT，原 `batch_size=8` | 63.618 s | 65.766 s | 266.205 s | 4.048 核 |

该补测只改变原版公开的帧并发参数，未修改官方源码；矩阵库仍使用相同环境预算。[独立完整补测记录](world_cpu_official_serial8_20261004.public.json)保留参数、实际用核、完整输出精度及源码版本。它不代表原版正常八帧并发的有效速度。

三维原版协议的三插值、mask 与 batch/query 变体共 10 次配对，全部像素逐位一致、完整 header 一致。九种扩展共 18 次配对，全部像素逐位等于固定旧 FNIT；六种 `float64` 扩展的完整 header 相同，三种 `fmriprep/periodic` 仅按本次修复保留 target 时间单位。最大 RSS：1 CPU 官方/候选为 3,193,440 / 3,677,572 KiB，8 CPU 为 4,203,732 / 3,850,232 KiB。完整进程 CPU 时间/墙钟为官方 0.988 / 6.248、候选 0.990 / 4.373 个等效用核，均未将线程预算写成实际用核。

H100 完整 490 帧的旧主线与 all_v18 输出整份文件 SHA-256 相同，全部像素和 header 一致，峰值 allocation 均为 2,002,300,416 bytes。API 33.351→32.756 s 是共享 GPU 上的各一次观察。[完整 GPU 门控报告](../../validation/multimodal_cpu_20261004/gpu_world_20261004.public.json)记录冻结源码和输出哈希；该证据验证 CUDA 路径未改变。

本轮真实 fMRI 链来自私有影像，只公开聚合指标；不增加该个体的脑图。已核验的公开完整 T1 对照图见 [普通入口 CPU 报告](CPU_BENCHMARK_20261004.md)，不将该图标作 World 本轮结果。

### 5.4 原版并发失配的实际定位

固定容器的 `resample_series_async()` 第 403 行通过任务池并发帧，`worker()` 第 11 行使用 `run_in_executor()`；`resample_vol()` 第 295 行调用 Nibabel `apply_affine()`，其第 97 行为 `pts @ rzs.T`。现场矩阵库为 OpenBLAS 0.3.30、pthreads、SapphireRapids，环境设置 8 线程。

使用原始序列前 32 个完整三维帧、同一真实 dense/BBR 和逐帧 motion，三次各八帧并发原生 `resample_vol()` 诊断中，实际仿射返回坐标分别有 7、4、3 帧失配；最大偏差 77.956 体素，坐标错误帧与采样输出错误帧全部对应。诊断中仅串行化该原生 affine 调用，继续并发八帧的 SciPy 采样，三次全部 32 帧的坐标及输出对正确参照均为零误差。

该证据把问题定位到本环境的 NumPy/OpenBLAS 并发仿射路径；未审计底层矩阵库内部竞争源码，不把内部原因写成已证实的数据竞争。诊断性 affine 锁未进入正式原版计时、未修改容器或生产 FNIT。[完整聚合诊断](world_cpu_official_parallel_diagnosis_20261004.public.json)包含所有逐帧数字、重复次数、源码函数 SHA-256 和行号。

完整 490 帧再以原生 `nthreads=8`、双方 `OPENBLAS_NUM_THREADS=1` 配对。官方保存文件与正确官方 1 CPU 的整份 SHA-256 相同，候选文件与此前候选 1/8 CPU 相同，输入 hash 全部相同；完整网格与 header 精度通过。[公平八帧并发记录](world_cpu_parallel8_blas1_20261004.public.json)是有效的八核预算参照：官方 API 46.472 s、完整进程 56.806 s；all_v18 候选 API 55.070 s、完整进程 57.128 s。API 慢 8.598 s，进程慢 0.322 s（0.6%），未将串行官方结果作为最终速度参照。

随后进行一次有界完整 490 帧 profile：采样 helper 墙钟 33.531 s、输入解码 5.354 s、逐批坐标 4.559 s、输出批写 3.526 s、NIfTI 保存 11.332 s；62 次帧池构造/提交/关闭合计不足 0.4 s。工作线程累计 prefilter 和 query 时间分别为 19.290 / 199.906 s，互相重叠，不能相加为 API 墙钟。[聚合 profile](world_cpu_profile_v18_20261004.public.json)明确标为插桩诊断。依据这个实际热点，最新修订只在 CPU 原版协议将输出数组改为 Fortran 布局，并整次复用帧池；保持原 batch、坐标和查询算式。NumPy 内存布局改变，保存的 NIfTI 像素与 header 契约保持原样。

### 5.5 最新 CPU 输出布局修订：完整 490 帧 1/8 CPU 通过

World 源码 SHA-256 为 `0b24714e94085feb4d40762e9144789d9f8b835aa42d00614a612be8c3046d83`，已收入主任务 all_v20。最新两预算各一次完整配对，双方使用相同亲和性和 OpenBLAS 子线程 1，原版 `nthreads` 等于各自 CPU 预算；候选 batch8/query262144、完整 490 帧和原网格保持不变。CPU1 使用主任务 all_v20 冻结，CPU8 使用 all_v18 执行源加相同 World 修订；四个实际执行源 SHA-256 完全相同：

| CPU 预算 | 原版 API 读写 | FNIT API 读写 | 原版完整进程 | FNIT 完整进程 | 脑内 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 291.460 s | 239.193 s | 301.475 s | 241.778 s | 1.487e-9 |
| 8 | 46.610 s | 45.040 s | 56.297 s | 46.831 s | 1.487e-9 |

| CPU 预算 | 原版/FNIT 最大 RSS | 原版/FNIT CPU 时间÷墙钟 |
|---:|---:|---:|
| 1 | 3,193,488 / 3,621,288 KiB | 0.992 / 0.992 核 |
| 8 | 4,397,944 / 3,818,064 KiB | 5.462 / 5.284 核 |

最新候选 1/8 CPU 的完整 NIfTI 文件 SHA-256 相同，并与 all_v18 候选相同；正确官方 1/8 CPU 文件也相同；全部输入 hash 一致。因此全部 442,288,210 值、完整 header、TR、单位、qform/sform 和 extensions 均保留原候选结果，对官方仍为全网格 RMSE 2.325e-8（113 个值不同）、脑内 RMSE 1.487e-9（10 个值不同）。44 项定向回归通过；主任务 all_v20 共 721 项集成回归通过、2 项本地 FSL 外部测试跳过。[最新两预算完整报告](world_cpu_latest_20261004.public.json)汇总源码、参数、精度和计时，单独[CPU1 all_v20 记录](world_cpu_latest_cpu1_20261004.public.json)与[CPU8 bit gate](world_cpu_pool_layout_20261004.public.json)保留各次冻结来源。

最新 API 在两预算的单次完整观察中均快于正确原版；八核观测由旧布局 55.070→45.040 s，单核由 243.894→239.193 s。每预算各一次完整新配对，不估计稳定加速比，不将旧源码时间写成此次重测。CUDA、`float64` 和 `periodic` 继续采用原分配和算式。

主任务还以 all_v20 在 H100 完整重跑全部 490 帧：与旧主线的整份保存文件逐字节相同，442,288,210 值、所有 header 字段与 extensions 均相同，peak allocation 均为 2,002,300,416 bytes。[最新 GPU20 完整门控](../../validation/multimodal_cpu_20261004/gpu_world_v20_20261004.public.json)绑定本次 `0b24714e` 源码。API 34.103→36.943 s 为共享 GPU 上各一次观察，运行前利用率 100% / 97%；不据此推断稳定 GPU 速度变化。

## 6. 更新记录

| 版本 | 范围 |
|---|---|
| all_v20 / CPU 布局修订 `0b24714e` / 2026-10-04 | canonical CPU 输出按帧连续存储、整次复用帧池；完整490两预算均通过整份候选NIfTI bit gate。原版→FNIT：1CPU API291.460→239.193s，8CPU API46.610→45.040s，各一次观察；721项集成回归通过、2外部测试跳过。 |
| all_v18 / 2026-10-04 | CPU 原版协议 FP64 nearest/linear 查询、每帧一次 cubic prefilter 与受控帧池；fmriprep 3D target 时间标志修复。30 次完整正式观察完成，候选490帧跨线程文件相同；44 项针对性回归和主任务711项集成回归通过；H100完整490输出文件与旧主线相同。官方8CPU失配另作诊断。 |
| all_v15 / 2026-10-04 | 完整490主例与真实3D参数覆盖28次观察；CPU cubic耗时约为官方2.3倍，nearest三处脑外半体素量化差别、3D时间单位差别已定位。 |
| 2026-10-02 | 公共 WorldTransformChain 接入 FNIRT volume；完整真实 pipeline 的历史结果见功能主页。 |

## 7. 参考和原实现

- [fMRIPrep 25.2.4 resampling.py](https://github.com/nipreps/fmriprep/blob/25.2.4/fmriprep/interfaces/resampling.py)。本轮直接检查固定容器实际函数并绑定容器 SHA-256。
- [Nitransforms](https://github.com/nipy/nitransforms)，原版 RAS dense/affine/motion 组合协议。
- [Nibabel 5.3.2 affines.py](https://github.com/nipy/nibabel/blob/5.3.2/nibabel/affines.py)，固定容器实际仿射入口；行号与函数 SHA-256 在诊断记录中绑定。
- Esteban et al. *fMRIPrep: a robust preprocessing pipeline for functional MRI*. Nature Methods 16, 111–116 (2019)。
- [FNIT World sampler](../../src/fnit/_world_resampling.py) 和 [独立 CPU 对照工具](../../tools/benchmark_applywarp_world_cpu.py)。
