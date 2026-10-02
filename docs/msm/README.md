# FNIT MSM：MSMSulc 与 MSMAll

2026-10-02 的 `9f9f63e` 完整 surface 复测中，新版串行与旧版 `954ad19`、新版并行与新版串行的球面和全部 490 帧时序逐值相同。对独立官方单线程完整链，球面角差均值仍为左 **0.221050°**、右 **0.319595°**，CIFTI 时间 r 均值 **0.977911**。执行优化保持 FNIT 已有结果；这次独立完整链仍未逐值匹配官方。见[最新完整复测](../../validation/fmri/surface_gpu_parallel/README.md)；本页 `4f7bd9f2` 的固定输入专项结果保留其原实测范围。

`fnit.msm` 提供独立的 MSMSulc、[MSMAll 多特征配准](msmall.md)和 [VN、DR/WRN 与特征准备](features.md)。本页介绍 MSMSulc：先按 newMSM 的有限差分规则估计刚性初始化，再用 162→642→2,562 个控制点优化脑沟相似度和三角形应变；HOCR 降阶与 FastPD 选择联合位移。输出球面保持原生顶点顺序。运行时不调用官方 MSM 或 FreeSurfer；准备阶段使用 Connectome Workbench。MSMAll 使用独立的 `MSMAllConfig` 与 `run_msmall`，默认 MSMSulc 调用保持原接口。

新增 MSMAll 以一例真实 WRN `C` 特征独立验证：完整一级和三级配置的双侧球面及固定 490 帧投影均与官方逐值相同。H100 冷/热配准分别为 **20.31/20.39 s** 与 **167.72/166.63 s**，对应官方单线程为 **105.02 s** 与 **2,155.63 s**。这是准备好的连接特征到球面及固定 BOLD 的专项对照，结果与上方默认 MSMSulc 完整 surface 的测量分别记录；输入、配置、范围及修复原因见 [MSMAll 功能页](msmall.md)和[匿名汇总](../../validation/msm/msmall.current.public.json)。

MSMAll 集成测试还定位了共享 MSMSulc 入口的 CUDA 启动问题：独立 Python 进程中，第一次分配 CUDA 张量前重置显存统计可能报 `invalid-device`。现在先初始化 CUDA，再进入阶段计时和显存统计；配准计算未变。MSMSulc 与 MSMAll 两种新进程 GPU 调用测试均已通过。

默认使用 HCP/sMRIPrep 的四级配置：`simval=3,2,2,2`，最大迭代数 `50,10,15,15`。官方 newMSM 将历史仿射相似度值 3 转为 Pearson 2；FNIT 保持该行为。几何、相似度和优化标量使用 float64，写出的 GIFTI 顶点为 float32；没有使用 FP16/BF16。刚性坐标与离散成本在 PyTorch 上计算，刚性加权相似度、每轮 Rodrigues 矩阵及 HOCR/FastPD 使用包内独立 C++ 算子；矩阵缓存后，所有位移标签在 GPU 应用。默认优化路径缓存固定几何并合并传输，`execution="reference"` 保留逐块检查供回归对照；二者使用相同的算法和停止条件。

## 输入与调用

先用同一 recon-all 结果和 `fnit-setup-fmri-surface-assets --output-dir /absolute/path/hcp_surface_assets --fmriprep` 准备 HCP 参考文件。需要 `surf/lh|rh.{white,pial,sphere,sphere.reg,sulc,thickness}` 和 `mri/orig/001.mgz`；T2w、FLAIR 不参与此配准。`prepare_fmriprep_surface_inputs` 产生的 `initial_spheres` 分别是左、右 FS→fsLR 初始球面，顶点顺序与原生 mesh 相同。

```python
from fnit import prepare_fmriprep_surface_inputs
from fnit.msm import MSMSulcConfig, prepare_msmsulc_inputs, run_msmsulc

prepared = prepare_fmriprep_surface_inputs(
    subject_dir="/absolute/path/recon-all/sub-0001",  # 已完成 T1w recon-all 的目录
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP fsLR 资源目录
    output_dir="/absolute/path/work/initial",       # FS→fsLR 初始球面及 ROI 工作目录
    wb_command="wb_command",                         # Workbench 命令
    overwrite=False,                                 # 是否替换准备结果
    parallel=True,                                   # 左右半球独立准备
    cpu_threads=8,                                   # 总 CPU 预算；并行时左右各四个
)
inputs = prepare_msmsulc_inputs(
    subject_dir="/absolute/path/recon-all/sub-0001",  # 与上一步相同的 recon-all 目录
    initial_spheres=prepared.initial_spheres,         # (左球面, 右球面)，原生顶点顺序
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP 164k 参考球面和 sulc
    output_dir="/absolute/path/work/msm-inputs",    # 双侧球面、sulc、仿射矩阵工作目录
    wb_command="wb_command",                         # Workbench 命令
    parallel=True,                                   # 双侧转换和仿射准备并行
    cpu_threads=8,                                   # 本次准备的总 CPU 预算
)
spheres = run_msmsulc(
    inputs=inputs,                                    # {"L": MSMSulcInputs, "R": MSMSulcInputs}
    output_dir="/absolute/path/work/msm-output",    # 注册球面与 JSON 报告目录
    device="cuda:0",                                 # PyTorch 计算设备，也可为 cpu
    config=MSMSulcConfig(),                           # 默认 HCP 四级配置，也可填官方配置文件路径
    execution="optimized",                          # 缓存和合并传输；reference 用于执行方式对照
    parallel=True,                                   # 左右独立配准；False 保留串行执行
    cpu_threads=8,                                   # 局部邻域搜索的总 CPU 预算
)
print(spheres["L"])  # L.sphere.MSMSulc.native.surf.gii
print(spheres["R"])  # R.sphere.MSMSulc.native.surf.gii
```

`MSMSulcInputs` 每侧包括 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六个绝对路径，分别保存原生球面、FS→fsLR 旋转球面、原生脑沟图、参考球面、参考脑沟图和初始旋转矩阵。`run_msmsulc` 返回 `{"L": Path, "R": Path}`；每个 `.surf.gii` 含 N×3 顶点坐标及 F×3 三角形索引，可直接用于 Workbench 表面重采样。

`registration_report.json` 记录实际配置、仿射角度、逐级能量、更新数量、停止位置、展开操作、耗时、峰值已分配显存和写出折叠数。球面可传给 `fMRISurface_pipeline(registered_spheres=(spheres["L"], spheres["R"]))` 做固定球面投影对照；使用已注册球面时不再指定 `msm_config`。独立配准输出是工作文件，最终 fMRI 时间序列由 surface 流程写成 BIDS Derivatives。

### 左右并行与资源预算

`prepare_msmsulc_inputs` 和 `run_msmsulc` 默认 `parallel=True`。`cpu_threads=None` 依次读取 `OMP_NUM_THREADS` 与 `torch.get_num_threads()`；显式输入必须是正整数，表示双侧合计预算。预算 8 分为 4/4，预算 1 自动串行；`parallel=False` 用于逐侧执行。该参数限制局部邻域查询和准备阶段的 Workbench 子进程；PyTorch 使用调用方已有的线程池，库不改全局 CPU 线程设置。MSMAll 的独立配准入口使用相同执行参数，算法配置仍由各自配置类控制。

CUDA 并行时，左右模型和数组独立，各用一个 stream；所有 worker 和 stream 结束后按固定 L/R 顺序汇总报告。每侧仍按原顺序求刚性成本、位移标签、HOCR/FastPD 和能量停止条件。WLS、Rodrigues 矩阵与 HOCR/FastPD 保留包内有序 C++ 运算：真实验证已经表明末位舍入会影响球面邻域和后续离散选择；几何、特征和标签成本继续由 PyTorch GPU 计算。

库内部不重置调用方的 CUDA allocator 峰值。`registration_report.json` 的 `execution` 记录总线程预算、实际并行方式、stream 数和 `peak_scope`；双侧 `peak_allocated_gb` 共享调用方上次重置以来的设备峰值，不能相加作为新峰值。测量独立配准时，在调用前初始化设备、统一重置计数，结束后同步并记录整个调用。

| 配准计算 | 当前后端 | 严格结果边界 |
|---|---|---|
| 最近邻与特征插值 | PyTorch GPU；必要时 cKDTree CPU 回退 | GPU 检查相邻 27 格，并用外部格距离下界证明最近邻。近并列或证明范围不足保留原搜索；有序 CSR 保持既有稀疏权重与累加顺序。 |
| 三角形成本、标签应用、球面变形 | PyTorch GPU float64 | 与原顺序一致的标签、应变成本和逐级传递。 |
| WLS、Rodrigues、源码精度回退 | 包内 C++ CPU | 原标量运算次序；缓存旋转矩阵后 GPU 应用。 |
| HOCR/FastPD 与逐顶点展开 | 包内 C++ / CPU | 图模型独立，离散标签与展开的顺序保持。 |
| GIFTI/报告读写、准备阶段仿射与 ROI | nibabel / NumPy、Workbench CPU | 文件格式与准备的参考定义保持；双侧分别执行。 |

CPU 运算保留在影响离散解的严格标量边界和文件处理处。GPU 最近邻与有序稀疏路径的实际收益按完整双侧复测记录，不从单个算子或启动两个 worker 推断整链加速。

本轮还修复了两个共享子函数问题。`run_msmsulc` 和 MSMAll 原来逐侧重置全设备显存峰值，使调用方完整 API 统计失去前序峰值；当前统一保留调用方计数并写明 `peak_scope`。`_label_samples` 在没有非零位移标签时原返回形状 `(0,)`，后续拼接失败；现在空结果为 `(0,3)`，正常配置的标签数量、顺序与数值不变。空标签、CPU/GPU 和双侧执行控制已覆盖这些情况，真实完整复测记录见[本轮验证页](../../validation/fmri/surface_gpu_parallel/README.md)。

离散迭代中的 DATA 和控制网格保留 newMSM 的展开处理。最后从 DATA 变形到原生球面后，按官方行为写出有限坐标，并分别报告求解坐标和实际 float32 GIFTI 的翻折数、最小方向比及输入退化面数。最终原生球面不再额外优化；使用前应检查这份质控报告。

### 配置参数

下面的四元组依次对应刚性初始化和三个离散阶段。`config=None` 与 `MSMSulcConfig()` 相同；路径输入通过 `MSMSulcConfig.from_file` 读取受支持的官方选项。本配置仅用于 MSMSulc；MSMAll 使用独立配置，MCMC 等其他 newMSM 流程不在本配置支持范围。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `simval` | `(3, 2, 2, 2)` | 相似度；1 为 SSD，2 为 Pearson，刚性阶段的 3 按官方行为转为 2。 |
| `iterations` | `(50, 10, 15, 15)` | 每阶段最大迭代数，离散阶段按官方能量条件提前停止。 |
| `control_grid` | `(6, 2, 3, 4)` | icosphere 控制网格级别；离散控制点数为 162、642、2,562。 |
| `sampling_grid` | `(6, 4, 5, 6)` | 位移标签取样网格级别。 |
| `data_grid` | `(6, 4, 5, 6)` | 脑沟特征网格级别；离散数据点数为 2,562、10,242、40,962。 |
| `regularization` | `(0, 10, 7.5, 7.5)` | 应变正则项权重。 |
| `affine_step_size` | `0.01` | 刚性 Euler 更新的初始步长，单位为弧度。 |
| `affine_gradient_spacing` | `0.5` | 刚性有限差分的初始角度间隔，单位为弧度。 |
| `shear_modulus` / `bulk_modulus` | `0.4` / `1.6` | 三角形形状与面积应变权重。 |
| `strain_exponent` / `regularization_exponent` | `2` / `2` | 应变势和正则项的指数。 |

`MSMSulcConfig.ssd_affine()` 只将刚性阶段改为 SSD，即 `(1,2,2,2)`，用于复测采用该配置的参照。比较双方必须使用同一配置。

浮点配置按官方 `Option<float>` 的精度读取，再提升为 double 计算。因此步长 `0.01` 的有效值为 `0.009999999776482582`，shear/bulk 的有效值分别为 `0.4000000059604645` 和 `1.600000023841858`；报告保存这些实际数值。

官方配置的 `--numthreads=N` 是 CPU 执行参数，FNIT 识别该字段并使用所选 PyTorch 设备；旧 `--threads=N` 写法仍可读取。配置文件支持本页的 MSMSulc 选项，不覆盖 newMSM 的其他注册算法。

## 命令行调用

```bash
fnit-msm msmsulc \
  --inputs-json /absolute/path/msmsulc.inputs.json \
  --output-dir /absolute/path/work/msmsulc \
  --cpu-threads 8 \
  --device cuda:0
```

JSON 顶层含 `L`、`R`；每侧填 `MSMSulcInputs` 的六个文件路径，可用绝对路径或相对清单目录的路径。`--config` 与 `--execution` 对应上述 Python 参数；`--cpu-threads` 为总 CPU 预算，`--no-parallel` 对应 `parallel=False`。MSMAll 命令使用 `fnit-msm msmall`，支持相同执行参数；完整输入和例子见[功能页](msmall.md)。

## 原版对照命令

以下命令只在独立基准环境中运行。安装器提供的 HCP 配置 SHA-256 为 `46b250404cb2570b4f645d8e53c30fabde799663d61761d61cf54ff110318203`，未指定线程数。对照时复制配置并在末尾追加 `--numthreads=1`；线程数是配置选项。FNIT 不调用官方命令。

```bash
mkdir -p /absolute/path/reference
cp /absolute/path/hcp_surface_assets/MSMConfig/MSMSulcStrainFinalconf \
  /absolute/path/reference/MSMSulc.singlethread.conf
printf '\n--numthreads=1\n' >> /absolute/path/reference/MSMSulc.singlethread.conf

newmsm --inmesh=/absolute/path/work/msm-inputs/L.sphere_rot.surf.gii \
  --refmesh=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii \
  --indata=/absolute/path/work/msm-inputs/L.sulc.native.shape.gii \
  --refdata=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.refsulc.164k_fs_LR.shape.gii \
  --conf=/absolute/path/reference/MSMSulc.singlethread.conf \
  --out=/absolute/path/reference/L.
```

## 真实数据基准

### 本轮完整 surface 内调用

同一例已有 volume 和 recon-all/graymid、490 帧、TR 0.735 s、STC 关闭，三次均重新准备输入并估计双侧 MSMSulc。表中 MSM 时间**包含准备与配准**，完整 API 另含投影、CIFTI、QC 和最终保存；不包含前序 volume、recon-all、编译、包导入或 CUDA 初始化。

| 同一真实输入 | 旧版 `954ad19` 串行 | 新版 `9f9f63e` 串行 | 新版 `9f9f63e` 并行 |
|---|---:|---:|---:|
| MSM 准备＋双侧配准 | 202.029 s | 116.361 s | **94.773 s** |
| 完整 surface API，扣除捕获 | 436.245 s | 343.056 s | **243.695 s** |
| 物理 H100 | 1 | 1 | 0 |
| 完整 API 峰值 allocated，十进制 GB | 0.344 | 0.644 | 1.049 |

局部 CPU 总预算为 8，并行时左右各 4；CUDA 上限 20 GB、TF32 开启。注册球面、左右 GIFTI 与 CIFTI 的全部数值，及有效科学配置和 21 结构轴/metadata，旧→新版串行、串行→并行均严格相同。卡号和共享负载不同，耗时是各一次完整观测。新版初次在 GPU 1 初始化失败，未进入 API；表中并行为 GPU 0 新目录的成功运行。实际报告、严格 native 编译 SHA、逐侧执行计数及官方差异见[完整验证页](../../validation/fmri/surface_gpu_parallel/README.md)，实测/发布的全部 116 个 runtime 文件另由[源码回溯](../../validation/fmri/surface_gpu_parallel/publication_runtime.public.json)逐项核验。

### 共享 MSMAll 真实回归

共享 MSMAll 另用同一真实 C 特征复测：旧版串行→新版并行，coarse 一级 **23.237→11.430 s**，refine 三级 **181.790→90.796 s**，两侧注册坐标/拓扑/GIFTI metadata 严格相同。四次均为 GPU 0、CPU 总预算 8，新版 allocated 峰值 **0.2194 / 1.4662 GB**；计时包含既有特征读取、配准和写盘，不含特征估计或 BOLD 投影。历史保存的单线程官方球面坐标/拓扑也相同，metadata 不同，且缺历史逐输入 SHA，仅作为保存结果回归。详细边界见[真实 MSMAll 配对](../../validation/fmri/surface_gpu_parallel/msmall_paired.public.json)。

### 固定输入专项历史：`4f7bd9f2`

同一例真实 UKB 的双侧初始球面、sulc、HCP 模板及完整四级配置用于双方。正式精度参照为 `fsl-newmsm 1.0 h442c261_5` 的单线程输出；同输入两次独立运行的左侧球面逐位相同。官方 8 线程耗时单列，其重复球面有差异。FNIT 使用 H100；球面角差按原生对应顶点计算，490 帧时间相关先逐灰质坐标计算 Pearson r，再取均值。

最新球面、fsLR32k 时间序列、冷/热调用及 profile 汇总见 [MSM 验证页](../../validation/msm/README.md)。配准包含读取和球面/报告写盘，不包含 BOLD 投影、Python 导入与 CUDA 上下文初始化。volume 与 surface 的测量范围见 [验证汇总](../../validation/fmri/README.md)，各阶段时间单独记录。

| 对单线程官方参照 | 修复前左 / 右 | 修复后左 / 右 |
|---|---:|---:|
| 保存球面角差中位数 | 0.4351° / 0.4348° | **0° / 0°** |
| 保存球面角差 p95 | 1.1332° / 0.9300° | **0° / 0°** |
| 490 帧逐点时间 r 均值 | 0.920803 / 0.941545 | **1 / 1** |
| 时间序列 MAE | 21.0203 / 20.0491 | **0 / 0** |

本例修复后的 GIFTI 球面与固定 volume 生成的 CIFTI 均逐值一致。FNIT 双侧配准冷/热调用为 **201.99 / 198.08 秒**，官方单线程左右合计 **1,587.70 秒**，8 线程合计 **378.03 秒**。FNIT 峰值已分配显存 **0.344 GB**。这是共享 H100 上的单例测量；官方 8 线程重复结果存在差异，精度表统一使用可重复的单线程参照。

修复前版本未应用传入的四级配置，使用固定 schedule；现按相同官方配置执行逐级 DATA→原生球面→新网格变形、标签顺序和 HOCR/FastPD 联合优化。因此修复前后的时间用于描述版本表现，不作为相同算法的执行优化倍率。

本次在独立 MSM 子函数修复了缓存三角形面积、官方浮点配置精度、刚性 WLS 和 Rodrigues 旋转的运算差异。WLS 的 GPU 除法与指数舍入组合使成本相差 1 ULP；旋转中的融合乘加及三角函数舍入使零位移标签产生约 `6×10⁻¹³` 的坐标偏差。它们在网格顶点和边处改变后续邻域选择。采用 FNIT C++ 的官方运算顺序后，真实调用的全部 77 次刚性成本及对应逐点值逐位一致，前两轮控制点和 DATA 坐标也逐位一致；完整球面精度另按全流程测量。surface pipeline 继续调用修复后的 `fnit.msm`。

固定官方球面时，Workbench 投影与 CIFTI 组装对独立对照逐值一致，见 [固定球面报告](../../validation/fmri/surface_fixed_sphere.public.json)。pipeline 传递注册配置，并将写出球面的质控保留在 BIDS sidecar。

本例最终原生输出翻折数为左侧 **1**、右侧 **0**，与官方相同；报告按实际 float32 保存坐标计算，保留迭代中的展开处理。

## 最近版本与 benchmark 记录

| 实测或更新 | 范围与记录 |
|---|---|
| `9f9f63e` GPU 重采样与双侧并行 | 严格最近邻证明、有序 GPU CSR、独立 stream 与原生 GIL 释放；完整 surface 的球面/490 帧时序保持旧版数值。MSM 准备＋配准串行/并行为 **116.361 / 94.773 s**，完整 API 为 **343.056 / 243.695 s**；两次物理 GPU 不同。修复调用方峰值统计和空标签形状，见[最新完整复测](../../validation/fmri/surface_gpu_parallel/README.md)。 |
| `4f7bd9f2` 独立 MSMSulc | 修复缓存面积、浮点配置、刚性 WLS 和 Rodrigues 顺序；本例保存球面及固定 clean 时序逐值相同，见上表和[专项报告](../../validation/msm/current.public.json)。 |
| `7102c187` 完整 surface 历史 | 重新准备几何、估计球面并投影 preproc；CIFTI 时间 r 均值 0.977911，范围包含完整 surface，见[历史完整报告](../../validation/fmri/surface_e2e/README.md)。 |
| 2026-10 MSMAll 扩展 | 新增独立 MSMAll、VN/DR/WRN 与 C/CA/CAT 特征准备；共享应变成本保留既有 MSMSulc 运算顺序。MSMAll 的真实 C 模式结果单独记录，以上测量保留原源码快照。 |

## 参考文献与原实现

- Robinson 等，*Multimodal surface matching with higher-order smoothness constraints*，NeuroImage，2018，[DOI](https://doi.org/10.1016/j.neuroimage.2017.10.037)。
- Ishikawa，*Transformation of General Binary MRF Minimization to the First-Order Case*，IEEE TPAMI，2011，[DOI](https://doi.org/10.1109/TPAMI.2010.91)。
- Komodakis 等，*Fast, Approximately Optimal Solutions for Single and Dynamic MRFs*，CVPR，2007，[论文](https://www.csd.uoc.gr/~tziritas/papers/CVPR07_FastPD.pdf)。
- 原实现：[newMSM](https://github.com/rbesenczi/newMSM)、[HCP Pipelines](https://github.com/Washington-University/HCPpipelines)、[Connectome Workbench](https://github.com/Washington-University/workbench)。
