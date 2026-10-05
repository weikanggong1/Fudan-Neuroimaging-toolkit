<!-- 原正文完整归档；基线 140c3739ac6c7a6826bf9421202ef59bec7ffe67；只修复迁移后的相对链接。 -->

# MNI152、fsaverage 与 fsLR 皮层图转换

`convert_space` 转换标准空间中的皮层标量图或整数标签图。MNI152 使用 NIfTI，fsaverage 和 fsLR 使用左右半球各一份 GIFTI。支持双半球单帧图和多帧图。

体积到 fsaverage 使用 Wu 等发布的 RF-ANTs 映射；fsaverage 与 fsLR 之间使用 HCP 2017 年修订的球面对应关系，并按两侧平均顶点面积运行 Workbench `ADAP_BARY_AREA`。fsaverage 回到体积时使用 CBIG 发布的 RF-ANTs 最近顶点映射与皮层掩膜。这个路径与官方映射的体素位置一致。体积转表面会压缩皮层深度，表面回体积只填充皮层掩膜；这些步骤不是数学上的可逆变换。

## 1. 安装和模板文件

| 输入空间 / 输出空间 | MNI152 | fsaverage | fsLR |
|---|---|---|---|
| MNI152 | 此函数不提供体积间重采样 | 3k/10k/41k/164k | 32k/59k/164k |
| fsaverage | 默认或指定参考网格 | 换密度；同密度直接复制 | 32k/59k/164k |
| fsLR | 默认或指定参考网格 | 3k/10k/41k/164k | 换密度；同密度直接复制 |


主页的 `environment.yml` 安装 PyTorch、Nibabel、SciPy 和 Connectome Workbench。模板文件按官方版本下载，逐个核验 SHA-256。程序运行时不调用 FSL、FreeSurfer 或 MATLAB。

```bash
conda env create -f environment.yml
conda activate fnit
fnit-setup-space-assets --output-dir /data/fnit_space_assets
```

资产目录包含：

```text
/data/fnit_space_assets/
├── rf_ants/                 # CBIG 1490 人 RF-ANTs 正反向映射及 MNI 皮层掩膜
└── hcp_2017/                # HCP 2017 年 fsaverage/fsLR 球面和平均顶点面积
    └── resample_fsaverage/
```

MNI152 输入必须已经在 FSL MNI152 模板坐标中；函数不会从个体 T1 或 BOLD 自动估计配准。它通过 NIfTI 仿射矩阵读取世界坐标，因而可接受同一模板坐标下的 0.5、1、2 mm 或其他体素网格。不同的 MNI 模板变体不能只靠修改像素大小互换。

## 2. Python 调用、输入和输出

### 输入和输出结构

- **MNI152 输入**：一个已经配准到 FSL MNI152 坐标的 NIfTI，形状为 `(X,Y,Z)` 或 `(X,Y,Z,T)`；使用文件仿射将映射坐标转换到该输入网格。
- **表面输入**：按 `(左, 右)` 排列的 GIFTI。每帧为一个一维数据数组，两侧帧数一致；连续值为 metric/shape，整数标签设 `label=True`。
- **支持密度**：fsaverage 的 `3k/10k/41k/164k` 分别为每侧 2,562/10,242/40,962/163,842 个顶点；fsLR 的 `32k/59k/164k` 分别为 32,492/59,292/163,842 个顶点。
- **表面输出**：返回两侧文件路径；每帧一个 GIFTI 数据数组，保留输入帧顺序。
- **体积输出**：返回一个 NIfTI 路径，形状为 `(*reference.shape[:3],T)`；单帧去掉最后一维。连续图存为 float32，标签图存为 int32，掩膜外为零，仿射和网格与参考图一致。
- **参考网格**：未指定时使用 CBIG 256³、1 mm 网格；指定时可接入下游所需的 2 mm、0.5 mm 等网格。更细的输出网格不会增加原映射的解剖细节；半体素边界的赋值规则见第 5 节。


```python
from fnit import convert_space

surface_files = convert_space(
    source="/data/atlas_MNI152_1mm.nii.gz",  # 输入：已经配准到 FSL MNI152 的 3D/4D NIfTI
    source_space="MNI152",                     # 输入空间；体积图固定写 MNI152
    target_space="fsLR",                       # 输出空间：fsaverage 或 fsLR
    output_dir="/data/out/fsLR32k",            # 输出目录
    assets_dir="/data/fnit_space_assets",      # 已通过 fnit-setup-space-assets 安装的资产目录
    source_density=None,                       # 体积输入没有表面顶点密度
    target_density="32k",                     # fsLR 输出密度：32k、59k 或 164k
    reference=None,                            # 仅目标为 MNI152 时使用；此处不使用
    device="cuda:0",                           # PyTorch 体积采样设备；也可设为 cpu
    label=False,                                # 连续值图；整数标签图设为 True
    wb_command="wb_command",                   # Conda 环境中的 Connectome Workbench 命令
)
# surface_files == (左侧 GIFTI 路径, 右侧 GIFTI 路径)

volume_file = convert_space(
    source=surface_files,                      # 输入：(左侧 GIFTI, 右侧 GIFTI)，顺序固定
    source_space="fsLR",                      # 输入表面空间
    target_space="MNI152",                    # 输出为 MNI 皮层 NIfTI
    output_dir="/data/out/mni05",             # 输出目录
    assets_dir="/data/fnit_space_assets",     # 同上
    source_density="32k",                     # 输入 fsLR 顶点密度
    target_density=None,                       # MNI 输出用 reference 决定网格
    reference="/data/MNI152_ref_0p5mm.nii.gz", # 输出网格及仿射；可为 0.5/1/2 mm 等
    device="cuda:0",                           # CUDA 最近顶点查表；cpu 时使用等价融合内核
    label=False,                               # 连续值；标签图设为 True
    wb_command="wb_command",                   # fsLR 到 fsaverage 的面积校正重采样
)
# volume_file == /data/out/mni05/space-MNI152_cortex.nii.gz
```

不传 `reference` 时，MNI 输出使用 CBIG 附带的 1 mm、256×256×256 皮层掩膜网格。函数返回的是路径，不是影像数组。

| 参数 | 含义 |
|---|---|
| `source` | MNI152 时为一个 NIfTI 路径；表面空间时为 `(左, 右)` GIFTI 路径。每侧顶点数必须与 `source_density` 一致。 |
| `source_space`、`target_space` | `MNI152`、`fsaverage`、`fsLR` 三选一。 |
| `source_density`、`target_density` | fsaverage：`3k`、`10k`、`41k`、`164k`；fsLR：`32k`、`59k`、`164k`。MNI152 一端设为 `None`。 |
| `output_dir` | 输出位置。表面图为 `L/R.<空间>.<密度>.func.gii`，标签为 `label.gii`；体积为 `space-MNI152_cortex.nii.gz`。 |
| `assets_dir` | 下载并校验过的 HCP/CBIG 文件目录。 |
| `reference` | 目标为 MNI152 时指定参考 NIfTI，只取前三维的网格与仿射；未指定时使用 CBIG 1 mm 掩膜。0.5 mm 等自定义网格沿用 PyTorch float32 最近偶数舍入，与 Workbench `floor(voxel+0.5)` 不完全一致，详见第 5 节。 |
| `device` | `cpu` 或可用的 `cuda:<编号>`。正向体积采样和 CUDA 回体积由 PyTorch 计算；CPU 回体积使用等价的 Numba 融合查表。GPU 默认 TF32、数据为 float32。表面重采样由 Workbench 在 CPU 完成；`device="cpu"` 的纯表面路径延迟加载 Torch/SciPy，并按显式 `OMP_NUM_THREADS` 拆分双半球总预算。 |
| `label` | `True` 使用最近邻体积采样及 Workbench `-label-resample`；`False` 使用三线性体积采样及 `-metric-resample`。回体积使用 CBIG 发布的最近顶点表。 |
| `wb_command` | Workbench 可执行文件路径或命令名，默认从环境 `PATH` 搜索。 |

输出体积只有皮层掩膜内赋值。多帧输入保持帧次序：NIfTI 为 `(X,Y,Z,T)`，GIFTI 每帧一个数据数组。标签值超过 float32 精确整数范围（2²⁴）时不适用。MNI 与表面之间的体素分辨率及表面顶点密度是两个独立参数。

体积输出显式保存为连续值 `float32` 或标签 `int32`，保留参考图的网格和仿射；参考图是整数掩膜或图谱时也不会将连续结果量化成整数。相同表面空间、相同密度的输入直接复制文件；`MNI152→MNI152` 网格重采样不属于本函数。

CPU 回体积将掩膜采样、双半球最近顶点查表及逐帧赋值合并为 Numba 内核，使用与原 PyTorch 路径相同的 float32 归一化坐标和最近偶数舍入。CPU 线程数遵循调用方的 PyTorch 预算和当前 Numba 线程上限，调用完成或失败后恢复原 Numba 设置。CUDA 继续使用原 PyTorch 采样内核，Workbench 球面重采样步骤仍由官方命令完成。

fsLR CIFTI 包含皮层下体素时，先按 HCP 方法用 Workbench 提取两侧 GIFTI，再将这两个文件作为 `source`。本函数只转换皮层部分，不把 CIFTI 中的皮层下数据投到皮层：

```bash
wb_command -cifti-separate input.dscalar.nii COLUMN \
  -metric CORTEX_LEFT L.func.gii -metric CORTEX_RIGHT R.func.gii
```

## 3. 命令行调用


```bash
fnit-space-convert \
  --volume /data/atlas_MNI152_1mm.nii.gz \
  --source-space MNI152 --target-space fsaverage --target-density 41k \
  --assets-dir /data/fnit_space_assets --output-dir /data/out/fsaverage41k \
  --device cpu

fnit-space-convert \
  --left /data/L.fsaverage.41k.func.gii --right /data/R.fsaverage.41k.func.gii \
  --source-space fsaverage --source-density 41k --target-space MNI152 \
  --reference /data/MNI152_ref_2mm.nii.gz \
  --assets-dir /data/fnit_space_assets --output-dir /data/out/mni2mm \
  --device cpu
```

纯 CPU 表面转换可显式设置总线程预算：

```bash
OMP_NUM_THREADS=8 fnit-space-convert \
  --left /data/L.fsaverage.41k.func.gii --right /data/R.fsaverage.41k.func.gii \
  --source-space fsaverage --source-density 41k --target-space fsLR --target-density 32k \
  --assets-dir /data/fnit_space_assets --output-dir /data/out/fsLR32k --device cpu
```

总预算为 1 时依次处理左、右半球；预算大于 1 时分给左右两个 Workbench 子进程，8 线程为 4+4。若调用进程已加载 PyTorch，同时遵循其较小的线程上限。未设置 `OMP_NUM_THREADS` 时沿用顺序调用及原环境。该并行分支仅用于两端都是表面的 `device="cpu"` 调用。

源码运行可把 `fnit-space-convert` 换为 `PYTHONPATH=src python -m fnit.space_conversion`，两者进入同一个 `main`。Workbench 必须在 `PATH` 中；Python API 可通过 `wb_command=` 指定其路径。

## 4. 原软件调用


CBIG 官方 MATLAB 正向投影使用 `CBIG_RF_projectMNI2fsaverage`，反向投影使用 `CBIG_RF_projectfsaverage2Vol_single`。示意命令如下；`lh_map`、`rh_map`、`reverse_map` 和 `mask` 均来自本功能安装的 CBIG 文件。

```matlab
[lh, rh] = CBIG_RF_projectMNI2fsaverage('/data/atlas_MNI152_1mm.nii.gz', 'linear', lh_map, rh_map);
[mni_cortex, mni_hemi] = CBIG_RF_projectfsaverage2Vol_single(lh, rh, 'nearest', reverse_map, mask);
```

HCP 2017 年的 fsaverage6 到 fsLR32k 左半球连续值命令为：

```bash
wb_command -metric-resample L.fsaverage6.func.gii \
  fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii \
  fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
  ADAP_BARY_AREA L.fsLR.32k.func.gii \
  -area-metrics fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii \
                fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii
```

右半球把 `L` 换为 `R`。反向交换源、目标球面及面积文件；标签图将 `-metric-resample` 换为 `-label-resample`。

## 5. 最新官方对照、耗时和脑图（2026-10-04）

在 CPU评测节点 的 Intel Xeon Gold 6418H 上，使用同一组物理核和 1/8 线程上限串行交替运行官方与 FNIT。每例预热一次，再比较三次完整进程运行的中位数。参考版本为 CBIG 固定 tag `v0.18.1-Update_stable_project_unit_test`、MATLAB R2018b、FreeSurfer 7.1.1 的原始 MATLAB/网格依赖、Workbench 2.0.0；输入来自 FSL 6.0.7.22 的实际 MNI152 模板和 Harvard–Oxford 图谱、HCP 2017 年平均顶点面积图。

两通道由原始 T1 模板和同网格去脑外组织的 T1 模板组成，已逐值核对来源；不是两次独立扫描。0.5 mm 参考是官方 1 mm 模板经 Workbench 重采样得到的网格。全部输入和资源先后校验 SHA-256；本次只公开模板派生图及聚合指标。

### 完整进程配对

主表的 FNIT 列使用统一 benchmark worker，计入 Python 启动、Torch 预配置、函数首次导入及完整读写。原版列包含 MATLAB/Workbench 启动、原函数和格式桥接。该表对应不可变的 `candidate_all_v8`；最新源码另通过 21 个真实输入检查，输出与该冻结版逐值相同，并确认每次调用恢复 Numba 线程设置。

| 转换 | 官方 1 线程 | FNIT worker 1 线程 | 官方 8 线程 | FNIT worker 8 线程 | 精度 |
|---|---:|---:|---:|---:|---|
| MNI152 T1 2 mm → fsaverage164k | 8.625 s | 2.220 s | 5.000 s | 2.158 s | 连续图左/右 MAE 0.001212/0.000987 |
| HCP fsaverage164k 平均面积 → 默认 MNI | 36.909 s | 5.101 s | 34.414 s | 4.580 s | 双半球/整图逐值相同 |
| HCP fsaverage41k 平均面积 → fsLR32k | 2.431 s | 4.270 s | 1.086 s | 2.645 s | 双半球/整图逐值相同 |
| Harvard–Oxford 皮层标签 → fsLR32k | 14.912 s | 8.532 s | 10.519 s | 4.505 s | 双半球/整图逐值相同 |
| fsLR32k 两个真实模板通道 → MNI 2 mm | 75.594 s | 10.191 s | 57.236 s | 6.027 s | 双半球/整图逐值相同 |

正向连续图保持现有 float32 PyTorch 插值：左右最大绝对差为 0.017090/0.016113，relative L2 为 2.848×10⁻⁷/2.276×10⁻⁷。默认反向、2 mm 反向、标签及纯 Workbench 路径均逐值一致，体积仿射差为零，全部输出有限。8 线程是计算预算上限；原版 MEX k-d tree 等步骤仍主要使用一个核。MATLAB 启动和共享存储会影响完整进程时间，完整重复序列和实际 CPU 利用见[公开数值报告](cpu_benchmark_20261004.public.json)。

### 本次优化后的实际表面 CLI

纯表面 CPU 路径现延迟加载仅体积计算需要的 Torch/SciPy，并复用项目的双半球并行实现。下表以新进程运行 `python -m fnit.space_conversion`，同样包含全部读写；官方为依次运行两侧 Workbench 命令。它与上方统一 worker 的口径分开记录。

| HCP fsaverage41k → fsLR32k | 官方 Workbench 链 | 原 CLI | 当前 CLI | 精度 |
|---|---:|---:|---:|---|
| 1 线程 | 2.385 s | 4.019 s | 2.731 s | 每次双半球逐值相同、intent 一致 |
| 8 线程 | 0.972 s | 2.604 s | 0.891 s | 每次双半球逐值相同、intent 一致 |

8 线程时当前 CLI 将总预算拆为 4+4，本例略快于官方顺序链。1 线程仍比裸 Workbench 慢约 14.5%，主要额外工作为 Python 启动和 GIFTI 校验；本项速度目标尚未达到。CUDA 计算路径沿用原有算子和 float32 精度。

### 当前源码的 H100 检查

最新组合源 v28 的 `space_conversion.py` 与 `_space_conversion_cpu.py` 和已测 `candidate_all_v17` 逐文件 SHA-256 相同，分别为 `91e3942b487cb91ee58e1310a7ab29aa670d31720e7abe6edb194e1a679a4afd`、`45911a090eff848c90db77eb547436eeba3a694a915652869181569a72881b6b`；版本绑定见[当前源码清单](../multimodal_cpu_20261004/source_v28_20261004.public.json)。沿用该相同源码在 H100 的正向与默认反向检查：各运行两组 AB/BA 进程对照，每个进程先完整预热一次，再计时三次含读写的 API。对照为本次优化前 FNIT，完整保存输出数组逐位相同，NIfTI 头和仿射、GIFTI intent/dtype/metadata 均一致。

| 完整 API | 优化前，两组中位数 | 当前，两组中位数 | 两版相同的峰值 allocation |
|---|---:|---:|---:|
| MNI152 T1 2 mm → fsaverage164k | 0.250 / 0.232 s | 0.229 / 0.241 s | 6,232,576 B |
| fsaverage164k → 默认 MNI | 2.252 / 2.160 s | 2.023 / 2.017 s | 205,128,704 B |

正向逐侧核对全部 163,842 个 float32 值，反向核对全部 16,777,216 个体素；两方向的最大误差均为 0、相同值比例为 100%。实测设备为共享 H100 PCIe，TF32 开启，进程 allocation 上限为 20 GB；这些时间是共享负载下的观察。逐次时间、源码 SHA-256 和完整数组逐位核对见[相同源码的 GPU 报告](../multimodal_cpu_20261004/gpu_final_20261004.public.json)。本轮 H100 只直接测试 MNI↔fsaverage 两条路径；fsLR 跨空间和密度转换的官方 Workbench CPU 结果单列，未计为 GPU benchmark。

### 内部调用和分步骤时间

统一 worker 内的 `run_case` 排除 Python 进程启动与公共 Torch 预配置，但仍包含函数首次导入、读取、完整计算和最终保存。本次未另外运行已预加载模块的多次 API 计时。

| 转换 | FNIT worker 内 1/8 线程 | 原 CBIG 函数与 MAT 保存 1/8 线程 |
|---|---:|---:|
| MNI152 T1 2 mm → fsaverage164k | 0.405/0.455 s | 0.760/0.481 s |
| HCP fsaverage164k 平均面积 → 默认 MNI | 3.274/2.666 s | 26.340/21.698 s |
| HCP fsaverage41k 平均面积 → fsLR32k | 2.545/1.023 s | —（仅 Workbench） |
| Harvard–Oxford 皮层标签 → fsLR32k | 6.801/2.845 s | 0.667/0.820 s |
| fsLR32k 两个真实模板通道 → MNI 2 mm | 8.306/4.348 s | 52.005/44.141 s |

原 CBIG 内部计时排除 MATLAB 启动及 Python 格式桥接，最终保存格式也与 FNIT 不同；两列用于解释工作量。官方完整链的步骤中位数如下，单位为秒；逐步中位数之和可与整链中位数不同。

| 官方链 | 步骤顺序 | 1 线程 | 8 线程 |
|---|---|---:|---:|
| MNI152 T1 2 mm → fsaverage164k | MATLAB 正向 → MAT/GIFTI 桥接 | 8.075 / 0.474 | 4.557 / 0.442 |
| HCP fsaverage164k 平均面积 → 默认 MNI | GIFTI/MAT 桥接 → MATLAB 反向 → MAT/NIfTI 桥接 | 0.431 / 35.460 / 0.966 | 0.364 / 33.198 / 0.852 |
| HCP fsaverage41k 平均面积 → fsLR32k | Workbench 左 → 右 | 1.235 / 1.196 | 0.555 / 0.455 |
| Harvard–Oxford 皮层标签 → fsLR32k | MATLAB nearest → 桥接 → Workbench 左标签 → 右标签 | 7.867 / 0.436 / 3.326 / 3.311 | 7.490 / 0.423 / 1.289 / 1.317 |
| fsLR32k 两个真实模板通道 → MNI 2 mm | Workbench 左 → 右 → 输入桥接 → MATLAB 两帧反向 → 输出桥接 → 2 mm 重采样 | 3.237 / 3.333 / 0.416 / 66.655 / 1.521 / 0.595 | 1.222 / 1.368 / 0.390 / 52.452 / 1.605 / 0.591 |

### 其余 20 个功能变体

以下各运行一次，使用同一 1 线程预算；这些数值是功能覆盖观察。表面输入来自真实 HCP 平均面积，标签来自原 Harvard–Oxford 图谱经官方投影，多通道来自真实模板。`同空间复制`只校验文件和数据，不能作为算法加速参照。

| 变体 | 官方完整进程 | FNIT worker 完整进程 | 最大绝对差（左/右或体积） |
|---|---:|---:|---:|
| MNI 1 mm → fsaverage3k | 34.115 s | 7.627 s | 0.001465 / 0.000977 |
| MNI 1 mm → fsaverage10k | 38.567 s | 8.277 s | 0.003906 / 0.002441 |
| MNI 1 mm → fsaverage41k | 80.936 s | 9.782 s | 0.009766 / 0.006836 |
| MNI 2 mm → fsLR59k | 17.792 s | 9.031 s | 0.007812 / 0.006836 |
| MNI 2 mm → fsLR164k | 41.383 s | 12.886 s | 0.015137 / 0.012695 |
| fsaverage3k → 10k | 0.583 s | 2.371 s | 0 / 0 |
| fsaverage10k → 41k | 1.685 s | 3.522 s | 0 / 0 |
| fsaverage41k → 164k | 7.094 s | 9.081 s | 0 / 0 |
| fsLR32k → 59k | 2.789 s | 4.574 s | 0 / 0 |
| fsLR59k → 164k | 6.593 s | 8.331 s | 0 / 0 |
| fsLR164k → 32k | 6.695 s | 8.532 s | 0 / 0 |
| fsLR59k → fsaverage10k | 2.235 s | 4.224 s | 0 / 0 |
| fsLR164k → fsaverage3k | 5.392 s | 7.279 s | 0 / 0 |
| fsaverage3k → MNI 2 mm | 50.686 s | 9.082 s | 0 |
| fsaverage164k → MNI 0.5 mm 扩展 | 61.021 s | 7.179 s | 1.311056 |
| fsaverage41k 标签 → fsLR59k | 3.137 s | 4.874 s | 0 / 0 |
| fsaverage41k 标签 → 默认 MNI | 66.899 s | 12.090 s | 0 |
| MNI 两个真实模板通道 → fsaverage10k | 27.856 s | 7.830 s | 0.003906 / 0.002441 |
| fsLR32k 同空间复制 | 0.026 s | 2.019 s | 0 / 0 |
| fsaverage41k 标签同空间复制 | 0.019 s | 1.969 s | 0 / 0 |

### 自定义 0.5 mm 网格的取整合同

CBIG 原反向函数只直接输出其自身 256³ 掩膜网格。指定 `reference` 属于 FNIT 扩展：沿用 CPU/GPU 的 float32 归一化坐标及最近偶数舍入；当坐标恰为半体素时，可能与 Workbench `ENCLOSING_VOXEL` 的 `floor(voxel+0.5)` 不同。默认网格与本次 2 mm 网格的结果逐值一致，不能由此推广到所有自定义网格。

本次 0.5 mm 图形状为 `(363,435,363)`，两套完整输出最大绝对差 1.311056、MAE 0.006323、relative L2 0.224661，相同体素比例 92.6735%。全部 4,199,539 个差异均落在半体素且最近索引不同的位置：包含 429,826 个皮层掩膜边界差异和 3,769,713 个非零顶点值差异。对原版 CBIG 原生输出分别应用两种取整模型，可各自逐值重建 FNIT 与 Workbench 结果，模型最大误差均为零。输出仍沿用现有 CPU/GPU 规则；此扩展网格未宣称与 Workbench 严格等价。

### 公开模板脑图

![公开 MNI152 模板：原版 CBIG 与 FNIT 的 fsLR32k 到 MNI 2 mm 对照](../../docs/space_conversion/assets/public_space_comparison.png)

两通道均由公开 FSL 模板生成。本图展示第一个通道的三个切面；两套完整 `(91,109,91,2)` 输出逐值相同，仿射相同、全部有限，最大绝对差为零。差异行中的灰度为 T1 背景。可编辑矢量图见 [SVG](../../docs/space_conversion/assets/public_space_comparison.svg)，绘图及输出 SHA-256 见[图像聚合报告](../../docs/space_conversion/assets/public_space_comparison.json)和[绘图脚本](../../docs/space_conversion/plot_public_comparison.py)。

### 功能范围和复现

| 功能 | 真实输入方案 | 官方对照 |
|---|---|---|
| MNI152→fsaverage，3k/10k/41k/164k | FSL 实际 MNI152 T1 模板；整数图谱用于标签路径 | 原版 `CBIG_RF_projectMNI2fsaverage`；低密度再接 Workbench 面积校正重采样 |
| MNI152→fsLR，32k/59k/164k | 同一模板或图谱 | 原版 CBIG 正向投影再接官方 Workbench |
| fsaverage/fsLR 跨空间与同空间换密度 | HCP 官方平均顶点面积图；真实模板投影形成的标签图 | 官方 Workbench `ADAP_BARY_AREA` |
| fsaverage/fsLR→MNI152 默认网格 | 同一真实连续值或标签表面图 | 官方 Workbench 转 fsaverage164k 后调用原版 CBIG 反向函数 |
| 表面→指定参考网格 | 实际 2 mm 参考图；0.5 mm 为原模板重采样 | CBIG 默认输出再接明确指定的 Workbench 最近体素链；单独报告扩展网格的舍入差异 |
| 多帧 | 实际 4D 标准空间影像或明确标注的真实模板多通道组合 | 原版 CBIG 正向多帧；反向按原函数要求逐帧运行 |
| 相同表面空间和密度 | 双半球连续值或标签 GIFTI | 文件复制与逐帧数据/元数据核对 |

原版反向 `CBIG_RF_projectfsaverage2Vol_single` 使用 `.prop.mat` 中的球面坐标及官方 fsaverage 网格、k-d tree 来计算最近顶点；FNIT 使用上游发布的 `.vertex.mat` 预计算表。正式官方对照需另外准备原始 `.prop.mat` 与原版 MATLAB/MEX 依赖，仅供 benchmark 使用。单独用 `.vertex.mat` 查表的桥接脚本不能作为该原函数完整运行的计时参照。自定义参考网格是 FNIT 扩展，CBIG 原函数只直接输出自身掩膜网格。


统一适配器为 [`tools/benchmark_multimodal_cpu_space.py`](../../tools/benchmark_multimodal_cpu_space.py)，合同为 `run_case(case, output_dir, device)`、`reference_command(case, output_dir, resources)` 和 `reference_outputs(...)`，输出键为 `left/right` 或 `volume`。核心 case 字段包括 `id`、`source`、源/目标空间与密度、`assets_dir`、`reference`、`label`、`wb_command`；reference resources 指定原版 MATLAB、CBIG、FreeSurfer 及 Workbench 路径。它们只用于对照环境。

配置本机资源 manifest 后，完整进程配对命令如下。CPU 集合应选同一组物理核，实际 Workbench 及其依赖库对两侧使用同一环境。

```bash
python tools/benchmark_multimodal_cpu.py run \
  --manifest /data/space_cpu_manifest.json \
  --candidate-root /data/fnit_source --baseline-root /data/fnit_baseline \
  --output-dir /data/space_cpu_results --cpuset 66,70,74,78,82,86,90,94 \
  --threads 1,8 --repetitions 3 --api-repetitions 0 --backends official,candidate
```

## 6. 最近更新和 benchmark 记录

| 日期 | 变更 | 验证 |
|---|---|---|
| 2026-10-04，本次 CLI 更新 | 纯 CPU 表面路径延迟加载 Torch/SciPy；在显式总预算下复用双半球并行 | 1/8 线程实际 CLI 预热+三次配对；双半球每次逐值一致；21 项真实输入输出门槛全部通过，Numba 设置恢复。1 线程 CLI 速度目标仍未达。 |
| 2026-10-04，CPU 融合版 | 回体积合并掩膜、顶点查表和赋值；修复整数参考图量化连续输出；添加原版 CBIG/Workbench 适配器 | 10 个主例预算组合和 20 个功能变体；默认与 2 mm 反向、标签、纯表面逐值相同；0.5 mm 扩展取整差异已完整定位。 |
| 2026-09-29 | RF-ANTs 正反向、HCP 面积校正、多密度 API | 当时的 WSL 实际模板/沟深图及内部计时保留于[历史验证记录](README.md)。 |

定向测试覆盖最近邻边界、多帧双半球、输出 dtype、官方 MAT/NIfTI 轴顺序、异常时 Numba 状态恢复，以及不依赖体积计算库的纯表面 CLI。这些小数组用于回归检查，benchmark 使用上方的真实模板和图谱。

## 7. 参考文献和原实现代码库


1. Wu J, et al. Accurate nonlinear mapping between MNI volumetric and FreeSurfer surface coordinate systems. *Human Brain Mapping* 39:3793–3808, 2018. [DOI](https://doi.org/10.1002/hbm.24213)；[CBIG RF-ANTs 原代码与映射](https://github.com/ThomasYeoLab/CBIG/tree/v0.18.1-Update_stable_project_unit_test/stable_projects/registration/Wu2017_RegistrationFusion)。
2. Coalson TS, Van Essen DC, Glasser MF. Resampling between FreeSurfer and HCP fsLR spaces, 2017. [HCP 官方操作说明](https://wiki.humanconnectome.org/docs/assets/Resampling-FreeSurfer-HCP_5_8.pdf)；[HCPpipelines 模板资产](https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage)。
3. Glasser MF, et al. The minimal preprocessing pipelines for the Human Connectome Project. *NeuroImage* 80:105–124, 2013. [DOI](https://doi.org/10.1016/j.neuroimage.2013.04.127)；[Connectome Workbench 原代码](https://github.com/Washington-University/workbench)。
