# CPU 保序质心接入 M0：真实初始化前缀对照

## 1. 功能简介与验证范围

将已经验证的 CPU 保序质心归约接入 FNIT 原有初始化前缀。原图像准备、坐标变换和 M0 构造保持原实现，在私密注册模块中替换 CPU `_centroid` 的调用。

一个冻结的真实刚体输入对完成了一次连续的 FNIT 准备和初始化：两幅准备图像的全部体素字节、几何字段、两个质心，以及 M0、Rsrc、Rtrg 均逐位匹配已保存的 FreeSurfer 源码 SDK 参考。此次参考由独立 SDK 观察器生成；它的编译器与安装版 FreeSurfer 不同，因此按 SDK 参考记录。

```mermaid
flowchart LR
    A[真实 moving / fixed 输入] --> B[原 FNIT 图像准备]
    B --> C[CPU 保序质心]
    C --> D[原 M0 构造]
    D --> E[保存准备图像与几何 / M0]
    E --> F[对照已保存 SDK 参考]
```

停止位置位于金字塔、Schur 和 QR 之前。完整 robust-register 正式验收仍为 **17/20**。默认后端和生产源码未改变。

## 2. Python 调用、输入与输出

公开可复核函数复用前一阶段的 [centroid_serial.py](../centroid_serial_cpu_probe_20261006/centroid_serial.py)，没有复制一份相同内核。函数说明及调用条件见 [保序质心说明](../centroid_serial_cpu_probe_20261006/README.md)。本次 M0 集成属于私密候选入口，尚无新增生产 Python API。

下面只演示公开质心函数，不将独立调用标为完整 M0 重建：

```python
from pathlib import Path
import importlib.util
import nibabel as nib
import numpy as np

# 修改为本仓库中公开验证函数的路径。
centroid_module_path = Path(
    "validation/robust_register/centroid_serial_cpu_probe_20261006/centroid_serial.py"
)
centroid_spec = importlib.util.spec_from_file_location(
    "fnit_centroid_cpu_probe", centroid_module_path
)
centroid_module = importlib.util.module_from_spec(centroid_spec)
centroid_spec.loader.exec_module(centroid_module)

# 输入已准备的三维影像。调用者确认有限值、正总强度和匹配网格。
prepared_image_xyz = nib.load("prepared_image.nii.gz").get_fdata(dtype=np.float32)
number_x, number_y, number_z = prepared_image_xyz.shape
# 原 C++ 按 d、h、w 迭代，因此 X 是展开后变化最快的维度。
prepared_values_x_fast = prepared_image_xyz.ravel(order="F").copy()
prepared_values_x_fast.setflags(write=False)
outside_value = np.float64(0.0)  # 本次已验证准备图像的 outside 为零。

centroid_voxel_one_based = centroid_module.centroid_serial(
    prepared_values_x_fast,
    number_x,
    number_y,
    number_z,
    outside_value,
)
```

| 参数 | 格式与意义 |
| --- | --- |
| `prepared_values_x_fast` | 一维、连续、只读 Float32 数组，长度必须为 `number_x * number_y * number_z`；来自准备后的 XYZ 网格。 |
| `number_x`、`number_y`、`number_z` | 三个正整数，分别是网格 X、Y、Z 尺寸。无默认值。 |
| `outside_value` | Double outside 值；本次只验证 `0.0`。无默认值。 |
| 返回值 | 三个 Double 质心坐标，使用原实现的 **1 起始体素坐标**。它们不是 scanner RAS 毫米坐标。 |

本次集成候选的输出包含：source/target 两幅 Float32 准备数组、XYZ 形状及几何字段；source/target 三维 Double 质心；三个 4×4 Double 矩阵 M0、Rsrc、Rtrg；完整比较及分步骤时钟。

候选只接受前次 FNIT 已生成并验收的两幅准备数组。其他 CPU dtype、布局、梯度张量或输入域调用原函数；GPU 张量在第一分支直接调用原函数。当前两幅图的正总强度依据原函数成功返回作源码推论，未新增完整输入有限值或质量检查，未扩大为通用生产保证。

## 3. 命令行调用

本阶段没有新增生产 CLI。准备状态和 M0 是内部阶段，公开质心函数用 Python 调用。完整注册的现有命令仍使用原默认实现；本次私密候选未接入该命令。

## 4. 原软件调用与参考

对应内部步骤是 FreeSurfer robust-register 的图像准备、`CostFunctions::centroid` 和初始变换构造，没有独立的官方 M0 命令。完整注册的上下文示例：

```bash
OMP_NUM_THREADS=8 mri_robust_register \
  --mov moving.mgz \
  --dst fixed.mgz \
  --lta registered.lta \
  --sat 50
```

该命令属于完整注册。本次执行的是停止在金字塔之前的独立源码 SDK 观察器，其已保存输出只供隔离比较；当前候选计算没有读取参考质心或 M0。

## 5. 最新真实精度与运行时间

### 精度

| 输出 | 比较规模 | 本次差异 |
| --- | ---: | --- |
| source 准备图像 | 107,055 个 Float32 体素 | 全部原始字节一致，0 个不同体素 |
| target 准备图像 | 107,055 个 Float32 体素 | 全部原始字节一致，0 个不同体素 |
| 两幅图的形状、dtype、布局、RAS、outside 与几何 | 全部保存字段 | 全部一致 |
| source / target 质心 | 6 个 Double | 逐位一致，最大绝对差 0 |
| M0 | 16 个 Double | 逐位一致，最大绝对差 0 |
| Rsrc / Rtrg | 32 个 Double | 逐位一致，最大绝对差 0 |

共核对 **54 个 Double 字**。使用原 Torch 分组归约时，该输入的 M0 最大绝对差为 `4.1211e-13`；替换保序归约后归零。这定位了本例初始化尾差的来源。首轮 A/b、QR 解和后续收敛仍需逐段对照。

独立复核读取原 SDK、候选及比较文本，逐词检查 54 个整数位模式；两幅准备数组的全字节比较由实际 worker 完成。此次未重跑参考观察器，原生执行次数为零。

### 计时

| 层级或步骤 | 本次时间 |
| --- | ---: |
| 候选初始化入口，包含导入、准备、JIT、库身份检查与保存 | 15.0791 s |
| 完整观察组 worker，含资源校验与保存参考比较 | 29.1198 s |
| 外层观察组 | 29.3008 s |
| 额外 Numba 导入 | 0.2027 s |
| 冷 typed JIT | 0.4707 s |
| target 质心：一次真实核调用 | 0.5498 ms |
| source 质心：一次真实核调用 | 0.2338 ms |
| target / source 展开与哈希 | 0.5810 / 0.6391 ms |
| 四次库身份检查 | 分别 3.0732、3.1665、3.1648、3.1918 s |

这些是嵌套时钟，不能相加为总耗时。CPU 预算为 8 线程；质心内核串行，禁用 fastmath、parallel 和 cache，进程地址空间上限为 20,000,000,000 字节。

未记录原生质心或 M0 的独立时钟，也未完成完整注册的配对速度实验，因此暂不计算提速比。此处 15.0791 秒是带完整身份检查的初始化入口，不是 recon-all 或完整注册时间。

准备、typed JIT 后两次核调用之前、完整输出保存之后的库身份门分别观察到 100、123、123、123 个实际库文件，均精确属于已测得的 123 个文件集合。该集合允许子集，不按名称或目录前缀放行。库文件身份与 LLVM 名称检查均不作为动态 BLAS 调用轨迹。

本次不生成额外脑图：全体素字节差为零，记录完整比较规模及零差结果。此前真实输入、原失败与原始输出保留在私密验证目录。

## 6. 最近阶段记录与剩余工作

| 阶段 | 实际结果 |
| --- | --- |
| 原准备 / M0 观察 | 准备图像、几何、Rsrc/Rtrg 已一致；Torch 质心和 M0 有 Double 尾差。 |
| 裸保序质心与元数据闭合 | 保存的六个 Double 字逐位匹配；保留原 observer 失败记录，随后仅补完整库身份元数据。 |
| 共同导入库审查 | 一次 import-only；123 个实际库文件闭合，GPU 未初始化。JIT、图像准备与数值计算为零。 |
| 本次完整 CPU 初始化前缀 | 原准备自产、两次保序质心与完整 M0 连续执行一次；全部准备、几何和 54 个 Double 字匹配。 |
| 本次收尾 | 18 份源码和 473 个绑定资源前后一致；原进程全部退出、共用锁释放、任务授权关闭。元数据收尾没有新增科学计算。 |

剩余：首轮半程变换、A/b 构造和 QR 解；后续注册与原 20 项验收；未知 CPU 输入域、通用异常处理及 GPU 回归。在这些门通过前，保留候选入口并继续使用原生产实现。

## 7. 原代码与参考文献

- [FreeSurfer robust-register 原代码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register)
- [源码中的质心函数](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register/CostFunctions.cpp#L1308)
- [前次公开保序质心函数与说明](../centroid_serial_cpu_probe_20261006/README.md)
- [完整机读摘要](benchmark_summary.json)
- Reuter M, Rosas HD, Fischl B. Highly accurate inverse consistent registration: a robust approach. *NeuroImage*. 2010;53(4):1181–1196. [DOI](https://doi.org/10.1016/j.neuroimage.2010.07.020)
