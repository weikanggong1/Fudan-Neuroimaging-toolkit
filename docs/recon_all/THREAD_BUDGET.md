# recon-all 的 CPU 线程预算

## 作用与限制

`threads=4` 原先主要传给 `torch.set_num_threads` 和支持线程选项的表面程序。
它没有设置 Numba，也没有统一原生子进程的 OpenMP、BLAS 或 ITK 环境。
已有 gpucw1 整例记录的 Numba 掩码为 128。因此不能把原报告里的
`threads=4` 解释为各阶段已经使用同一个 CPU 预算。

当前公开重建入口已接入 `thread_budget.py` 的 context manager，默认
`threads=4` 同时设置 Torch intraop 和调用线程的 Numba 掩码，退出恢复
调用方状态。它不改变算法、影像、精度、CUDA allocator 或坐标空间；没有对应的
FreeSurfer 独立命令，属于 FNIT 调度的线程管理。依赖的 Torch 和 Numba
已包含在主页 `environment.yml`，不增加依赖或调用外部脑影像软件。
原生子进程的环境副本助手尚未接入各生产 wrapper，不能将 Python 预算
当作所有原生程序的实际线程数。本次已完成双例 CA normalize 和 sub-01
LH 标准球面的真实同输入 128/4 掩码回归，范围及时间见后文。

### Python 作用域

`thread_budget(threads=4)` 是 context manager。`threads` 为正整数，单位
是 CPU 工作线程，默认 4；布尔值、非整数或小于 1 时抛出 `ValueError`。
进入时设置 Torch intraop 和当前调用线程的 Numba 掩码，正常退出或异常
退出均恢复调用方原值。其返回报告可写入 JSON，退出后补充恢复结果：

| 字段 | 结构和含义 |
|---|---|
| `requested_threads`、`scope` | 请求数量和作用范围 |
| `torch.before/effective/restored` | 进入、生效、恢复后的 intraop 数量 |
| `torch.interop_unchanged` | 未修改的 interop 数量 |
| `numba.before/effective/restored` | 进入、生效、恢复后的 Numba 掩码 |
| `numba.initial_capacity` | Numba 导入时的线程容量上限 |
| `numba.mask_only`、`threading_layer` | 掩码性质和实际后端 |
| `environment_modified` | 恒为 false；不修改父进程环境 |
| `restoration_complete` | 退出后两项设置是否恢复 |

Numba 的 `get_num_threads()` 会初始化其线程池。设置掩码为 4 不会将已
建立的 128 个线程全部删除，不能把该值当作操作系统看到的进程总线程数。
请求超过 `NUMBA_NUM_THREADS` 的初始容量时，在修改 Torch 之前明确报错，
不能在已导入的进程中扩容；应在新进程导入 Numba 前设置该环境变量。

Torch intraop 是进程设置；不同调用线程不应同时修改。Numba 掩码属于
当前调用线程。助手不设置 Torch interop，不覆盖已初始化 BLAS 的内部
线程池，也不承诺多个库或多个子进程同时占用的 CPU 数量等于 `threads`。
后端设置失败会传播原异常，设置未生效会抛 `RuntimeError`。

```python
import json
from fnit.recon_all.thread_budget import thread_budget
from fnit.recon_all.sphere_standard_run import run_standard_sphere

with thread_budget(
    threads=4,  # 当前作用域的 Torch intraop 和 Numba 工作掩码
) as thread_report:
    sphere_report = run_standard_sphere(
        inflated="/data/sub-01/surf/lh.inflated",  # surface RAS 坐标，单位 mm
        smoothwm="/data/sub-01/surf/lh.smoothwm",  # 同顶点顺序和同有序面
        output="/data/diagnostic/lh.sphere",  # 新诊断输出，不覆盖生产目录
        finish_device="cpu",  # 保留本次冻结输入的清理设备
    )
# 此时报告已经记录恢复数量；示例不表示该真实影像已经运行过。
print(json.dumps({"threads": thread_report, "stage": sphere_report}, indent=2))
```

### 原生子进程环境

`native_thread_environment(threads=4, environ=None)` 返回
`(child_env, report)`。`environ=None` 时复制当前完整环境；显式 Mapping
作为待复制的环境。它将以下变量设为预算的十进制字符串：

`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、
`NUMEXPR_NUM_THREADS`、`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`、
`NUMBA_NUM_THREADS`。

`report` 包含请求、作用范围、这些变量的前后值，以及
`parent_environment_modified=False` 和 `active_worker_counts_verified=False`。
它不记录其余环境值，不启动程序，不修改 `os.environ`。非法线程数抛
`ValueError`；环境不能转换为字典时传播 Python 类型错误。示例：

```python
import os
import subprocess
from fnit.recon_all.thread_budget import native_thread_environment

native_environment = dict(
    os.environ,
    FREESURFER_HOME="/data/fnit-assets",  # FNIT 声明资产，非预装程序目录
)
child_environment, child_thread_report = native_thread_environment(
    threads=4,  # 新子进程初始化对应库时读取的线程预算
    environ=native_environment,  # 复制已有环境，保留设备映射和 allocator 设置
)
subprocess.run(
    args=["/opt/conda/envs/fnit/bin/mri_em_register", "-uns", "3",
          "-mask", "brainmask.mgz", "nu.mgz", "/data/fnit-assets/average/RB_all_2020-01-02.gca",
          "transforms/talairach.lta"],  # 已有固定阶段的 Conda 源码构建命令
    cwd="/data/sub-01/mri",  # 已有阶段的相对输入基准目录
    env=child_environment,  # 仅影响该新进程
    check=True,  # 原生阶段失败时传播异常
)
```

环境变量不保证程序采用该预算，显式线程选项仍须与预算一致。FNIT
`tools/n4_itk/n4_itk.cpp` 在各滤波器构造前调用
`SetGlobalDefaultNumberOfThreads(1)`，N4 是源码明确的单线程例外。
设置 ITK 环境为 4 不能据此宣称 N4 已使用 4，也没有修改这个默认。
拓扑 GA 原有 `-threads 1` 和 `OMP_NUM_THREADS=1` 同样须保留。
此前 2026-09-30 的 e036 双例原生日志中，`mri_em_register` 的 OpenMP
可用数量和结束时 `FSRUNTIME` 都为 1；不能把 Numba 的 128 掩码归给 EM。
固定源码支持 `-threads`，当前 wrapper 没有传入该选项；WM 编辑日志没有
实际线程数条目，不能从顶层配置推断。

2026-10-01 的驱动在程序启动前显式设置 OMP/BLAS 预算为 4。两个 1b
候选和两个 e036 受控基线的 EM 日志均记录 `FSRUNTIME ... 4 threads`，
与此前历史环境不同。[当前日志条目及来源快照](../../validation/recon_all/python_gpu_port/performance_20261001/thread_runtime_logs_snapshot.json)
保留字节数、SHA-256 和行号；采集时仍在运行的基线日志明确标为非最终。
这个记录证明 EM 的实际条目，不能推广到其他原生程序的活跃线程数。
两个独立进程并行且总预算为 4 时，各进程应分配 2；给每个进程 4 会
扩大总预算。使用已初始化 CUDA 的 API 时应采用新 exec/spawn 进程，
不能 fork 继承 CUDA 状态后执行计算。

## 并行核的写入独立性

| 现有核 | `prange` 的独立写入 | 顺序依赖 |
|---|---|---|
| `sphere_standard_average.average_standard_gradient` | 每个顶点只写下一轮数组的自身三分量 | 顶点内邻居累加顺序固定；全部顶点完成后才交换两轮数组 |
| `ca_normalize_python._convolve_short` | 每个 x 切片只写该切片的体素 | 单体素卷积累加和舍入顺序固定；三个轴依次完成 |
| `normalization.normalize_gaussian_source._convolve_axis_strict` | 每个 z 切片只写该切片的体素 | 单体素累加顺序固定；三个轴依次完成 |

这些核没有跨 `prange` 任务的浮点求和归约，改变工作掩码有保持结果的
代码基础，仍须同输入实测。第二次归一化的初始 aseg bias 使用第三个 CPU
核，即使后面的 gentle bias 路径在 GPU 上，也不能忽略其 Numba 预算。
球面配准的 `_average_numpy` 则是串行 `@njit`，增加 Numba 线程不会把它
变成并行函数。也不能据此将双侧 `_surface_pair` 直接并发：双方顺序写入
同一个 `mri/surface.defects.mgz`，LH 初始化、RH 合并存在文件依赖。

## 已完成的验证

本次新增 9 项标准库 unittest，覆盖参数检查、Numba 上限、有效掩码记录、
正常/异常/嵌套恢复、后端设置失败、未生效设置和环境副本。2026-10-01
本地 9/9，0.402 s；headcw 独立临时目录 9/9，0.012 s。
这些是状态管理测试，不是影像 benchmark。

同日 headcw 使用项目 Conda 的 Torch 2.5.1、Numba 0.67.0 验证真实 API：
Torch 96、Numba 192（TBB）进入，作用域内为 4/4，退出恢复为 96/192。
请求 193 明确报错。`CUDA_VISIBLE_DEVICES=''`，CUDA 全程未初始化。
实测助手 SHA-256 为
`bf0b40e5c6c7232d0de23dee3c491a3c27850d79922d81df71689933a9692fca`。

已有 e036f57 冻结 sphere cProfile 中，`average_standard_gradient` 的
LH 累计 11.21 s、RH 6.36 s；`_distance_sse` 为 98.12/50.54 s。
这些带剖析开销的单次记录说明线程预算需要记录，但不能证明 128 掩码
是此次 WM 666 s 或整例变慢的原因。WM 的原生自报时间包含其内部执行
和读写，应继续使用独立 CPU/墙钟与子步骤记录定位。

### 2026-10-01 真实冻结阶段：128 对 4

本次在 nodecw10 使用完整提交
`1b8c36d25a68e253a1e59b6d02114890afa467de`，Torch 2.5.1、Numba 0.67.0。
Torch intraop 在两组均为 4；Numba 初始容量和掩码均为 192，各阶段先执行
128 掩码，再执行 4，结束恢复 192。下面时间包括输入/输出和首次 JIT，
没有预热、交换顺序或重复配对，因此用于结果回归，不是线程性能验收。

| 真实冻结输入/阶段 | Numba 128（s） | Numba 4（s） | 同输入结果 |
|---|---:|---:|---|
| sub-01 CA normalize | 20.328 | 22.838 | norm/ctrl 全部元素差 0；dtype、形状和 affine 相同 |
| sub-02 CA normalize | 20.447 | 23.015 | norm/ctrl 全部元素差 0；dtype、形状和 affine 相同 |
| sub-01 LH standard sphere | 209.395 | 218.038 | 顶点数与有序面相同；坐标差 0 |

两例 CA normalize 的 `norm.mgz` 和 `ctrl_pts.mgz` 最大/P99 元素误差均为 0，
affine 最大误差为 0 mm。球面的最大/P99 坐标误差均为 0 mm，顶点对应关系
成立后才作同索引比较。这些是数组、dtype 和几何回归，不宣称所有压缩文件
字节一致。记录中掩码 4 没有比 128 更快；一次顺序固定、包含首次 JIT 的
时间不能证明其稳定速度，也不能用于解释此前 WM 的 666 s 异常。

这三份报告绑定实际输入和所有 recon-all Python 源文件 SHA-256；整理此页
时已确认以下四个相关模块与当前工作区匹配：

| 实际测试模块 | SHA-256 |
|---|---|
| thread_budget.py | `bf0b40e5c6c7232d0de23dee3c491a3c27850d79922d81df71689933a9692fca` |
| ca_normalize_python.py | `abb436826e600101c2c5087e2367e8a36a7c038c99eda10bc4f5be1e7486661a` |
| sphere_standard_run.py | `7544ccb77153a7ac242b88d1c9752bdd08ce5765ba483cd644a4075e94e90405` |
| sphere_standard_average.py | `f79dc72a574d3bef756f06b97ab2546d35662c1d6acbd769e034fa542d84bd78` |

实际 [`benchmark_thread_budget.py`](../../validation/recon_all/python_gpu_port/benchmark_thread_budget.py)
的 SHA-256 为
`97d79c25253bb38a73acf3cccb5bd4ffe4b4293f2b61da9f20926e0eb9e30828`。
报告通过已有 headcw ControlMaster 只读获取，没有复制影像、权重或许可证。

| 原始报告 | 文件 SHA-256 |
|---|---|
| [sub-01 CA](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_ca_sub01/report.json) | `73c6074d6fd761f9f9e17a67ac9c2a9d597d88ff280dc0ea5e1f6ee522e30e0d` |
| [sub-02 CA](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_ca_sub02/report.json) | `f0a5ecd72bf33c9336d45a3377f963b96adbcdb447aab2ffc488092f4a3e778d` |
| [sub-01 LH sphere](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_sphere_sub01_lh/report.json) | `5eb86d5ed201d899c3507fc1ddc49a944798ceb605acfe73052db017f87ebea0` |

输入 CA 链仅为同一例 FNIT 自产 `nu`、`brainmask`、LTA 和声明 GCA，
球面仅为同一例 FNIT 自产 `lh.inflated`、`lh.smoothwm`。没有用官方结果
补输入，阶段算法、标签与坐标定义均保持不变。

尚未完成 sub-01 RH 及第二例的标准球面掩码回归、预热后的顺序交换计时
和原始 T1 受控整例。本页不宣称整例完成、提速或整体指标等效。

## 复现与补齐命令

沿用仓库已执行的脚本，诊断输出必须是不存在的新目录。下面分别复现
sub-01 LH sphere 和 CA normalize；补齐 RH 或 sub-02 时显式修改半球/被试，
不要覆盖本次报告。两例 CA 和本例 LH 已有结果，无需为了文档重复运行。

```bash
# 已安装本次代码的 FNIT 源目录；当前进程只使用 CPU。
source_directory=/path/to/FNIT/src
python_executable=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
subject_directory=/data/fnit/sub-01  # 本次相同的 FNIT 自产冻结被试
assets_directory=/data/fnit/assets  # 已声明资产，包含实际 GCA
output_directory=/data/diagnostic/thread_sphere_sub01_lh_repeat  # 必须不存在
code_commit=1b8c36d25a68e253a1e59b6d02114890afa467de  # 实际测试源码提交

PYTHONPATH="$source_directory" CUDA_VISIBLE_DEVICES='' "$python_executable" \
  validation/recon_all/python_gpu_port/benchmark_thread_budget.py \
  --subject "$subject_directory" \
  --assets "$assets_directory" \
  --output "$output_directory" \
  --stage sphere \
  --hemi lh \
  --masks 128 4 \
  --torch-threads 4 \
  --code-commit "$code_commit"
```

`--subject` 是自产阶段输入；`--assets` 是声明资产；`--output` 是隔离输出；
`--stage` 选择 sphere/ca；`--hemi` 默认 lh、仅 sphere 使用；`--masks` 是
Numba 运行顺序；`--torch-threads` 为相同 Torch 预算；`--code-commit` 必须
记录实际部署提交，不能沿用与当前代码无关的标签。CA 的具名参数为：

```bash
output_directory=/data/diagnostic/thread_ca_sub01_repeat  # 另一个不存在的新目录
PYTHONPATH="$source_directory" CUDA_VISIBLE_DEVICES='' "$python_executable" \
  validation/recon_all/python_gpu_port/benchmark_thread_budget.py \
  --subject "$subject_directory" \
  --assets "$assets_directory" \
  --output "$output_directory" \
  --stage ca \
  --masks 128 4 \
  --torch-threads 4 \
  --code-commit "$code_commit"
```

此脚本记录实际掩码、输入/源代码哈希、完整阶段读写和数组/网格比较。
进一步性能判断须先预热并重复交换顺序；整例需从原始 T1 和空目录开始。
线程管理本身没有独立官方命令，阶段参考对应 `mri_ca_normalize` 和
`mris_sphere`，官方程序仅在独立 benchmark 路径使用。
