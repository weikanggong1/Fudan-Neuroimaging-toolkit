# recon-all 的计时、线程与显存剖析

PyTorch 2.5.1 的 native allocator 根据 `PYTORCH_NO_CUDA_MEMORY_CACHING` 是否存在决定绕过缓存，`0` 和空字符串也会关闭。`enabled` 在初始化前移除该变量；`auto` 保留已有值。已初始化 API 的实际策略仍标为未知，不能用后来修改的环境推断。依据为[实际安装提交的分配器源码](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/c10/cuda/CUDACachingAllocator.cpp#L3129-L3133)。本次整例使用值 `1`，这项报告边界修正没有改变其计算或缓存策略。

[重建入口](README.md) · [线程预算](THREAD_BUDGET.md) · [SynthSeg 实际精度](SYNTHSEG_PRECISION.md)

## 默认运行与诊断运行

`run_recon_all_python(..., profile_stages=False)` 和 CLI 默认不增加阶段
CUDA 同步。`StageProfiler` 的 `synchronize` 参数必须显式传入；入口默认
传入 `False`，需要剖析时传入 `True`。关闭剖析不会删除各算法本来已有的
同步，也不会减少必要计算、文件写出或质量检查。

开启 `profile_stages=True` 或 CLI `--profile-stages` 后，剖析器在当前
进程已经初始化 CUDA 时，在阶段前后同步指定设备，并分别记录等待时间。
CPU 原生阶段也会得到这两项等待记录，因此可以区分原生函数内部耗时和
父进程 CUDA 等待。初始化前的 CPU 阶段不会为了计时建立 CUDA context。
默认模式下，仅返回 GPU 张量的异步函数可能尚未完成计算；GPU 阶段配对
计时应使用同步诊断或该函数已有的同步边界。

`device` 是当前进程可见的逻辑设备，如 `cuda:0`、`cuda:1` 或 `cpu`。
所有剖析同步及 Torch 显存查询都显式使用该目标，不使用无参数的默认 GPU。
设置 `CUDA_VISIBLE_DEVICES` 后，逻辑编号与物理编号可能不同；外部监控
用物理 UUID 对照。剖析器不更改设备可见性、TF32 或 autocast。

## 输入、参数和输出

单例 API 的具名参数如下。输入输出空间沿用标准单 T1 流程：体积为
1 mm conform 网格；表面为 surface RAS、单位 mm；剖析记录没有影像空间。

| 参数 | 默认值与说明 |
|---|---|
| `t1` | 原始单幅 T1 影像路径 |
| `subject_dir` | 不存在或为空的输出目录，禁止覆盖已有结果 |
| `weights_dir`、`assets_dir` | 已校验的声明权重与资产目录 |
| `device` | `cuda:0`；必须对应实际目标设备 |
| `threads` | `4`；Torch intraop 和调用线程的 Numba 掩码，退出恢复原值 |
| `native_bin_dir` | `None`；自动使用当前 Conda 的源码构建程序 |
| `profile_stages` | `False`；是否增加并单列阶段 CUDA 同步 |
| `cuda_allocator_cache` | `auto`；缓存选择见下一节 |

成功返回报告字典，并写入 `subject_dir/fnit-native-free-run.json`。报告
包含逐阶段记录、程序哈希、输出路径、线程与精度状态、完整性和网格检查。
执行完成不自动代表严格复现或整体指标等效。输入或依赖验证失败会抛异常，
可能尚未写出阶段报告；已开始的计算函数失败会保存失败阶段与异常并继续
抛出，不返回成功字典。该剖析包装没有独立 FreeSurfer 等价命令；标准
单 T1 工作流的官方参考命令和输出范围见[阶段说明](CONDA_CPP_STAGES.md)。

`StageProfiler(device=..., synchronize=..., allocator=...)` 的 `allocator`
是 `configure_cuda_allocator` 返回字典。其
`run(name, function, *args, **kwargs)` 原样调用函数并返回函数原结果；
参数和影像语义由该函数规定。每次记录放在 `last_row`，不自动写文件。
计时和 CPU 时间单位为秒，显存单位为字节：

| `last_row` 字段 | 统计范围 |
|---|---|
| `name`、`seconds` | 阶段名称；全程墙钟，包含前后同步、函数及统计开销 |
| `function_seconds` | 函数内部全程，包含该函数的模型加载、传输和文件读写 |
| `cuda_pre_sync_seconds`、`cuda_post_sync_seconds` | 分别记录阶段前、后的目标设备等待 |
| `cuda_synchronization_requested`、`cuda_synchronized` | 是否请求同步、是否至少有一次同步成功完成 |
| `parent_cpu_seconds` | 当前进程消耗的 CPU 时间；多线程时可大于墙钟 |
| `child_cpu_seconds` | 本阶段已回收子进程的 user+system CPU 时间差 |
| `torch_memory_stats_status` | Torch 统计是否可用，见下节 |
| `gpu_peak_allocated_bytes`、`gpu_peak_reserved_bytes` | 仅统计可用时出现，属于当前进程及指定设备 |
| `gpu_peak_scope` | 同步诊断为阶段；默认模式为已有累计峰值区间 |
| `error` | 阶段前处理、函数或阶段后同步失败时的原异常字符串 |
| `torch_memory_stats_error` | 显存统计查询失败的单独异常；此时统计状态为 `failed` |

默认模式不重置 Torch 峰值，记录可能包含 API 调用前的同设备分配；调用方
若自行重置过峰值，累计区间也随之改变。同步模式在阶段前 CUDA 已经激活
且统计可用时重置峰值。CPU 时间属于当前进程及其已回收子进程，不能当作
GPU 内核时间；在同一进程并发执行其他任务会影响这些统计。

阶段前同步或峰值重置失败时，仍记录本次阶段，函数尚未执行则
`function_seconds=0`；不会沿用上一阶段的 `last_row`。前后同步等待
分别保留已经尝试的时间，`cuda_synchronized=True` 只代表至少一次成功，
不代表两个边界都成功。统计查询失败时记录 `failed` 和
`torch_memory_stats_error`，不填零显存、不替换函数返回值或遮盖原异常。
入口继续抛出原阶段错误，调用方可以据日志区分计算失败和统计不可用。

入口另保留阶段函数现成的 `timings_seconds`，以及字典形式的 `seconds`
作为 `substep_seconds`。例如 MNI 的模型/转换/求逆/检查，annotation 的
准备/Gibbs/岛屿清理，可以帮助定位阶段内部的主要耗时。

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    t1="/data/sub-01_T1w.nii.gz",  # 原始 T1，不使用官方中间结果
    subject_dir="/data/fnit-profile/sub-01",  # 新建或为空的重建目录
    weights_dir="/data/fnit-weights",  # 固定清单校验过的模型权重
    assets_dir="/data/fnit-assets",  # 固定清单校验过的模板和图谱
    device="cuda:0",  # 当前可见设备中的目标 GPU
    threads=4,  # Torch intraop、Numba 工作掩码；不等于进程总线程数
    native_bin_dir=None,  # 当前 Conda 的独立源码构建程序
    profile_stages=True,  # 同步诊断；生产默认为 False
    cuda_allocator_cache="auto",  # 保留已有 API allocator，首次调用延续低显存策略
)
print(report["timing"])  # 显示完整计时范围
print(report["stages"])  # 每阶段函数、同步、CPU 与可用显存记录
```

单个已有函数也可以直接剖析，包含函数内部的 nibabel 读写：

```python
from fnit.recon_all.mri_mask_gpu import mask_volume
from fnit.recon_all.profiling import StageProfiler, configure_cuda_allocator

allocator_report = configure_cuda_allocator(
    device="cuda:0",  # 显式目标设备；不创建 GPU 张量
    policy="auto",  # 首次调用与已初始化 API 的区别见下节
)
profiler = StageProfiler(
    device="cuda:0",  # 同步与统计使用相同 GPU
    synchronize=True,  # 显式请求诊断同步
    allocator=allocator_report,  # 记录缓存是否已知以及统计是否可用
)
profiler.run(
    name="brainmask",  # 写入 last_row 的阶段名称
    function=mask_volume,  # 复用现有 PyTorch 函数，不换算法
    input_file="/data/fnit-subject/mri/T1.mgz",  # conform 强度图
    mask_file="/data/fnit-subject/mri/synthstrip.mgz",  # 同网格二值掩膜
    output_file="/data/fnit-diagnostic/brainmask.mgz",  # 新诊断文件
    threshold=None,  # 原默认非零选择规则
    invert=False,  # 保留掩膜内体素
    outside_value=0,  # 掩膜外填充值
    device="cuda:0",  # 计算和剖析使用同一目标 GPU
)
print(profiler.last_row)
```

## CUDA allocator 和统计可用性

`configure_cuda_allocator(device, policy="auto")` 只在 CUDA 尚未初始化时
选择缓存环境，不分配张量。报告包含请求、进入时的初始化状态、选择前后
`PYTORCH_NO_CUDA_MEMORY_CACHING` 及能够确认的生效状态。

| 场景 | 行为与报告 |
|---|---|
| CUDA 未初始化，`auto` | 保留已有环境；环境缺省时设置关闭缓存，延续原低显存策略 |
| CUDA 未初始化，变量值为 `"1"`、`"0"` 或 `""` | 对 Torch 2.5.1 native allocator 均是变量存在，报告关闭缓存且 Torch 统计不可用 |
| CUDA 未初始化，`disabled` | 设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` |
| CUDA 未初始化，`enabled` | 移除关闭缓存变量；可以得到缓存分配器的 Torch 统计 |
| CUDA 已初始化，`auto` | 保留调用方实际 allocator；报告 `preserved_preinitialized_unknown` |
| CUDA 已初始化，`enabled/disabled` | 抛 `ValueError`，要求在初始化前选择 |
| CPU | `not_applicable`，不设置 CUDA 环境 |

已经初始化后再查看环境变量不能确认 allocator 已改变。验证包装器
`run_initialized_cuda_api.py` 会在创建四字节保留张量前选择显式缓存策略，
随后以 `auto` 调用 API；其侧车记录初始化前的选择和目标 UUID。
生产 Talairach 子进程另有自己的初始化：它复制父环境后移除关闭缓存变量。
父进程关闭缓存不能据此代表该子进程也关闭缓存。

已知关闭缓存或 CUDA 未激活时，`torch_memory_stats_status="unavailable"`，
不填入假的零峰值。已初始化且策略未知、又未观察到有效统计时为
`unavailable_or_allocator_not_observed`。allocated/reserved 的数值 0
不能当作整个进程没有 GPU 占用的证明。即使统计可用，它们也不包含
所有 CUDA context、库工作区或其他进程的显存。

厚度函数的 KD-tree 半径查询使用 `min(4, torch.get_num_threads())`，
报告实际 `kdtree_workers`；线程 1/4 的候选与可达性检查逐值一致。
这一修正在提交 `1b8c36d` 的整例快照之后完成，整例使用预算 4，
查询仍为 4 个工作线程。不同源码版本的阶段报告分别保留，不把新源码
标为已完成该整例；真实网格回归见[厚度说明](SURFACE_THICKNESS.md)。

顶层 Torch 峰值还可能取父进程与 Talairach 子进程各自峰值的最大值，
这不是父子同时合计占用；不能把两个不同时间的峰值相加。
进程合计与 20 GB 预算要使用下一节的同次查询记录，并展示采样缺口。
预算为 **20,000,000,000 字节**，约 18.626 GiB；GB 除以 10⁹，
GiB 除以 2³⁰，不以两种单位的相同数字替代预算。

## API、整例与外部命令的计时范围

`total_seconds` 从公开 API 进入起，包含线程预算设置/恢复、输入和资产
校验、模型加载、数据传输、计算、阶段输出与检查。它包含内部报告写出，
但公开入口最后一次报告写出、CLI 启动、模块导入和最终 JSON 输出位于
该计时边界之外。`timing.pipeline_seconds` 从内部依赖校验结束开始，
不包含公开入口的线程预算设置/恢复；`validation_seconds` 单列校验，
`thread_setup_and_restore_seconds` 单列公开包装与内部计时的差值。
该差值包含线程设置、恢复以及内部末尾报告写出和返回开销，
`thread_setup_and_restore_scope` 明确其范围，不将它解释为纯线程操作时间。
每阶段秒数相加通常小于公开 API 全程，不能将差值自动视为 GPU 等待。

公开入口在阶段异常后也保存本次线程恢复状态和 API 全程；恢复失败时
将本次已有 `complete` 改为 `failed`。阶段与恢复同时失败时继续抛原阶段
异常，报告另记 `thread_budget_restoration_error`。失败元数据读写错误不
遮盖原异常；输入校验拒绝非空目录时保留以前报告的字节。
这项后续修复通过八项本地模拟调度测试，未重跑算法整例；上传至 headcw
的自动审批拒绝了源码上传，因此尚无该补丁的远端环境复测。

`run_monitored.py` 的 `command_wall_seconds` 从命令创建之前计到命令退出，
包含 Python/CLI 启动、导入、加载、读写和结果输出。它在命令退出后、等待
监控线程结束之前取值，因此监控收尾等待不被直接加入被测命令墙钟。
已初始化 API 包装器另有 `seconds_including_context_and_run`，计入其
创建 CUDA context、保留张量和前后同步；仍须外部命令计时覆盖完整启动。

benchmark 包装器本身没有官方等价命令。候选和官方参考应分别使用相同
外层计时、硬件、线程、输入和监控方式；只加速某个局部函数不能宣布
整例已提速。报告必须绑定实际代码、程序、输入、权重和资产版本/哈希。

## 父子进程合计显存监控

`validation/recon_all/python_gpu_port/run_monitored.py` 接收：

| 参数 | 默认值和说明 |
|---|---|
| `--gpu-uuid` | 必需，物理目标 GPU UUID |
| `--output` | 必需，新诊断目录，已存在时拒绝覆盖 |
| `--interval` | `2.0` s，每轮查询后的等待间隔 |
| `--query-timeout` | `5.0` s，单次 `nvidia-smi` 查询的超时 |
| `--` 后的命令 | 完整候选命令；继承显式设备、线程和 allocator 环境 |

输出 `command.log`、`gpu_samples.csv` 和 `monitor.json`，保留退出码、
命令墙钟、请求间隔、实际最大采样间隔、失败查询数量及采样峰值。
CSV 的 `parent_bytes/children_bytes/total_process_bytes` 使用同一次
compute-apps 查询中目标 UUID 上的当前父子进程占用；额外 whole-GPU
查询有自己的耗时，两次查询不是同时进行。监控器不选择 CUDA 设备、
不改变精度和算法，失败查询记录缺失状态，不填零。

实际采样间隔还包含查询与遍历开销，不能把 `--interval 2` 宣称为连续
两秒的完整峰值观测。`peak_sampled_process_bytes` 是成功样本的最大值，
`continuous_peak_verified=False`；缺少有效样本时为 null。被测命令失败
时返回其退出码并保留日志；不是重建完成证明，也不是独立部署验收。

```bash
# 将该变量替换为已核对的实际物理 UUID；可见设备中逻辑编号为 cuda:0。
FNIT_PROFILE_GPU_UUID='GPU-替换为实际UUID'
export CUDA_VISIBLE_DEVICES="$FNIT_PROFILE_GPU_UUID"
fnit_profile_command=(
  fnit-recon-all
  /data/sub-01_T1w.nii.gz  # 原始 T1
  /data/fnit-profile/sub-01  # 新建或为空的重建目录
  --weights-dir /data/fnit-weights  # 已校验权重
  --assets-dir /data/fnit-assets  # 已校验资产
  --device cuda:0  # 映射到上面的 UUID
  --threads 4  # 当前被试的 CPU 工作预算
  --profile-stages  # 单列 CUDA 等待与 CPU 时间
  --cuda-allocator-cache auto  # 首次调用默认，或保留初始化 API 状态
)
python validation/recon_all/python_gpu_port/run_monitored.py \
  --gpu-uuid "$FNIT_PROFILE_GPU_UUID" \
  --output /data/fnit-profile/monitor-sub-01 \
  --interval 2 \
  --query-timeout 5 \
  -- "${fnit_profile_command[@]}"
```

## 验证范围

接口回归位于 [test_profiling.py](../../tests/recon_all/test_profiling.py)，
检查默认不增加同步、同步显式使用目标 GPU、失败函数记录、初始化 API
保留 allocator、初始化前选择策略、`1/0/空字符串` 的 presence 判断，
以及阶段前同步失败与统计失败时保留当前阶段和原异常。
线程状态恢复测试及真实 CPU API
检查见[线程预算](THREAD_BUDGET.md)。这些不替代 GPU 整例实测。

当前已有[旧有效 TF32 的真实 SynthSeg 诊断](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/old_effective_tf32/actual-forward.json)
及其[外部监控](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/old_effective_tf32_monitor/monitor.json)：
阶段读写全程 32.033 s，命令墙钟 46.721 s，采样峰值 19,411,238,912 字节，
22 次样本、无查询失败、最大采样间隔 2.266 s；缓存关闭时 Torch 统计明确
为 unavailable。这是冻结同输入阶段数据，不是当前缓冲优化或整例提速结果。
当前版本的修复后阶段与整例仍须绑定最终源码和报告分别补充。

## 前向实际后端记录

`record_network_forward(module, inputs, records, **metadata)` 接收已经构造的
PyTorch 模块、实际前向输入 Tensor、可追加的记录列表，以及 JSON 兼容的可选元数据。
前三个参数没有默认值，metadata 默认空；函数返回 None，只追加一条字典。
图像坐标与单位由输入张量决定，本函数不采样图像。记录设备、输入/模型 dtype、
matmul/cuDNN TF32、autocast，以及 cuDNN 的 enabled、benchmark、deterministic。
缺少模块/张量属性、列表不能追加时抛出原异常；不加载模型、不改策略、不同步 CUDA。

```python
import torch
from fnit.recon_all.profiling import record_network_forward

network = torch.nn.Conv3d(
    in_channels=1,  # 已构造模型的输入通道数；此小模型仅演示记录接口
    out_channels=1,  # 模型输出通道数
    kernel_size=1,  # 卷积核边长
).to(device="cuda:0", dtype=torch.float32).eval()
input_tensor = torch.zeros(
    size=(1, 1, 16, 16, 16),  # batch、channel、三个体素轴；例子不代表真实benchmark
    dtype=torch.float32,  # 不使用FP16或BF16
    device="cuda:0",  # 实际目标设备
)
forward_records = []  # 调用者保存，内部没有全局缓存
record_network_forward(
    module=network,  # 真实流程应传已经加载权重的网络
    inputs=input_tensor,  # 传真正将要前向的张量
    records=forward_records,  # 追加JSON兼容字典
    model="example_conv3d",  # 可选元数据：辨认模型
)
print(forward_records[0])
```

该记录函数属于重建内部步骤，没有独立CLI或官方等价命令。
新增后端字段在 `db479a0` 只改变日志；97项相关测试通过，未重跑该日志补丁的整例。
两例真实整例实际运行 `ff372d7`，受 main 公用采样器更新影响的注册阶段在
`ece5e23` 用两例相同输入回归，矩阵及warp逐位相同；
[源码与报告绑定](../../validation/recon_all/optimizations/20261001_serial/whole/integrated_main/source_after_main_merge.json)
和[完整耗时、显存及精度](../../validation/recon_all/optimizations/20261001_serial/FINAL_RESULTS.md)
分别保留测试范围。示例张量不充当真实数据验证。
