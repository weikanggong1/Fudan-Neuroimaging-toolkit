# recon-all 的进程显存记录

## 1. 功能与范围

`ProcessTreeDeviceSampler` 记录指定 GPU 上父进程及存活子进程在一次快照中的显存，并单独记录整卡占用。默认复用主页 Conda 已声明的 `nvidia-ml-py`，通过 Python NVML 查询；缺包或查询失败时保留原因并回退 `nvidia-smi`。它不相加不同时间的进程峰值，不改变影像算法、精度或 CUDA allocator。

容器中的 `nvidia-smi` 可能返回宿主机 PID，而 `/proc` 只能看到容器 PID。本轮在 A100 实测发现这种情况：GPU 有非零占用，原采样器却把无法对应的 PID 全部当成外部进程，将自身占用写成 0。修正后，不能确定归属的值为 `null`，状态为 `ownership_unresolved`；整卡和全部计算进程的记录仍保留。

## 2. Python 调用、输入和输出

```python
import os
from fnit.recon_all.profiling import ProcessTreeDeviceSampler

memory_sampler = ProcessTreeDeviceSampler(
    device="cuda:0",          # 当前进程可见的逻辑 GPU，尊重 CUDA_VISIBLE_DEVICES
    parent_pid=os.getpid(),    # 需要跟踪的父进程 PID，子进程通过 /proc 自动发现
    interval=0.5,             # 计划采样间隔，单位为秒
    query_backend="auto",    # 默认 NVML，错误时记录原因并回退命令查询
)
try:
    memory_sampler.sample_if_due(force=True)  # 立即采样，不提前创建 CUDA context
finally:
    memory_sampler.close()                  # 释放本实例 NVML 引用，原报告保留
memory_report = memory_sampler.report()      # 返回可保存为 JSON 的字典
```

构造参数为 `device`、`parent_pid`、`interval`、`query_backend` 和 `trusted_host_pid_map`。`query_backend` 可选 `auto`、`nvml`、`nvidia-smi`：显式 `nvml` 失败仅保存错误，不静默回退；非法名称抛 `ValueError`。`sample_if_due(force=False)` 通常按间隔执行；CPU 设备返回 `not_applicable`，不导入 NVML 或调用 NVIDIA 工具。`add_worker(pid)` 记录额外 worker 身份，实际显存仍按父进程的存活子树计算。`close()` 可重复调用，之后的采样会抛 `RuntimeError`；旧调用方遗忘关闭时，NVML 对象释放也会平衡其初始化引用。

`trusted_host_pid_map=None` 默认不声明 driver 到本地 PID 的映射。可选字典以 driver PID 整数为键，每项必须含 `local_pid`（可见 PID）、`starttime_ticks`（Linux stat 启动 tick）、`pid_namespace`（如 `pid:[…]`）和 `source`（可信宿主来源说明）；可附整数列表 `nspid_chain`。调用者必须先由获授权的宿主来源取得映射，不能以同卡空闲或 GPU 内存反推。采样器复制参数，并在每次查询后核对本地启动 tick、namespace 及提供的链；退出、PID 复用、namespace/链改变则拒绝该映射。非法参数在查询前报 `ValueError`。

`NSpid` 的第一项属于当前 proc 挂载对应的 PID 命名空间，之后是逐层内部编号；当前挂载可能不显示宿主层。链须从该 proc 视图的 `Pid` 开始，非空且各项为正整数；矛盾元数据拒绝，旧内核缺字段保持允许。可信参数中的链须从 `local_pid` 开始。可见链仅证明其所列层级，不能自行确定 driver namespace；显式宿主映射可以绑定一个不在可见链中的 driver PID。

内部 `NvmlProcessQuery()` 没有参数：`devices()` 返回物理索引字符串到 UUID 的字典；`snapshot(uuid)` 返回 `(进程列表, 整卡字节数或 None, 整卡查询错误或 None)`，进程为 `{"pid": int, "bytes": int}`。进程显存缺失、负数或 unavailable 哨兵会抛异常，不能转换为零。它不做 PID 归属判断，不读写影像，也不调用 FreeSurfer 程序。

输出字段如下，全部显存单位为字节：

| 字段 | 含义 |
|---|---|
| `target_gpu_uuid` | 实际物理卡 UUID；已初始化 API 优先核对 CUDA 枚举 |
| `samples` | 每次快照、进程列表、归属状态及查询耗时 |
| `samples[*].pid_mapping` | 本地身份、可信映射及其来源、拒绝原因和缺少证据的原因；不读 cmdline |
| `peak_tree_total_bytes` | 可确认的父子进程同期合计最大值；任一次归属不明时为 `null` |
| `observed_resolved_tree_peak_bytes` | 仅在归属明确的快照中观测到的最大值，不能代表不明时段 |
| `peak_target_compute_process_sum_bytes` | 目标卡全部计算进程的同期合计最大值，包含其他任务 |
| `peak_target_device_used_bytes` | 目标卡整体占用的采样最大值，包含驱动及其他任务 |
| `sampling_interval_seconds` | 计划间隔，不代表实际采样频率 |
| `max_observed_interval_seconds` | 实际相邻快照的最大间隔 |
| `failed_samples` | NVIDIA 查询、UUID 或数据解析失败记录 |
| `query_backend_requested` / `query_backend` | 请求后端及实际最后使用的后端 |
| `query_backend_errors` | 自动回退的初始化、查询或关闭错误，与失败采样分开 |
| `query_closed` | 是否已显式释放本采样器的 NVML 引用 |
| `device_memory_query_sources` | 整卡原始读数来源，区分 NVML v1 字节与 CLI MiB 转换 |

**成熟采样子函数修复（2026-10-11）：** 旧逻辑将本地 PID 自动当成相同编号的 driver PID，并用本地 `/proc/<driver_pid>` 是否存在区分外部进程；不同 namespace 的编号碰撞可能产生虚假归属。当前没有可信映射时，即使数字相同也保持未知。`NSpid` 缺失、单值或没有 driver namespace 锚点的多级链均不能自行证明宿主身份；现代内核的可信链可通过上述显式映射使用，不需要停用 NVML。

无法查询显存、MIG 映射不受支持或 PID 归属不明时，保留原因，不静默记零。连续执行的进程查询和整卡查询带有时间差；进程合计来自同一次进程查询，整卡值单独解释。整卡值是采样时设备占用的上界观察，不能视为精确 FNIT 峰值，也不能保证采样间隙始终低于 20 GB。无归属证据的 driver 行归 `unresolved_processes`，不能仅凭本地 proc 可见性写入 `external_processes`。仅当计算进程列表及整卡查询都明确为零时保留观测零；这不制造 PID 映射。

PyTorch 的 `allocated/reserved` 仍由阶段报告另记。关闭 allocator 缓存时两者可能为零，这不能证明 CUDA context、模型或其他库没有占用显存。20 GB 预算为 **20,000,000,000 字节**。

整卡原始读数不可跨后端直接相等比较。本轮驱动现场检查发现，空闲卡的 NVML v1 `used` 为 928,710,656 字节，而 v2 同次序查询的 `used` 为 3,604,480、`reserved` 为 925,106,176 字节；CLI 分别显示 3 MiB used、882 MiB reserved。报告保留原始值及 API 来源，不把驱动保留量当作本任务张量，也不无条件减去固定常数。父子进程合计仍来自一次计算进程列表；这与整卡读数是不同字段。

## 3. 命令行调用

采样器属于整例测量内部步骤。下列独立工具只测试查询开销和 API 状态，不运行 T1，也不代替影像 benchmark：

```bash
python tools/benchmark_recon_process_memory.py \
  --output-path /results/new_memory_query.json \
  --device cuda:0 \
  --repeats 12 \
  --interval 0.1 \
  --held-bytes 134217728 \
  --code-version ACTUAL_TESTED_COMMIT
```

`output-path` 为新 JSON；`device` 为目标逻辑卡；`repeats` 为每个 ABBA 组的查询次数，至少 2；`interval` 为计划间隔秒；`held-bytes` 为保持存活的 uint8 缓冲字节数，至少 2；`code-version` 记录实际提交，模块和工具另记 SHA-256。GPU 缓冲创建后只同步目标卡，不改变父进程当前卡、精度或已有张量。失败保存原始错误并退出非零；导入时间由外层进程墙钟另计。

原始 T1 整例包装仍支持以下具名参数；路径由用户提供：

```bash
python tools/benchmark_recon_torch_end_to_end.py \
  --t1 /data/public_T1w.nii.gz \
  --output-root /results/new_empty_run \
  --weights-dir /resources/weights \
  --assets-dir /resources/recon_assets \
  --native-bin-dir /environment/native/bin \
  --device cuda:0 \
  --threads 4 \
  --hemisphere-workers 2 \
  --defects-backend native \
  --wm-edit-backend native \
  --sphere-normals-backend numba \
  --native-optimizations auto \
  --code-version ACTUAL_TESTED_COMMIT
```

输出目录必须不存在；失败保留 JSON 和日志，返回非零，不补跑为“连续整例”。上例测试当前默认算法组合，不能代表其他实验后端。

## 4. 对应原软件调用

这是测量辅助函数，没有对应的 FreeSurfer 算法命令。NVML 路径调用 `nvmlDeviceGetComputeRunningProcesses` 和 `nvmlDeviceGetMemoryInfo(v1)`；命令路径调用 `nvidia-smi --query-compute-apps` 和 `--query-gpu`。软件版本、来源及查询范围由报告保存。官方 recon-all 的耗时必须在独立参考路径实测。

## 5. 本轮验证

2026-10-11 当前版本的碰撞与可信映射、PID 复用、已有 UUID remap、NVML 释放和 allocator 统计合同另见 [PID 命名空间修复](../../validation/recon_all/optimizations/20261011_gpu_pid_namespace/README.md)。真实 raw GA v1 公开记录仍为父子树峰值 `null`、目标卡计算进程合计峰值 8,212,447,232 字节、整卡峰值 8,227,127,296 字节。现场内核 `3.10.0-1160.el7.x86_64` 无 `NSpid`，未发现可读完整宿主 proc；新代码没有为该环境伪造映射。此修复只改测量归属，未运行新的 MRI、未修改冻结 raw GA v1/v2。

真实 A100 测试发现约 553 MiB 的非零目标卡占用，却没有可对应的容器 PID；旧值 0 不能作为显存结论。修复的六种情形及原 UUID 契约在同一安装副本 **7/7 通过**：不可见宿主 PID、明确 namespace 别名、已知外部负载、非空整卡但空进程查询、确实空卡，以及宿主 PID 恰好与无关容器 PID 相同。最后一种情形不能仅凭 PID 数字相同就判定为外部负载。

2026-10-10 新增后端回归：设备选择、初始化/查询错误回退、unknown PID、不可用字节、引用释放及已有半球调度共 **64 项通过**。实测范围为同一 A100、CPU 亲和性 96–99、4 线程预算，持有 128 MiB GPU 缓冲的 CLI 及已有 CUDA 的 Python API；每种调用采用同进程内 CLI/NVML/NVML/CLI 顺序。完整收据与启动失败见 [查询报告](../../validation/recon_all/optimizations/20261010_nvml_process_memory/README.md)。

这些契约和缓冲查询不是 MRI benchmark。旧快照保留原始记录；读取旧报告时，只要无法确定进程归属，就标记未知。当前不能据此宣布 20 GB、600 秒整例目标或 MRI 算法加速通过。

## 6. 更新记录

- 2026-10-09：修复容器 PID 归属不明时误记零，增加独立整卡占用及全部计算进程合计，保留实际查询间隔。
- 2026-10-10：复用 Python NVML，减少每次启动命令的查询开销；增加明确后端、自动回退原因、原始整卡来源和引用释放。主页依赖已声明，无新 MRI 依赖。
- 2026-10-11：修复 PID 数字相等的虚假归属风险，增加显式可信映射、持续身份核对及失败证据；影像算法、CUDA 同步/allocator/精度和旧报告不变。
- 已运行的冻结源码和报告不被原地修改；后续候选使用新的源码版本和测量哈希。

## 7. 参考资料

- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA NVML 设备查询 API](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
- [NVIDIA Python NVML 固定版本及源码发行包](https://pypi.org/project/nvidia-ml-py/13.580.82/)
- [Linux proc_pid_status](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
- [PyTorch CUDA 显存管理](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
