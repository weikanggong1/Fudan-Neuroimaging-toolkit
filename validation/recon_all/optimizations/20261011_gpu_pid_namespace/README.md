# GPU PID 命名空间归属修复

## 1. 功能简介

修复成熟 `ProcessTreeDeviceSampler` 的身份判断：driver PID 与容器 PID 数字相等，并不能证明是同一进程。旧逻辑还把本地 `/proc/<driver_pid>` 存在视为外部进程证据。当前版本移除这两项推断，记录映射依据和失败原因。影像计算、GPU 查询、设备选择、CUDA 同步、allocator 和精度函数保持原语义。

本轮只读现场检查与新代码合同分开保存。现场原代码 SHA 为 `17b12fdc…`；新实现的完整 SHA 见 [源清单](reports/SOURCE_MANIFEST.json)。现有 raw GA v1/v2 源码和产物保持冻结。

## 2. Python 调用、输入与输出

```python
import os
from fnit.recon_all.profiling import ProcessTreeDeviceSampler

memory_sampler = ProcessTreeDeviceSampler(
    device="cuda:0",                  # 逻辑设备，尊重 CUDA_VISIBLE_DEVICES
    parent_pid=os.getpid(),            # 当前父进程，发现其存活子树
    interval=0.5,                     # 计划采样间隔，秒
    query_backend="auto",            # NVML；失败记录原因后回退 nvidia-smi
    trusted_host_pid_map=None,         # 尚无可信宿主映射，不能从显存推测 PID
)
try:
    memory_sampler.sample_if_due(force=True)
finally:
    memory_sampler.close()            # 只释放本采样器的 NVML 引用
memory_report = memory_sampler.report()
```

五个参数的完整说明见 [进程显存 API](../../../../docs/recon_all/GPU_PROCESS_MEMORY.md)。可选 `trusted_host_pid_map` 的每个 driver PID 整数键对应以下身份记录：

| 字段 | 格式与含义 |
|---|---|
| `local_pid` | 正整数，当前可见进程 PID |
| `starttime_ticks` | 非负整数，`/proc/<pid>/stat` 字段22，Linux 启动 tick |
| `pid_namespace` | `pid:[整数]`，本地进程 namespace inode 标识 |
| `source` | 非空字符串，获授权的可信宿主映射来源；不写凭据 |
| `nspid_chain` | 可选正整数列表，由可信来源绑定的完整可见链 |

来源证明由调用者提供。采样器逐次核对本地 starttime、namespace 和提供的链；失效记录在 `rejected_mappings`，不静默换用同编号的新进程。`NSpid` 从当前 proc 挂载所对应的 PID 命名空间开始，随后列出逐层嵌套的内部编号；容器自身挂载的 proc 可能不包含宿主层。它能提供这个可见层级中的编号，不能独自确定 GPU driver 使用哪个命名空间或证明 host PID。缺失、单值和没有可信 driver 锚点的多级链都保持未知。现代系统已有可信映射时可以显式传入并持续核对；本轮 CFFF 没有提供此映射。[Linux man-pages](https://kernel.googlesource.com/pub/scm/docs/man-pages/man-pages/%2B/refs/tags/man-pages-6.11/man/man5/proc_pid_status.5)

存在 `NSpid` 时，链须非空、各项为正整数，第一项与该 proc 视图的 `Pid` 及路径 PID 相同。矛盾元数据按身份不可用拒绝，缺字段本身仍是合法旧内核情况。可信参数中的可选链同样须从 `local_pid` 开始，非法参数在查询前报错；可信 driver PID 可以不出现在此可见链中，其关联仍须由获授权的宿主来源建立。

`samples[*].pid_mapping` 返回 aliases、本地身份、可信记录、拒绝与缺证原因。归属未确定的 driver 行保留为 `unresolved_processes`；父子树峰值为 `null`，计算进程合计与整卡占用分别保留。`external_processes` 保留兼容字段，不能以本地 proc 可见性填充。实际计算列表及整卡查询均明确为零时可以记录观测零。显存单位均为字节；整卡查询和进程查询先后执行，不能相加各自峰值。

## 3. 命令行与复现

采样器无新增 CLI。以下合同不运行影像或创建 CUDA 工作负载；已有 CLI 见 API 专页。

```bash
export PYTHONPATH=src
python -m pytest -q \
  tests/recon_all/test_process_memory_namespace.py \
  tests/recon_all/test_nvml_process_memory.py \
  tests/recon_all/test_profiling.py \
  tests/recon_all/test_hemisphere_parallel.py \
  -k 'namespace or nvml or allocator or short_gpu_uuid or profiler' \
  --junitxml=/results/pid_namespace_contracts.xml

python -m pytest -q \
  tests/recon_all/test_profiling.py::test_production_does_not_synchronize_or_reset \
  tests/recon_all/test_profiling.py::test_profile_targets_requested_device_and_records_function_exception \
  tests/recon_all/test_profiling.py::test_failed_presynchronization_records_current_stage_and_preserves_error \
  --junitxml=/results/pid_namespace_stage_contracts.xml
```

非法 mapping 参数在 GPU 查询前抛 `ValueError`；查询失败仍记录错误；采样器关闭后再次采样抛 `RuntimeError`。同卡空闲不是映射证明。`add_worker` 当前只记录身份元数据，实际范围仍是父进程存活子树；重挂进程需另行确定生命周期范围。

## 4. 对应原软件调用

这是测量函数，没有 FreeSurfer 算法命令。沿用 `nvmlDeviceGetComputeRunningProcesses`、`nvmlDeviceGetMemoryInfo(v1)`，或者 `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory` 和 `--query-gpu=uuid,memory.used`。本修复没有新增程序、Python 包或库依赖。

## 5. 最新验证与真实观察

**新代码：51 项 focused 合同通过。** 首组48项用时35.77秒，包含数字碰撞、缺失/单值/无锚点多级 NSpid、合法可见多级链、矛盾/空/非正链拒绝、可信多级映射、旧内核可信宿主记录、PID 复用与 namespace/链失效、已有设备 remap、NVML关闭和 allocator；补充3项用时12.72秒，核对关闭 allocator 时未知统计、目标卡同步和失败记录。完整测试记录见 [contracts.json](reports/contracts.json)。这些是代码合同，不是 MRI benchmark。

[AST 核对](reports/unchanged_functions.json)确认 StageProfiler、allocator/精度记录、GPU 查询、设备选择、关闭以及子树发现函数与旧版相同，比较排除说明字符串。此检查不替代实际影像运行。

**旧代码真实现场观察：** 内核 `3.10.0-1160.el7.x86_64` 的允许 proc 元数据没有 `NSpid`，未见可读完整宿主 proc；权限拒绝保留。raw GA v1 树峰值为 `null`，目标卡计算进程合计峰值 **8,212,447,232 字节**，整卡峰值 **8,227,127,296 字节**，最大采样间隔 **1.2808552719652653 秒**。v2 已确认的本地树 PID 与一次 driver PID 查询无法映射；该单次值不作为 FNIT 峰值。见 [只读审查公开副本](reports/readonly_audit.public.json)，原私有审查 SHA 保留，公开副本仅删非必要 GPU UUID，其他数值/布尔/null 保真。

整卡值是采样时设备占用上界的观察，含其他任务与驱动，不是精确 FNIT 占用或连续20GB保证。计算进程合计是另一独立观察值。没有新增真实 GPU/MRI 计算、整例速度或质量结论；旧两个 raw run 不应用本次实现。

## 6. 版本与 benchmark 记录

- 2026-10-09：未知宿主/容器归属由误写零改为 `null`；原报告保留。
- 2026-10-10：复用 NVML，保存查询来源、自动回退与引用释放；旧查询实测保持其原版本边界。
- 2026-10-11 只读审查：旧源码 HEAD `5d1cdb5e`、SHA `17b12fdc…`，确认当前环境缺少可用 PID 映射。
- 2026-10-11 本修复：删除未经证明的数字 alias，引入显式可信映射和逐次身份复核；新模块 SHA/合同另绑，不重标上述现场观察。

## 7. 代码与参考资料

- [FNIT profiling 源码](../../../../src/fnit/recon_all/profiling.py)
- [命名空间合同](../../../../tests/recon_all/test_process_memory_namespace.py)
- [进程显存 API 与 NVML/proc 官方资料链接](../../../../docs/recon_all/GPU_PROCESS_MEMORY.md#7-参考资料)
- [原公开 raw GA v1 失败报告](../20261011_python_n4_dependency_route/raw_t1_failure_v1.json)

不读取其他进程 cmdline、私有日志或凭据；未访问不可读宿主目录，未修改生产入口或仍运行的冻结任务。
