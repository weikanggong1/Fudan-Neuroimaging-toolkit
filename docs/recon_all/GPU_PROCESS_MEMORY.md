# recon-all 的进程显存记录

## 1. 功能与范围

`ProcessTreeDeviceSampler` 记录指定 GPU 上父进程及存活子进程在一次快照中的显存，并单独记录整卡占用。它不相加不同时间的进程峰值，不改变影像算法或精度。

容器中的 `nvidia-smi` 可能返回宿主机 PID，而 `/proc` 只能看到容器 PID。本轮在 A100 实测发现这种情况：GPU 有非零占用，原采样器却把无法对应的 PID 全部当成外部进程，将自身占用写成 0。修正后，不能确定归属的值为 `null`，状态为 `ownership_unresolved`；整卡和全部计算进程的记录仍保留。

## 2. Python 调用、输入和输出

```python
import os
from fnit.recon_all.profiling import ProcessTreeDeviceSampler

memory_sampler = ProcessTreeDeviceSampler(
    device="cuda:0",          # 当前进程可见的逻辑 GPU，尊重 CUDA_VISIBLE_DEVICES
    parent_pid=os.getpid(),    # 需要跟踪的父进程 PID，子进程通过 /proc 自动发现
    interval=0.5,             # 计划采样间隔，单位为秒
)
memory_sampler.sample_if_due(force=True)  # 立即采样，不提前创建 CUDA context
memory_report = memory_sampler.report()  # 返回可保存为 JSON 的字典
```

`device`、`parent_pid`、`interval` 是全部构造参数。`sample_if_due(force=False)` 通常按间隔执行；CPU 设备返回 `not_applicable`，不调用 NVIDIA 工具。`add_worker(pid)` 记录额外 worker 身份，实际显存仍按父进程的存活子树计算。

输出字段如下，全部显存单位为字节：

| 字段 | 含义 |
|---|---|
| `target_gpu_uuid` | 实际物理卡 UUID；已初始化 API 优先核对 CUDA 枚举 |
| `samples` | 每次快照、进程列表、归属状态及查询耗时 |
| `peak_tree_total_bytes` | 可确认的父子进程同期合计最大值；任一次归属不明时为 `null` |
| `observed_resolved_tree_peak_bytes` | 仅在归属明确的快照中观测到的最大值，不能代表不明时段 |
| `peak_target_compute_process_sum_bytes` | 目标卡全部计算进程的同期合计最大值，包含其他任务 |
| `peak_target_device_used_bytes` | 目标卡整体占用的采样最大值，包含驱动及其他任务 |
| `sampling_interval_seconds` | 计划间隔，不代表实际采样频率 |
| `max_observed_interval_seconds` | 实际相邻快照的最大间隔 |
| `failed_samples` | NVIDIA 查询、UUID 或数据解析失败记录 |

`NSpid` 可用时按明确的 namespace 别名匹配；缺失时不猜 PID。无法查询显存、MIG 映射不受支持或 PID 归属不明时，保留原因，不静默记零。连续执行的进程查询和整卡查询带有时间差；进程合计来自同一次进程查询，整卡值单独解释。

PyTorch 的 `allocated/reserved` 仍由阶段报告另记。关闭 allocator 缓存时两者可能为零，这不能证明 CUDA context、模型或其他库没有占用显存。20 GB 预算为 **20,000,000,000 字节**。

## 3. 命令行调用

采样器属于整例测量的内部步骤，没有独立 CLI。原始 T1 整例包装支持以下具名参数；路径由用户提供：

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

这是测量辅助函数，没有对应的 FreeSurfer 算法命令。NVIDIA 数据源为 `nvidia-smi --query-compute-apps` 和 `--query-gpu`；执行软件版本和查询范围由报告保存。官方 recon-all 的耗时必须在独立参考路径实测。

## 5. 本轮验证

真实 A100 测试发现约 553 MiB 的非零目标卡占用，却没有可对应的容器 PID；旧值 0 不能作为显存结论。修复的六种情形及原 UUID 契约在同一安装副本 **7/7 通过**：不可见宿主 PID、明确 namespace 别名、已知外部负载、非空整卡但空进程查询、确实空卡，以及宿主 PID 恰好与无关容器 PID 相同。最后一种情形不能仅凭 PID 数字相同就判定为外部负载。

这些契约不替代真实影像 benchmark。旧快照保留原始记录；读取旧报告时，只要无法确定进程归属，就标记未知。更新后的整例显存与墙钟必须以新测量结果为准，当前不能据此宣布 20 GB 或 600 秒整例目标通过。

## 6. 更新记录

- 2026-10-09：修复容器 PID 归属不明时误记零，增加独立整卡占用及全部计算进程合计，保留实际查询间隔。
- 已运行的冻结源码和报告不被原地修改；后续候选使用新的源码版本和测量哈希。

## 7. 参考资料

- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/)
- [Linux proc_pid_status](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
- [PyTorch CUDA 显存管理](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
