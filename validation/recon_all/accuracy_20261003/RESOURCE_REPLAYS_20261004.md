# 四例 CUDA OOM 的冻结版本重跑

## 功能和范围

本轮重跑使用原始 T1、原冻结代码、相同 H100 UUID、4 个总线程和原 CLI／已初始化 CUDA 的 Python API 调用方式。只修改输出和诊断目录；不改算法、精度、分块、缓存策略或验收标准。原失败目录保留。本工具属于 benchmark 调度，不是 recon-all 生产函数，没有对应的 FreeSurfer 内部命令。

`verify_replay_bindings.py` 核对输入、原配置和 launch 回执、冻结源码归档、权重、资产、独立 Conda 程序及 Python 包版本。`resource_admission.py` 在获得已有 benchmark 共享锁之后再次读取目标 GPU 的空闲显存，满足准入条件才启动原监控器和整例执行脚本。`run_resource_replay_queue.py` 顺序执行四例，并分开记录队列结束、每例退出和整例完成回执。

准入要求为 **20,000,000,000 字节空闲显存**，用于此次资源排查。它不保证运行峰值低于 20 GB，也不能约束不遵守共享锁的其他 GPU 使用者。显存的原始单位是 `nvidia-smi` 的 MiB，乘以 1,048,576 转成字节。20 GB 与 20 GiB 不混用。

## 待重跑的原失败

| 原始 T1 | 原调用方式 | 原失败位置 | 此次目的 |
|---|---|---|---|
| ds000030_sub-10189 | initialized_cuda_api | RH 子进程首次创建 float32 标量 | 原采样仍有整卡余量，不能归因于整卡满载；测试是否重现 |
| ds000114_sub-06 | cli | brain_second_normalize | 原故障窗口整卡接近满载，检查资源恢复后的行为 |
| ds000114_sub-07 | cli | input_talairach / SynthStrip conv3d | 同上 |
| ds000114_sub-08 | cli | input_talairach / SynthStrip conv3d | 同上 |

冻结生产代码为 `816e5610417a4c587caf321049438a9554139016`，归档 SHA-256 为 `0107e059153ff46ace5ab477539c232810f3309db0d569c861a02b6c69b460e7`。目标 GPU 为 `GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e`。双侧并行保持原来的 2 个子进程、各 2 个线程。TF32 和已有 FP32 例外依照冻结版本执行，没有启用 FP16／BF16。

## 输入、输出及参数

### 资源绑定核对

Python 函数 `verify(cohort_root=..., expected=..., report_path=...)`：

- `cohort_root: pathlib.Path`：既有十例实验目录，包含原配置、manifest、冻结源码及原 launch 回执。诊断工具读取既有记录，不把官方中间结果交给生产流程。
- `expected: dict`：原 `resources_verified_816e5610.json` 的完整 JSON；权重、资产和程序逐文件给出 SHA-256 与字节数。
- `report_path: pathlib.Path`：此前不存在的新 JSON 路径。
- 返回 `dict`：`status`、`mismatches`、`cases`、`resources`、源码文件数、运行环境与 GPU 快照。失败回执和异常处理见实际脚本；调用方必须检查 `status`；资源缺失、校验/查询异常写 `failed` JSON，报告已存在时抛 `FileExistsError`，不覆盖旧回执。

```bash
# 原十例实验目录；复用原影像、权重和环境。
FNIT_COHORT_ROOT=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003
# 新工具固定目录，与生产冻结源码分开。
FNIT_REPLAY_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/resource_replay_tools_v1
# 既有 Conda 解释器，保持原 prefix。
FNIT_REPLAY_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
"$FNIT_REPLAY_PYTHON" "$FNIT_REPLAY_TOOLS/verify_replay_bindings.py" \
  --cohort-root "$FNIT_COHORT_ROOT" \
  --expected "$FNIT_REPLAY_TOOLS/resources_verified_816e5610.json" \
  --report /tmp/fnit-new-bindings-report.json
```

### 单例资源准入

完整参数见 [RESOURCE_ADMISSION.md](RESOURCE_ADMISSION.md)。`--config` 为原配置 JSON，`--retry-root` 为新空目录，`--report` 为新准入报告，`--lock` 为 `/tmp/fnit-shared-benchmark.lock`；`--poll-seconds` 为等待轮询间隔，`--query-timeout` 为单次 GPU 查询超时，`--maximum-wait-seconds` 为最大等待秒数。只读检查使用 `--preflight-only`。没有 GPU 查询权限、输入哈希不符、输出目录已存在或等待超时时停止；不得覆盖旧结果。监督阶段每秒检查本次进程树，信号取消时处理跨 session 的半球子进程，确认没有本次活跃计算后才释放锁。

### 四例顺序队列

```bash
# plan 中固定解释器、包装器 SHA-256、共享锁及四例新输出路径。
"$FNIT_REPLAY_PYTHON" "$FNIT_REPLAY_TOOLS/run_resource_replay_queue.py" \
  --plan "$FNIT_REPLAY_TOOLS/replay_plan.json" \
  --report /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/baseline_resource_replays_v1/queue_status.json
```

- `--plan: JSON 文件路径`：字段 `python`、`admission_script`、`admission_script_sha256`、`shared_lock`、`cases`。每个 case 提供 `id`、`original_config`、`retry_root` 和 `admission_report`。
- `--report: 新 JSON 文件路径`：记录包装器 PID、主机、脚本和计划哈希、启动/结束 UTC 时间、逐例命令、退出码、准入报告及 completion 回执。队列结束不自动表示所有例验收通过。
- 影像保持原 NIfTI 输入网格与毫米坐标约定；输出沿用冻结 recon-all 的 conformed MRI 和 surface RAS 定义，包装器不重采样或改写影像。

每例产物为 `attempt_01/subject/`、`attempt_01/diagnostics/`、`attempt_01/monitor/`，包括原 profile、进程树显存采样、命令日志与完成/失败回执。等待时间单独记录，不计入被监控整例算法墙钟；原整例时间包含校验、加载、传输和文件读写。

## 实测与结论边界

2026-10-04 04:32 UTC 的资源核对通过：四例输入/配置与原启动回执相符；源码归档内 1,212 个常规文件一致；11 个权重、102 个资产和 15 个独立程序的 SHA-256 与字节数一致；PyTorch 2.5.1 / CUDA 11.8 / cuDNN 90100 等包版本相同。报告为 [replay_bindings_verified_20261004.json](runtime/replay_bindings_verified_20261004.json)。

第三例原故障窗口的整卡采样约 42–43 GB，本次进程树约 2 GB；后面三例的同期整卡采样接近容量。原始采样不能排除两次采样之间的瞬时峰值，也没有证明第三例的初始化 OOM 原因。详见 [third_bootstrap_memory_audit_20261004.json](runtime/third_bootstrap_memory_audit_20261004.json) 和原故障报告。本次重跑完成前不报告整例成功、提速或错误已修复。13:02现场首例左右均生成新的orig/inflated网格，必经的RH首次CUDA分配和同步已通过，原bootstrap OOM本次未重现；整例和另三例尚未完成。证据见 [resource_replay_rh_bootstrap_passed_20261004.json](runtime/resource_replay_rh_bootstrap_passed_20261004.json)。

H100 上仍有其他计算任务，当前重跑用于可执行性和 OOM 排查；共享负载下的耗时不能当作独占硬件提速。未完成物理隔离安装验收。生产流程读取原始 T1、自产中间结果及声明资源；官方软件只在独立参考 benchmark 使用。

## 最近更新和参考

- 2026-10-04：新增原资源绑定复核、固定 UUID 显存准入、跨 session 子进程清理和四例顺序重跑；生产算法保持冻结。
- 上一轮十例实验及严格复现/最终指标分开统计见 [ACCURACY_10_T1_20261003.md](../../../docs/recon_all/ACCURACY_10_T1_20261003.md)。
- [FNIT 源码](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)、[FreeSurfer recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)、[FreeSurfer 源码](https://github.com/freesurfer/freesurfer)。生产算法与对应官方阶段命令沿用该冻结版本的说明。
