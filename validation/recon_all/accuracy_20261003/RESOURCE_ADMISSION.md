# 冻结整例重跑的显存准入

本工具用于协调者已确认的四例 CUDA 资源失败重跑。它只包装现有 `run_monitored.py` 和 `execute_whole_case.py`，没有修改冻结816源码或生产算法。目标GPU固定为 `GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e`；启动前明确要求 `minimum_free_bytes=20_000_000_000`。所有准入显存报告使用字节，nvidia-smi整MiB读数乘1048576，阈值不改为20GiB。

流程：原配置与原launch核对 → 输入和冻结入口SHA核对 → 当前源/权重/模板/程序逐文件SHA登记 → 查显存 → 尝试共用flock → 锁内再次查显存 → 足够时调用原监测与执行入口。锁忙或显存不足均在锁外等待；等待不计入算法时间。锁只协调使用同一路径的任务，不能防止其他任务在检查后抢占GPU。

## 输入、输出和Python调用

`query_gpu(timeout=5)`返回目标UUID、UTC时间及`total_bytes/used_bytes/free_bytes`；`admitted(sample)`按固定UUID和明确字节阈值判断。`retry_config(original, retry_root)`复制配置，仅改变`output`和`diagnostic_root`。原失败输出、原config、API/CLI模式、参数、输入、权重、程序路径、commit和archive绑定均保留。

`benchmark_tool_paths(config)`选择两项验证驱动，不改生产源码。配置没有 `benchmark_tools` 时保留历史 `code_root/validation/` 路径；新生产冻结树可以用独立工具目录，必须显式声明且只有 `monitor`、`whole_case_driver` 两项，每项含绝对 `.py` 文件 `path` 和冻结 `sha256`。任一文件缺失、SHA 漂移、相对路径、不完整声明或 null 配置均报错。显式工具加入 `inventory(config)`，在准备、等待、取得锁后和启动命令生成时核验。`benchmark_command(config, retry_root)`返回完整字面 argv 列表；目标 GPU、20 GB 准入和固定线程保持原协议，没有 shell 插值或算法替换。

```python
from pathlib import Path
from resource_admission import benchmark_command, inventory

whole_case_configuration = frozen_whole_case_configuration  # 原始 T1、源码、线程和资源均已有 SHA 绑定
whole_case_configuration["benchmark_tools"] = {
    "monitor": {"path": str(frozen_monitor_script_path), "sha256": frozen_monitor_script_sha256},  # 已验证的进程树显存监控
    "whole_case_driver": {"path": str(frozen_whole_case_driver_path), "sha256": frozen_whole_case_driver_sha256},  # 原始 T1 与空目录入口
}
resource_file_sha256 = inventory(config=whole_case_configuration)  # 包含真实输入、资源和两项工具；不读取许可证
execution_arguments = benchmark_command(
    config=whole_case_configuration,  # 参数与冻结配置一致
    retry_root=Path(fresh_attempt_directory),  # 新目录；subject、diagnostics 和 monitor 分别保存
)
```

该接口属于验证调度，文件没有影像坐标和数值单位；资源及显存一律字节、等待及墙钟一律秒。示例变量由实际冻结清单提供，不接受根据目录名推断版本。源树外工具不能成为生产流程的影像输入，原参考结果仍只用于独立对照。

```python
# 调度检查，不调用影像算法。
from resource_admission import query_gpu, admitted
current_gpu_sample = query_gpu(timeout=5)  # 查询超时为5秒
can_start_now = admitted(current_gpu_sample)  # 固定UUID及20,000,000,000字节要求
```

CLI参数：`--report`为新JSON文件，必须不存在；`--preflight-only`只查一次、不取锁、不准备重跑；执行时必须提供`--config`（原冻结config）、`--retry-root`（独立且不存在的新目录）、`--lock`（协调者确认的共享锁）。`--poll-seconds`默认30秒，`--query-timeout`默认5秒，两者必须大于0且不超过60秒；`--maximum-wait-seconds`默认3600秒，控制准备和等待的总超时，0表示只尝试一次。阈值和GPU不可在CLI修改。

输出包括报告、`retry_config.json`、`subject/`、`diagnostics/`及`monitor/`。报告包含原config/launch SHA、当前逐文件资源SHA、原配置绑定、锁前与锁内采样、等待时间、实际执行命令及状态。`complete`仅表示原执行入口退出0；精度等效需另行评估。查询失败返回1，等待超时或只读资源不足返回75，中断返回130。SIGINT/SIGTERM记录`interrupted`；递归快照本工具后代PID、PGID和`/proc/stat`启动时间，并停止祖先继续fork，再补收集后代。逐身份发送TERM/CONT，最多等待10秒后KILL；包括半球worker的新session，不以monitor leader退出作为清理完成。只有全部已归属后代不再活跃计算时才释放锁；Z/X状态不算计算。清理异常或不可中断D状态仍存活时，报告明确写`cleanup_error_lock_held`或`kill_waiting_lock_held`并持续保留锁，每次等待不超过1秒。不会遍历其他任务的argv或环境，不会终止其他任务。不得用已有重跑目录再次执行。

## 具名命令

```bash
FNIT_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
FNIT_ADMISSION=/path/to/reviewed/resource_admission.py
# 只读检查；报告应放在协调者的新runs目录。
"$FNIT_PYTHON" "$FNIT_ADMISSION" --preflight-only \
  --report /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon-resource-retry-20261004/preflight.json

# 正式执行仅由协调者审核上传后进行；下列变量必须指向确认的实际路径。
FNIT_ORIGINAL_CONFIG=/path/to/configs_v1/baseline_ds000114_sub-06.json
FNIT_RETRY_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon-resource-retry-20261004/ds000114_sub-06/attempt-01
FNIT_SHARED_LOCK=/tmp/fnit-shared-benchmark.lock
"$FNIT_PYTHON" "$FNIT_ADMISSION" --config "$FNIT_ORIGINAL_CONFIG" \
  --retry-root "$FNIT_RETRY_ROOT" --lock "$FNIT_SHARED_LOCK" \
  --report /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon-resource-retry-20261004/sub06-admission.json \
  --poll-seconds 30 --query-timeout 5 --maximum-wait-seconds 86400
```

本轮共享锁已由协调者通过原队列和`/proc/locks`现场核实为`/tmp/fnit-shared-benchmark.lock`；旧五会话锁不适用于本轮。资源目录SHA登记排除`.git`、Python/Numba缓存及许可证文件，不读取许可证内容。登记证明准备到启动资源未变；原config/launch未记录的历史权重SHA不能靠本次登记追溯证明，须由协调者原清单另行核对。

## 本版验证和记录

2026-10-04初版：4项聚焦单测通过，含精确十进制阈值、UUID错误、MiB→字节转换、配置/输出隔离，以及取得锁后的资源不足、查询错误和中断均退锁且不启动子进程。这些模拟数字是调度单测，不能称真实影像benchmark。

同日04:18:41 UTC在gpucw1用本工具的`query_gpu`/`admitted`函数通过已认证SSH进行只读实测：总容量85,520,809,984字节，占用33,399,242,752字节，空闲51,542,753,280字节，准入判断true。没有取得benchmark锁，也没有启动GPU计算或整例，读数只代表该时刻，正式执行须重新查询。源码通过SSH标准输入命令执行，没有上传或修改冻结树。

本工具没有独立原软件对应命令，属于 benchmark 调度包装器。当前真实重放和失败分型见[资源重放](RESOURCE_REPLAYS_20261004.md)；8f 启动修复的原始 T1 已完成，入口 2755.779 秒、138/138、父子同刻采样峰 10,951,327,744 字节，阶段、官方指标和脑图见[当前启动验证](../../../docs/recon_all/CUDA_STARTUP_BENCHMARK_20261004.md)。不能把准入成功解释为精度等效。

2026-10-04 后续版：新增源树外两项工具的 SHA 绑定与完整字面 argv，保留旧配置路径；8 项 CPU 调度回归通过（0.721 秒），包含实际子进程清理和既有锁语义。新增工具选项尚未完成新的精度候选原始 T1 整例；其实际运行单独记录。本次没有新增运行依赖，使用 Python 标准库。

参考：[Python flock](https://docs.python.org/3/library/fcntl.html#fcntl.flock)、[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。原FNIT监测与执行代码位于本仓库`validation/recon_all/python_gpu_port/run_monitored.py`和`validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py`。

2026-10-04中断清理修复：保留4项原单测，新增真实本地无GPU进程树测试（monitor→新session worker→grandchild），worker/grandchild忽略TERM触发有界升级KILL；清理期间独立进程始终无法取锁，退出后无活跃后代，另一个不相关session仍运行。另测试PID启动时间不符时不发送信号，共6项通过。gpucw1现场内核为3.10，Python无`pidfd_open`；兼容路径在发送每个信号前核对PID/starttime/PGID，并且测试强制走此路径。新内核可用时采用pidfd绑定身份；旧内核无法提供pidfd的原子身份保障，启动时间核对与STOP递归冻结用于缩小检查/发送之间的系统竞争窗口。该修复解决中断遗漏独立session的调度风险，未执行影像重跑，也不代表CUDA bootstrap或满载OOM已修复。
