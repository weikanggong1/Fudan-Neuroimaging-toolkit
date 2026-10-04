# 双半球真实标量 READY→GO 验证

## 功能、输入与输出

`replay_startup_group.py` 使用实际 `run_hemisphere_group` 和相同 `cuda_bootstrap_target:noop`。worker 只完成既有 FP32 标量 bootstrap、设备检查和无影像 noop；不读取 T1 或官方结果。脚本不取 GPU 锁、不启动外层监测；运行前由外层 runner 固定 GPU UUID、持有锁并监测自有进程。

`--output`：新输出目录。外层日志可已存在，但其中 `subject` 和 `startup_group.json` 必须不存在，避免覆盖旧验证。脚本创建空 mri/surf/label/stats/scripts。

`--initialized-parent`：可选；父进程先做一次 FP32 标量分配与同步，并在组执行期间保持上下文。父 bootstrap 失败直接记录，不重试。

`--startup-wait-seconds`：candidate 显式参数，通常 30；必须有限且非负。baseline 完全省略此参数，调用中不传新关键字。设备固定 `cuda:0`（物理映射由外层固定）、workers=2、threads=4、TF32 开启、allocator cache disabled；没有算法或精度 fallback。

输出 `startup_group.json` 为严格 JSON：实际源码文件 SHA、Torch/CUDA 版本、父初始化状态、请求/实际精度、每侧设备与 callable 结果、完整 group/全部 attempt 原始报告、函数进入判定和总墙钟。`source_manifest.json` 保存实际 FNIT Python 包及 noop 源码 SHA。每次日志/request/report 由实际 scheduler 留在 subject/scripts；baseline 按原 scheduler 记录能力报告。

## 调用

```bash
# 外层负责固定 PYTHONPATH 到被测 baseline 或 candidate 源码，并提供GPU锁/监测。
python replay_startup_group.py --output /path/to/new_baseline_group
python replay_startup_group.py --output /path/to/new_candidate_group --startup-wait-seconds 30
# 单独比较父API已持有CUDA上下文的模式；双方须使用相同选项。
python replay_startup_group.py --output /path/to/new_candidate_initialized --startup-wait-seconds 30 --initialized-parent
```

这不是原软件功能调用，因此没有 FreeSurfer/FSL 对照命令或影像 benchmark。拟交替执行 baseline/candidate 各 10 组，仅观察当时实际启动 OOM 和恢复；必须报告时序及共享 GPU 状态，不能从时间趋势推断底层驱动资源类型已修复。

## 函数进入证据

每侧必须仅返回一个 callable 结果；candidate 汇总所有 attempt 的 `operation_entered`，超过一次即失败。缺失 flag 保持 unknown，不能推断为函数未进入，也不能据此重试。旧 baseline 没有该 flag 时，只能报告每侧一个完成 worker 结果，不能声称直接计数已验证。现有 noop 不含独立调用计数器；脚本保持其源码不变。

总墙钟包括导入、初始化、启动等待和组收尾；失败 JSON 明确 parent bootstrap 与 hemisphere group，parent 和算法均不重试。脚本本身仅做 CPU `--help`、语法及报告判定检查；真实 GPU 验证由外层单独执行。

上游：[PyTorch CUDAStream.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDAStream.cpp)、[NVIDIA primary context API](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)。
