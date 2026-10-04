# CUDA首次启动诊断

## 功能与边界

本工具将历史 `torch.empty(1, dtype=torch.float32)` OOM 分解为CUDA初始化、首次分配与同步三个边界。原始对照直接运行冻结的 `fnit.recon_all.hemisphere_worker`，仅把其后调用目标换成不读取影像的诊断函数。拆分路径使用另一批全新进程，按同样顺序导入PyTorch、profiling、thread_budget（含Numba）后调用显式初始化。四轮依次比较双进程原始路径、双进程拆分、单进程两路径及仅诊断用的开启缓存路径，共32个新进程。

本工具没有独立FreeSurfer等价命令，属于FNIT进程启动排错。成功只说明此次启动通过；没有真实T1精度或整例性能结论。启动失败与原始影像、分区算法无关的假设仍须证据验证，不能以盲目重试或默认串行代替定位。

## 输入、输出和参数

`cuda_bootstrap_probe.py` 命令行：

- `--output`：此前不存在的新运行目录；写`summary.json`，每轮的`request.json`/`worker.json`或`boundary.json`以及C++栈日志。输出是JSON和文本，没有影像空间。
- `--source`：已冻结源码目录，原worker路径和SHA随报告绑定。
- `--lock`：已有benchmark共享锁，默认`/tmp/fnit-shared-benchmark.lock`。仅等待资源时不持锁；矩阵计算及本次后代进程清理结束前持锁。
- `--rounds`：轮数，默认4，允许1–20；v2每轮8个新进程，runtime边界矩阵每轮10个。
- `--boundary-script`：可选，独立runtime边界脚本的路径；指定后按原worker和四个直接CUDA边界交错对照。普通v2诊断留空。
- `--seed`：runtime矩阵的交错顺序种子，默认20261004，完整顺序写入planned_trials；不修改神经网络随机种子。
- `--minimum-free-bytes`：标量诊断的GPU空闲前提，默认2,000,000,000字节。不修改原T1整例的20GB准入或显存预算；本工具不加载网络模型。
- `--maximum-wait-seconds`：共享锁/资源等待超时，默认900秒；单轮子进程90秒未结束即失败并清理本次进程。
- `--child`：控制器内部参数，`split_disabled`或`split_enabled`，普通调用不用设置。

`split_child(args)` 在每个CUDA边界前后原子写出回执；初始化前只读取/proc、版本字符串及`is_initialized()`，不调用设备属性、显存、同步等会触发CUDA的观测。返回退出码0/1。`host_snapshot()` 返回指定资源变量、主机内存、进程限制及CUDA库路径，不读取完整环境、凭据或许可证。`cuda_bootstrap_target.noop(**kwargs)` 在原worker初始化之后返回设备状态与空闲/总显存，kwargs只用于兼容其调用签名；关闭缓存时PyTorch allocated统计标为不可用。

```bash
# 保持原实际Conda prefix，不切换生产环境。
FNIT_CUDA_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
# 原冻结源码；不修改它，只调用其worker。
FNIT_CUDA_SOURCE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610
# 新工具与新产物；不存在时才启动。
FNIT_CUDA_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/cuda_bootstrap_tools_v2
"$FNIT_CUDA_PYTHON" "$FNIT_CUDA_TOOLS/cuda_bootstrap_probe.py" \
  --source "$FNIT_CUDA_SOURCE" \
  --output /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/cuda_bootstrap_probe_v2 \
  --lock /tmp/fnit-shared-benchmark.lock \
  --rounds 4 \
  --minimum-free-bytes 2000000000 \
  --maximum-wait-seconds 900
```

## 已知实测及失败行为

2026-10-04在gpucw1完成v2独立矩阵，实际工具提交`865aec34ce30dee816ef67e203f35a2266ce8a5b`，生产对照冻结于`816e5610417a4c587caf321049438a9554139016`。共32个fresh-exec子进程，22成功、10失败；结果见[完整机器报告](runtime/cuda_bootstrap_probe_v2_20261004.json)。原 worker、显式 `torch.cuda.init()` 后的首次 empty、缓存开启及单进程路径都曾失败；拆分实验中的 init 调用本身通过。失败集中在前6个trial，之后22个子进程成功，因此存在时间混杂，不能声称某个设置修复了OOM。

所有失败C++栈落在`CUDAStream.cpp::initGlobalStreamState()`，固定PyTorch 2.5.1该位置的受检CUDA调用是`cudaDeviceGetStreamPriorityRange`；分配器尚未调用`cudaMalloc`，也未创建非默认stream池。拆分路径的PyTorch初始化成功，首次empty才失败。矩阵边界的整卡采样一直为69,392,662,528字节使用、15,549,333,504字节空闲；此处为轮次前后采样，不能排除采样间瞬间变化。这个结果证明无需真实影像即可复现，但尚未确定上下文建立失败的底层原因，不宣称已修复。此前真实sub06两个新annotation worker在首标量分配时报OOM，故障窗口整卡采样2.804GB；详见[sub06回执](runtime/sub06_retry_annotation_bootstrap_audit_20261004.json)。CUDA内部错误需C++栈进一步区分。资源/锁超时、冻结源码变化和子进程失败写入报告；矩阵结束与全部子进程成功分别记录。取消只清理本次有PID/starttime身份绑定的进程，不重置GPU或终止其他用户任务。

显式拆分、改变缓存均是独立诊断对照，未进入生产。不启用FP16/BF16，两个子进程各2个工作线程，维持默认TF32。所有依赖是原Conda环境已有PyTorch和标准库，未新增安装依赖。

## 原实现与参考

- [PyTorch 2.5.1 CUDA empty](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/cuda/EmptyTensor.cpp)。
- [PyTorch 2.5.1 CUDA分配器](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp)。
- [CUDA Runtime文档](https://docs.nvidia.com/cuda/cuda-runtime-api/)。

## runtime上下文因果对照

v3保留原worker对照，每轮将其与四个runtime边界模式按固定种子随机交错，两新进程同时启动，各2线程。不同模式必须使用新进程，不能把前一个模式创建的上下文带入下一个。

`cuda_runtime_boundary.py --mode=<模式> --output=<新目录>` 的模式：

| 模式 | PyTorch init之后的首个边界 | 判断范围 |
| --- | --- | --- |
| direct_priority | 直接调用实际libcudart的priorityrange | CUDA runtime自身调用是否失败 |
| context_sync | 显式torch.cuda.synchronize，然后FP32标量 | 提前上下文同步是否改变边界 |
| context_driver | Driver主上下文retain及setcurrent，再priorityrange和标量 | 显式driver上下文建立；只释放自有retain |
| sticky_probe | peek、priorityrange、peek、getlast，记录原始返回码 | 同宿主线程错误槽；清错是诊断干预，无标量恢复 |

全部ctypes句柄从本进程实际映射读取并以RTLD_NOLOAD附着，不另载stub或其他CUDA版本。首干预之前只读/proc，不调用显存、设备属性或NVML；torch.cuda.init本身作为已知边界单独记录。报告含events边界序列、原始返回码、实际库路径与SHA、GPU UUID、线程和主机限制。返回0表示本模式全部检查通过，非零表示错误或清理失败；模式执行完成不代表生产修复。它没有影像输入/输出、坐标空间或独立官方影像命令；全部依赖已在主页Conda中。

```bash
# 使用既有实际环境；不安装另一套CUDA。
FNIT_BOUNDARY_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
# 新的只读冻结源码对照与独立工具目录。
FNIT_BOUNDARY_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/cuda_bootstrap_tools_v3
"$FNIT_BOUNDARY_PYTHON" "$FNIT_BOUNDARY_TOOLS/cuda_bootstrap_probe.py" \
  --source /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610 \
  --output /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/cuda_runtime_matrix_v3 \
  --boundary-script "$FNIT_BOUNDARY_TOOLS/cuda_runtime_boundary.py" \
  --rounds 4 \
  --seed 20261004 \
  --minimum-free-bytes 2000000000 \
  --lock /tmp/fnit-shared-benchmark.lock \
  --maximum-wait-seconds 900
```

## v3实测与带系统调用跟踪的配对

v3于2026-10-04 08:25:58–08:27:17 UTC在同一gpucw1/GPU0完成40个新进程，38成功、2失败。提前同步模式在同一轮双进程报OOM，原worker、直接priority、driver context及sticky诊断各8次通过。这组结果不能说明driver模式已修复，因为原始路径在本轮也全部通过；提前同步已显示失败，不能直接作为生产修复。完整回执见[runtime矩阵](runtime/cuda_runtime_matrix_v3_20261004.json)。

新增`cuda_paired_context_probe.py`对原worker/driver、直接priority/driver两类交替配对。`--source`、`--output`、`--lock`、`--boundary-script`语义同上；`--pairs`默认8，允许8或12；`--seed`默认20261004；`--strace`仅跟踪自己新创建进程的ioctl/mmap/munmap/mlock/munlock/brk，无完整环境或文件IO；`--child-timeout-seconds`默认90。返回0表示实验矩阵执行完毕，子进程成功率单列；返回1表示包装器超时、取消或自身失败。每对保存NUMA buddyinfo、节点meminfo、少量zone字段、proc库绑定、调用返回码和strace文件SHA。跟踪计时包含ptrace开销，不能作为整例性能数据。

并发Popen没有首API屏障，导入耗时和干预耗时仍可能不同；启动间隔写入回执，配对不一致不单独证明因果。取消只清理带PID/starttime身份绑定的本次进程，全部退出后才释放共享锁。没有真实T1输入/输出，没有影像坐标或原软件影像等价命令。

```bash
# 既有Conda解释器；物理GPU和总4线程由工具固定并记录。
FNIT_PAIRED_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
# 使用独立工具目录；不修改冻结生产源码。
FNIT_PAIRED_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/cuda_bootstrap_tools_v4
"$FNIT_PAIRED_PYTHON" "$FNIT_PAIRED_TOOLS/cuda_paired_context_probe.py" \
  --source /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610 \
  --output /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/cuda_paired_context_v4 \
  --pairs 8 \
  --seed 20261004 \
  --strace \
  --minimum-free-bytes 2000000000 \
  --lock /tmp/fnit-shared-benchmark.lock \
  --maximum-wait-seconds 900
```

## v4驱动上下文实测与NUMA对照

v4在2026-10-04 08:37:19–08:38:07 UTC完成8对、16个新进程，14成功、2失败。第5对（从0计数）的driver与直接runtime分支均失败；`cuDevicePrimaryCtxRetain`在08:37:55.564768返回2，`cudaDeviceGetStreamPriorityRange`在08:37:55.568011返回2。该失败已定位到NVIDIA driver创建主CUDA上下文，而非FNIT模型或标量cudaMalloc。相邻对照正常，不能将提前driver初始化当作修复。strace中失败没有用户态mmap/brk/mlock ENOMEM；ioctl syscall返回0而内核reply未解码，NUMA页碎片仍仅是线索。

新增`cuda_numa_worker_probe.py`只用于同硬件诊断，输入为`--policy default|node1`、`--request`原scalar请求、`--report`worker新报告路径、`--policy-report`新NUMA回执路径；不读取影像，拒绝非scalar请求。default保留原策略，node1在导入Torch之前设置BIND-node1并读回核验，两侧均由同wrapper执行冻结worker。只使用当前服务器已现场核验的系统libnuma，不进入FNIT安装或生产依赖。设置失败不启动worker、不回退；finally恢复原主线程策略并核验，已生成辅助线程的继承策略仅在退出时消失。CPU affinity、kernel、页缓存和GPU配置保持不变；所有变化限定本次新进程。

控制器`--numa-wrapper`指定此脚本后改为default_worker/node1_worker对照，其他v4默认行为不变。回执单列策略准入/恢复失败和CUDA worker失败，不能混计；即使node1通过，也须有同时段原路径失败及真实同输入回归，才能考虑独立的部署策略，不直接改生产默认。

```bash
# NUMA仅诊断，不改变GPU精度或原生产函数。
FNIT_NUMA_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
FNIT_NUMA_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/cuda_bootstrap_tools_v5
"$FNIT_NUMA_PYTHON" "$FNIT_NUMA_TOOLS/cuda_paired_context_probe.py" \
  --source /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610 \
  --output /cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/cuda_numa_context_v5 \
  --numa-wrapper "$FNIT_NUMA_TOOLS/cuda_numa_worker_probe.py" \
  --pairs 8 \
  --seed 20261004 \
  --minimum-free-bytes 2000000000 \
  --lock /tmp/fnit-shared-benchmark.lock \
  --maximum-wait-seconds 900
```

## v6固定RM返回状态与生产启动处理

2026-10-04 09:10:54–09:11:34 UTC，gpucw1/GPU0完成16个新标量进程，全部成功。仅附着自有诊断进程的源码编译ioctl工具记录6,420个固定RM头，与strace白名单计数逐项匹配，header复制失败0。非零状态86出现64次，上游535源码定义为`NV_ERR_NOT_SUPPORTED`；它在每个成功进程均出现，不是OOM证据。成功syscall保留的errno17也不表示失败。没有在本轮捕获失败臂，未确认更多底层原因，详见[独立审计](runtime/CUDA_IOCTL_V6_STATUS_AUDIT_20261004.md)、[完整v6报告](runtime/cuda_ioctl_context_v6_20261004.json)。工具不进入生产、不升级驱动、不重置GPU。

已经验证的边界是NVIDIA主CUDA上下文建立可返回OOM，先于FNIT影像算法及cudaMalloc。NUMA页碎片、特定驱动内部资源仍待证明；既有整卡满载失败与这种首次上下文失败分别报告。

生产调度增加实际worker的READY→GO屏障，函数未进入的可信启动OOM可在30秒批预算内重启失败进程。每次原始错误保留；成功READY侧持有上下文，算法每侧最多执行一次。函数、导入或后同步错误均不重试。真实消息`CUDA error: out of memory`已覆盖，普通host OOM不重试；父进程退出或期限到达使READY worker退出。此修改是启动资源处理，并非NVIDIA驱动根因修复。28项CPU回归通过，后续[真实阶段对照](STARTUP_STAGE_QUEUE.md)及原始T1空目录整例各自记录，不能以标量通过替代整例。
