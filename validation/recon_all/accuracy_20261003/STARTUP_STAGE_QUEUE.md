# CUDA 启动处理与真实阶段验证队列

## 功能、输入与输出

`run_startup_stage_queue.py`依次调用阶段控制器，每项独立取共享锁、核对输入和源码、采样显存并清理自有进程。计划在运行前固定，失败项保留并继续下一项；用户取消后结束队列。它不运行原始T1整例，没有对应的独立FreeSurfer命令，也不更改影像算法。

输入`--plan`是JSON，字段如下：

- `queue_output`：不存在的新队列记录目录。
- `scope`：此次验证范围。
- `source_provenance`：原算法提交、启动补丁提交、源码差异清单及归档SHA。
- `jobs`：有序列表。每项包含`name`、`role`、`kind`、`config`绝对路径和该JSON的`sha256`。config的全部字段见[阶段控制器](STAGE_BENCHMARK_CONTROLLER.md)。

输出`queue.json`包含每项退出码、控制器报告路径及SHA、墙钟、开始/结束UTC和队列汇总。`queue_complete`表示所有计划项已尝试；`all_jobs_succeeded`另行记录。阶段输出完整性、数值回归和整例完成分别判断，不能用队列完成替代。异常写原始traceback；配置SHA漂移立即失败。取消依据控制器保留的`cancellation_signals`，不会继续下一项。

```bash
# 保持原来安装的Conda解释器，完整路径由已核对的服务器INDEX给出。
FNIT_STAGE_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
# 只读的新验证工具目录。
FNIT_STAGE_TOOLS=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/startup_stage_tools_v1
# 事先冻结的8个标量启动组和4个真实annotation阶段配置。
FNIT_STAGE_PLAN=/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/startup_stage_regression_v1/plan.json
"$FNIT_STAGE_PYTHON" "$FNIT_STAGE_TOOLS/run_startup_stage_queue.py" \
  --plan "$FNIT_STAGE_PLAN"
```

2026-10-04部署的候选只将冻结`816e5610`中的两个半球调度模块换为`5f75ed5c`版本。全部457个`.py`文件核对后，差异恰为这两个模块，其余算法相同。它是启动处理候选，未包含正在开发的其他精度补丁。计划使用同一GPU0 UUID、总4线程、TF32和原分配缓存禁用策略；每项有新目录、独立Numba缓存及0.5秒请求采样。阶段准入要求10,000,000,000字节空闲，不更改原始T1整例的20,000,000,000字节准入。

标量组用于实际CUDA启动验证，不能替代真实benchmark。两例`ds000114_sub-06/sub-07`使用各自FNIT自产检查点，基线与候选核对七个数据输入和八个资产。sub06两侧都预初始化父CUDA，sub07两侧均不预初始化；这不是历史整例父进程驻留状态的完整复现。数值回归要求六套标签、色表、坐标和有序面相同。整例对官方的既有误差及未确认的等效标准仍单独保留。

完整实测待队列完成后写入[CUDA启动处理](../../../docs/recon_all/CUDA_STARTUP_RESOURCE_WAIT.md)和[本轮精度记录](../../../docs/recon_all/ACCURACY_10_T1_20261003.md)。无新增Python或原生依赖。
