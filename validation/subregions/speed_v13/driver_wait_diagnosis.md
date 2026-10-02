# 冻结 v12 对照运行：CUDA 驱动侧等待诊断

日期：2026-10-01。对象为 GPU1 上 PID `5740` 的 `paired_stage_v12_20261001`，不是当前新 speed profile。仅读取日志、`/proc`、进程状态和 NVIDIA 监控，并用已有 `perf` 采样 8 秒；未安装依赖、发送信号、修改运行源码或启动 GPU 计算。

## 观察结果

- 日志在 08:55:17.137（服务器本地时间）进入 `thalamus segmentation stage 2/2`；09:20 左右最后一次日志读取仍停在该行。该阶段内部没有 closure 进度日志，因此无法由日志判断当前步数或已完成成本评估数。
- PID 5740 在 09:19:23 的状态为 `R`，累计 CPU 99.2%，主线程累计 88.1%，另一线程约 10.6%；`VmSwap=0`，`VmRSS≈3.58 GB`，读取计数中 `read_bytes=0`。这些证据不支持当前正在等待磁盘或交换空间的判断。
- 使用 `perf record -q -F 49 -p 5740 --call-graph dwarf,4096 -- sleep 8` 得到 **460 个 cycles 样本，丢失 0**。按采样 CPU cycles 的 self 比例归约：

| 共享库 | Self CPU cycles 比例 |
|---|---:|
| `libcuda.so.535.216.03` | 79.13% |
| 未解析内核地址 | 6.06% |
| `libc-2.17.so` | 5.54% |
| `libtorch_cpu.so` | 2.77% |
| `python3.11` | 1.50% |
| `libtorch_python.so` | 1.49% |
| `libtorch_cuda.so` | 1.00% |

- 栈归约中，`cudaLaunchKernel` 的 children 比例为 **82.58%**；可解析分支包含 `__sched_yield` 和 `at::native::gpu_index_kernel`。该短窗口主要消耗在 CUDA 驱动提交/等待路径，未表现为 Python 空间索引构造循环占据 CPU。
- `perf` 的旧版 DWARF 解析器报告部分 DWARF 5 信息无法解析，因此以上共享库归约比逐层调用栈更可靠，不能据此定位完整 Python 帧或某一 closure。
- `nvidia-smi pmon -i 1 -c 3 -s um` 三次现有负载观察：

| 采样 | FNIT PID 5740：SM / memory util | 另一计算 PID 48583：SM / memory util |
|---|---|---|
| 1 | 49% / 23% | 49% / 23% |
| 2 | 74% / 35% | 24% / 11% |
| 3 | 61% / 29% | 37% / 17% |

FNIT 占显存 6802 MiB，另一计算进程占 23498 MiB；GPU1 同时有一个占 804 MiB 的常驻进程，其 pmon 利用率为未报告。全卡日志在相邻采样中为 100% 利用率。PID 48583 启于 09:17:18，晚于本次 synthetic stage 2 开始约 22 分钟，所以这个 PID 只能证明采样窗口的竞争，不能单独解释此前全部停留。运行开始时的 GPU1 整卡记录亦为 100%，但该 JSONL 没有保存其他进程的 ID 或逐进程利用率。

## 冻结源码对应的工作量

实际执行源码根目录：

`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930/source_v12`

| 文件与行 | 可证实的路径 |
|---|---|
| `src/fnit/gems/core.py:152` | `compact_em` 仅在 `em_relative_cost_stop is not None` 时开启。 |
| `src/fnit/gems/recipes/base.py:206–208` | recipe 使用 L-BFGS、`index_margin=3.0`、`adaptive_index=True`；未传 `index_refresh_interval`。 |
| `src/fnit/gems/core.py:157–161` | 每次 closure 可通过 GPU `.amax().item()` 检查位移；超过 1.5 体素时将顶点传 CPU 并重建空间索引。 |
| `src/fnit/gems/core.py:239–242,279` | 原 PyTorch strong-Wolfe L-BFGS，每次 step 只有一个迭代，线搜索仍可能执行多个 closure。 |
| `src/fnit/gems/core.py:258–267` | synthetic 阶段未开启 compact，使用完整网格栅格化，然后成本归约和 backward。 |
| `src/fnit/gems/rasterize.py:208–229` | dense 路径逐 batch 进行 Torch 候选四面体查找、selected 插值、归一化和索引写入，保存反向图；未调用 compact Triton 查找。 |
| `src/fnit/gems/rasterize.py:112–142` | `build_block_index` 的确是 CPU NumPy bounding box 加 Python 三层块循环，但本次短采样没有显示该函数占据主要时间。 |
| `src/fnit/gems/core.py:281–284` | 周期索引重建分支存在，但当前标准 recipe 未启用其参数，不能用它解释当前停留。 |

历史同源 `full_stage_v12.log` 中，第二合成阶段起于 02:11:36.905，下一个 intensity stage 日志为 02:12:16.922，相邻日志间隔 40.017 秒；这是阶段边界墙钟，包含边界间的准备工作，不能当作纯 closure 或 GPU kernel 时间。

## 可支持的结论与剩余未知

**可支持：** 本次 8 秒窗口中 FNIT 仍在使用 GPU，CPU 的高占用主要落在 CUDA 驱动提交/等待栈；冻结 v12 合成阶段的 dense Torch 逐批次路径会产生大量 kernel 提交，并在共享 GPU 上竞争。当前窗口不是以 Python `build_block_index` 为主的 CPU 卡顿，也没有磁盘/交换等待证据。

**不能从此窗口推出：** 整个 20 多分钟全部由其他 GPU 任务导致；其计算与驱动等待的独占时间；是否出现比历史更多的 strong-Wolfe trials；adaptive rebuild 的次数及候选四面体数量；是否有特定极端 trial 导致索引候选膨胀。这些量未由冻结日志记录。前一 brainstem 较快也不能证明后续阶段享有相同共享负载或相同 kernel 提交特征。

远端二进制采样保留于 `gpucw1:/tmp/fnit-p5740-readonly-perf.data`；本地只保存上述小型诊断。停止进一步采样。
