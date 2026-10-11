# recon-all 的 CPU 线程预算

## 1．功能简介

[`thread_budget.py`](../../src/fnit/recon_all/thread_budget.py) 为一个 Python 作用域设置 Torch CPU intraop 数量和当前调用线程的 Numba 工作掩码，默认各为 4，退出时恢复调用方设置。它没有影像输入、坐标变换或输出影像，不改变计算精度、CUDA allocator 或 GPU 选择。Torch、Numba 已在主页 Conda 环境中，无新增依赖。

2026-10-10 的全新进程验证发现一个状态管理错误：进入作用域前 Torch 已为 4，首次 `numba.get_num_threads()` 初始化 OpenMP 后却变成 36。旧逻辑只根据初始化前的 Torch 数量决定是否调用 setter，因此跳过了重新设置，随后以实际 `36/4` 报错。修复在读取 Numba 掩码后重新检查 Torch，按实际状态设置请求预算，并记录 `torch.after_numba_initialization`。退出恢复的是首次读取的调用方 Torch 状态。

本页最新结果是线程管理的真实 API 验证。它不代表 MRI 算法精度、整例提速或 GPU 显存达标；历史影像阶段回归单列在第 6 节。

## 2．Python 调用、输入与输出

### `thread_budget`

```python
import json
import numba
import torch
from fnit.recon_all.thread_budget import thread_budget

with thread_budget(
    threads=4,  # 当前作用域的 Torch intraop 数量与 Numba 工作掩码
) as thread_report:
    actual_thread_counts = {
        "torch": torch.get_num_threads(),  # 实际生效的 CPU intraop 数量
        "numba": numba.get_num_threads(),  # 当前调用线程的工作掩码
    }
# 退出后同一份报告补充恢复结果；这里没有运行影像阶段。
print(json.dumps(
    obj={"actual": actual_thread_counts, "thread_budget": thread_report},
    indent=2,  # 使用缩进显示报告结构
))
```

| 输入 | 类型、默认值与含义 |
|---|---|
| `threads` | 具名参数，正整数，默认 `4`，单位为 CPU 工作线程；不得超过 Numba 导入时的 `NUMBA_NUM_THREADS` 容量 |

返回值是 context manager；进入时产出可写入 JSON 的 `dict`，退出时更新同一个对象。所有线程数量字段都是整数，状态字段是布尔值；没有 shape/dtype、影像空间或长度单位。

| 输出字段 | 含义 |
|---|---|
| `requested_threads`、`scope` | 请求数量及作用范围 |
| `torch.before` | 首次读取、尚未调用 Numba getter 时的 Torch intraop 数量 |
| `torch.after_numba_initialization` | 首次读取 Numba 掩码后的实际 Torch 数量；Numba 已初始化时也记录这次读取后的状态 |
| `torch.effective/restored` | 生效及退出恢复后的数量；进入后尚未应用时为 `None` |
| `torch.interop_unchanged` | 未修改的 Torch interop 数量 |
| `numba.before/effective/restored` | 首次读取、生效及退出后的工作掩码 |
| `numba.initial_capacity` | Numba 导入时确定的容量上限 |
| `numba.mask_only/threading_layer` | 掩码性质及实际后端，本次为 `omp` |
| `environment_modified` | `False`；不修改父进程环境 |
| `restoration_complete` | 退出后 Torch、Numba 是否恢复到原值 |

布尔值、非整数、小于 1 或超过 Numba 容量的请求抛 `ValueError`；超容量检查在读取/修改线程状态之前进行。后端设置失败传播原异常，实际数量与请求不符抛 `RuntimeError`。正常退出与作用域内异常都会尝试恢复；恢复操作自身失败仍传播异常，此时不能宣称已经恢复。首次读取 Numba 时若其后端初始化失败，还没有进入带报告的作用域，异常直接传播。

Numba 设置的是工作掩码，不等于操作系统进程总线程数，也不删除已经创建的线程池。Torch intraop 为进程设置，不应由多个调用线程同时修改；Numba 掩码属于当前调用线程。嵌套作用域按退出顺序恢复。此助手不设置 Torch interop，也不统一已初始化 BLAS 的内部线程池。多个进程并行时须分别分配总预算，例如总 4 线程、双侧两个进程各 2 线程。[Numba 线程管理说明](https://numba.readthedocs.io/en/stable/user/threading-layer.html)

### `native_thread_environment`

这个名称保留接口兼容。助手可为任意新的自有 Python 或独立程序生成环境副本，不启动程序。

```python
import json
import os
import subprocess
import sys
from fnit.recon_all.thread_budget import native_thread_environment

child_environment, child_thread_report = native_thread_environment(
    threads=4,  # 子进程初始化相关库时使用的预算
    environ=dict(os.environ),  # 复制完整环境，保留设备映射与精度设置
)
completed_process = subprocess.run(
    args=[sys.executable, "-c", "import os; print(os.environ['OMP_NUM_THREADS'])"],
    env=child_environment,  # 仅传给这个新进程，不改变父进程 os.environ
    check=True,  # 子进程失败时抛出异常
    capture_output=True,  # 接收这次示例输出
    text=True,  # 将输出解码为字符串
)
print(json.dumps(obj=child_thread_report, indent=2))
```

| 输入/输出 | 结构及限制 |
|---|---|
| `threads` | 正整数，默认 `4`；非法值抛 `ValueError` |
| `environ` | `Mapping[str, str]` 或 `None`，默认 `None` 表示复制 `os.environ`；不能转为字典时传播 Python 类型错误 |
| `child_environment` | `dict[str, str]`，保留输入环境其余变量，把下面六个线程变量设为请求数量的十进制字符串 |
| `child_thread_report` | `dict`，含 `requested_threads/scope`、六个变量的 `environment_before/after`、`parent_environment_modified=False`、`active_worker_counts_verified=False` |

六个变量为 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`、`NUMBA_NUM_THREADS`。只记录这些变量，不把完整环境写入报告。该助手不改变已初始化的库；程序显式参数和源码设置仍可能覆盖环境，必须另行确认实际线程数。当前 `input_stage_parallel.py` 和 `hemisphere_parallel.py` 使用环境副本分配 worker 预算；这项静态接线核对不等于其整例资源验收。

## 3．命令行调用与复现

线程助手没有单独安装的 CLI。最新首次初始化验证可运行本页专属驱动：

```bash
repository_directory=/path/to/FNIT  # 含 Git 历史和当前候选模块的仓库
python_executable=/path/to/conda/env/bin/python  # 同一已安装 Torch/Numba 的 Python
output_directory=/tmp/thread_initialization_repeat  # 必须是不存在的新目录
baseline_commit=516a93e287d1610b50201e48e6c49cf63ff7e1a4  # 本次旧代码所在提交

"$python_executable" \
  "$repository_directory/validation/recon_all/optimizations/20261010_thread_budget_initialization/benchmark_first_context.py" \
  --repository "$repository_directory" \
  --baseline-ref "$baseline_commit" \
  --python "$python_executable" \
  --output "$output_directory"
```

`--repository` 是待冻结的仓库；`--baseline-ref` 默认 `HEAD`，本次显式使用上面提交；`--python` 默认驱动当前 Python，控制旧/新子进程使用的解释器；`--output` 必填且不能已存在。驱动从 Git 读取旧模块、冻结当前工作树模块，分别启动全新进程，设 `OMP_NUM_THREADS=4`、移除 `NUMBA_NUM_THREADS`，禁用可见 GPU；不预读 Numba 掩码，不提前初始化线程池。getter/setter 观察器只记录真实调用，不伪造返回值。

输出有 `baseline.json`、`candidate.json`、`comparison.json`、`frozen_modules.json`、原模块快照和子进程 stdout/stderr。`comparison.status=passed` 表示候选生效并恢复；`baseline_bug_reproduced` 单独记录旧错误是否复现。驱动捕获作用域异常并保存报告，因此子进程退出码 0 不等于 `context_completed`，须读取 `status/exception`。若环境选择了其他线程后端，旧错误可能不出现；报告实际初始化路径。输出创建、Git 读取或导入失败不伪造线程结果。

## 4．原软件调用与实现边界

没有对应 FreeSurfer 独立命令。这里管理的是 Torch 和 Numba 的运行时设置，属于 FNIT 调度内部步骤。`torch.set_num_threads` 控制 CPU intraop，[官方 API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html) 提供其定义；Numba 的设置控制其并行掩码。它不调用 FSL、FreeSurfer 或其包装软件。

第 6 节历史影像阶段对应 `mri_ca_normalize` 和 `mris_sphere`，原软件仅用于独立 benchmark。历史固定源码的 N4 单线程、拓扑 `-threads 1`、EM 自报线程数属于那些程序的运行记录，不能据此推断当前自有 Python 阶段的实际线程数，也不能把环境中写入 `4` 当作已验证活跃线程数。

## 5．最新代码的实际结果

2026-10-10，本地 WSL 的两个全新进程，Python 3.11.16、Torch 2.5.1、Numba 0.67.0、NumPy 1.26.4；36 个逻辑 CPU，可用 affinity `0..35`，实际 Numba 后端 `omp`。未把 affinity 限定为 4，以保留本次首次初始化容量 36 的触发条件。输入是运行时状态，没有 T1 或表面文件，因此无脑图、影像精度指标或坐标单位。

| 同一个首次作用域 | 旧模块 | 修复模块 |
|---|---:|---:|
| 进入前 Torch | 4 | 4 |
| 首次 Numba getter 返回掩码 | 36 | 36 |
| 首次 getter 后 Torch | 36 | 36 |
| 生效检查 Torch/Numba | 36/4，报错 | 4/4，成功进入 |
| 退出后 Torch/Numba | 36/36 | 4/36，恢复完成 |
| 状态 | `context_failed` | `context_completed` |
| 含 Python 冷启动、导入及报告的墙钟 | 2.990464 s | 3.156746 s |
| 仅作用域墙钟 | 0.001888 s | 0.011511 s |

上述一次配对用于状态回归，不能用毫秒级差值评价 MRI 性能。Torch interop 18、matmul/cuDNN TF32 标志和环境前后相同；CUDA 全程未初始化。它不设置 FNIT GPU 精度策略。本次没有 GPU 计算或显存采样，不能报告“零显存”。

| 实际测试对象 | 版本或 SHA-256 |
|---|---|
| 旧模块所在提交 | `516a93e287d1610b50201e48e6c49cf63ff7e1a4` |
| 旧 `thread_budget.py` | `bf0b40e5c6c7232d0de23dee3c491a3c27850d79922d81df71689933a9692fca` |
| 候选工作树 `thread_budget.py` | `95cb4c4b75f187da49b50f80d6b9750404a5bb9091fe9ef9f7c6d2a179bfca44` |
| 实际 v2 驱动 | `5584ccce5795c66e97062a7f75c30de52b431a997cf612636f497df94beec962` |
| Python 程序 | `2aec79dfcdfb70acd010432c153a9f04e98d4f390a65092a5426336ff5474ab8` |

候选当时尚未提交，按模块 SHA 绑定，不能把旧提交当候选代码版本。完整原始事件及各阶段状态见[旧模块](../../validation/recon_all/optimizations/20261010_thread_budget_initialization/fresh_v2/baseline.json)、[候选模块](../../validation/recon_all/optimizations/20261010_thread_budget_initialization/fresh_v2/candidate.json)、[比较](../../validation/recon_all/optimizations/20261010_thread_budget_initialization/fresh_v2/comparison.json)和[哈希收据](../../validation/recon_all/optimizations/20261010_thread_budget_initialization/fresh_v2/receipt.json)。v1/v2 模块相同；v2 修正驱动开始/完成时间字段，不追改 v1 原记录。

主任务已有线程与入口契约 JUnit，本页只读取现存 XML，未重新执行该命令：63 个 testcase、0 失败/错误/跳过，suite 17.520 s；控制台为 49 tests 加 14 subtests、17.56 s。两种计数/计时边界分别记录在[契约摘要](../../validation/recon_all/optimizations/20261010_thread_budget_initialization/fresh_v2/parent_contracts.json)。这项验证不包含本轮 MRI 精度、完整 pipeline 或服务器线程重复性。

## 6．更新与历史 benchmark

| 日期 | 修改或验证范围 |
|---|---|
| 2026-10-10 | 首次 Numba OpenMP 初始化后重新检查 Torch，记录实际状态；本页全新进程旧/新回归通过 |
| 2026-10-04 | 公开重建失败报告修复：阶段异常发生在内部总计写出前时，不访问不存在的总计；完整公开墙钟与失败状态仍保存 |
| 2026-10-01 | 初版作用域与环境副本；9 项状态管理测试，本地 9/9、0.402 s，服务器独立临时目录 9/9、0.012 s |

2026-10-01 的旧真实 API 记录使用 Torch 2.5.1、Numba 0.67.0、TBB，Torch 96/Numba 192 → 4/4 → 96/192，请求 193 明确报错，CUDA 未初始化。它没有触发本次 OpenMP 首次初始化条件；不是本版代码的验证。

同日冻结提交 `1b8c36d25a68e253a1e59b6d02114890afa467de` 在 CPU 节点比较以下真实自产输入：Torch 均为 4，Numba 容量 192，各阶段按 128 → 4 运行，最后恢复 192。时间包含读写和首次 JIT，无预热、交换顺序或重复配对，仅用于结果回归。

| 历史真实阶段 | Numba 128（s） | Numba 4（s） | 同输入结果 |
|---|---:|---:|---|
| sub-01 CA normalize | 20.328 | 22.838 | norm/ctrl 元素差 0；dtype、shape、affine 相同 |
| sub-02 CA normalize | 20.447 | 23.015 | 同上 |
| sub-01 LH standard sphere | 209.395 | 218.038 | 坐标差 0；顶点数及有序面相同 |

最大/P99 元素或坐标误差均为 0，affine 差 0 mm；这些是解码数组/几何回归，不宣称所有压缩文件字节一致。CA 输入为同例 FNIT 自产 `nu/brainmask/LTA` 与声明 GCA；sphere 为自产 `inflated/smoothwm`，未用官方结果补生产输入。本组 4 掩码没有更快，不能据单次固定顺序解释历史 WM 666 s 或整例速度。

| 历史实际测试模块 | 当时 SHA-256，非当前版本声明 |
|---|---|
| `thread_budget.py` | `bf0b40e5c6c7232d0de23dee3c491a3c27850d79922d81df71689933a9692fca` |
| `ca_normalize_python.py` | `abb436826e600101c2c5087e2367e8a36a7c038c99eda10bc4f5be1e7486661a` |
| `sphere_standard_run.py` | `7544ccb77153a7ac242b88d1c9752bdd08ce5765ba483cd644a4075e94e90405` |
| `sphere_standard_average.py` | `f79dc72a574d3bef756f06b97ab2546d35662c1d6acbd769e034fa542d84bd78` |

原报告保留输入和源码哈希：[sub-01 CA](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_ca_sub01/report.json)（`73c6074d6fd761f9f9e17a67ac9c2a9d597d88ff280dc0ea5e1f6ee522e30e0d`）、[sub-02 CA](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_ca_sub02/report.json)（`f0a5ecd72bf33c9336d45a3377f963b96adbcdb447aab2ffc488092f4a3e778d`）、[sub-01 LH sphere](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_final_1b8c36d/thread_sphere_sub01_lh/report.json)（`5eb86d5ed201d899c3507fc1ddc49a944798ceb605acfe73052db017f87ebea0`）。原[阶段驱动](../../validation/recon_all/python_gpu_port/benchmark_thread_budget.py) SHA 为 `97d79c25253bb38a73acf3cccb5bd4ffe4b4293f2b61da9f20926e0eb9e30828`，复现该历史结果须先冻结对应旧提交，不能给当前代码贴旧标签。

历史 EM 环境变化及实际自报线程记录见[日志来源快照](../../validation/recon_all/python_gpu_port/performance_20261001/thread_runtime_logs_snapshot.json)：2026-09-30 记录为 1，2026-10-01 受控驱动记录为 4；不推广到其他程序。历史 e036 球面 cProfile 的 average 累计 LH/RH 11.21/6.36 s、distance SSE 98.12/50.54 s，仅为带剖析开销的观察。上述旧研究未包含 RH/第二例球面、ABBA 或原始 T1 整例线程对照；此描述不代表当前整个项目的完成状态。

现有并行核仍须尊重每顶点/体素内累加顺序：标准球面平均按下一轮缓冲区独立写顶点；CA/归一化严格卷积分别写 x/z 切片并顺序执行三个轴。串行 `@njit` 不会因增加掩码自动并行；共享 `surface.defects.mgz` 必须按当前调度的独立 worker/合并规则处理，线程预算不能代替文件依赖管理。

## 7．参考文献与代码

- [FNIT 线程管理代码](../../src/fnit/recon_all/thread_budget.py)与[契约测试](../../tests/recon_all/test_thread_budget.py)：本功能没有独立神经影像算法或对应影像参考文献。
- [PyTorch `torch.set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)、[线程环境变量](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)与[PyTorch 源码](https://github.com/pytorch/pytorch)：本次实际运行 Torch 2.5.1，以报告记录的安装版本为准。
- [Numba threading layers / set_num_threads](https://numba.readthedocs.io/en/stable/user/threading-layer.html)与[Numba 源码](https://github.com/numba/numba)：本次实际运行 0.67.0，动态掩码受初始化容量约束。
- 历史阶段参考为 [FreeSurfer 源码](https://github.com/freesurfer/freesurfer)，只用于第 6 节独立 benchmark，不是本线程助手的运行依赖。
