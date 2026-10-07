# FNIRT CPU 自有归约有限核验（2026-10-06）

## 1. 功能与范围

独立诊断候选用现有 Numba/llvmlite 实现 FP64 点积、范数和严格 CSC 乘法。复用真实配准已有的共享 A/RHS、预条件对角线及逐轮向量：先通过 93 行算术门，再执行一次同一共享系统的自然停止检查。本轮没有改动生产模块、调用原始影像或启动新配准、组装、官方优化器。

结果：93 行的五项标量全部逐位一致；共享 solve3 自然停在第 69 轮，逐轮向量、标量和最终解逐位一致。该结果只覆盖已绑定的共享线性系统，完整 CPU 非线性估计尚未通过。

## 2. Python 调用、输入与输出

[own_reductions.py](source/own_reductions.py) 是自写候选，依赖项目已有 NumPy、Numba 及其 llvmlite。FP64 FMA 使用项目已有 `_raster_cpu.fma32` 的 LLVM intrinsic 接入方式，精度为 FP64。当前没有生产公共 API。

```python
import numpy as np
from own_reductions import dot, norm

# left_vector 和 right_vector 是等长、一维、连续、有限的 CPU float64 向量。
left_vector = np.asarray(existing_left_vector, dtype=np.float64)
right_vector = np.asarray(existing_right_vector, dtype=np.float64)
dot_value = dot(left_vector, right_vector)  # FP64 点积标量。
norm_value = norm(left_vector)             # sqrt(SumSquare) 的 FP64 标量。
```

诊断输入均保留在授权的服务器目录，公开文件仅包含标量与 SHA-256：

| 输入 | 格式与含义 |
| --- | --- |
| `solve{2,3}_rhs.f64` | little-endian FP64 RHS；实际维度从文件读出 |
| `r/z/p/q/r_after` | 每轮残差、预条件残差、方向、A 乘方向及更新残差 |
| `solve3_A.f64` | 同一系统的 row-major FP64 方阵；从 RHS 推断形状 |
| `solve3_diagonal.f64` | 同一 A 的正预条件对角线 |
| `solve3_x0.f64` | 该系统的零初值 |
| `solve{2,3}_trace.csv` | 已存 rho、分母、alpha 和更新前后相对残差 |
| `solve3_native_solution.f64` | 仅用于求解完成后的比较，不参与更新或停止 |
| 前置绑定清单 | 486 个旧输入和 17 个当前 FNIRT 源文件的大小、SHA |

[contracts.public.json](results/contracts.public.json) 给出每行标量的位比较；[norm_contracts.public.json](results/norm_contracts.public.json) 单列相对范数门；[summary.public.json](results/summary.public.json) 给出自然停止和逐轮比较；[residuals.csv](results/residuals.csv) 保存自有动态臂的相对残差。真实数组、解向量及参考软件代码/二进制不发布。

## 3. 诊断命令与参数

实际运行源见 [probe_owned.py](source/probe_owned.py)、[run_owned_v2.sh](source/run_owned_v2.sh) 和 [preflight_owned_v2.py](source/preflight_owned_v2.py)。运行器使用 canonical 索引中的任务目录，先取得共用 CPU outer lock，再取得本叶 inner lock；nodecw7 八核亲和性为 `32,36,40,44,48,52,56,60`，各线程预算为 8，CUDA 不可见。控制器最多 900 秒、求解子进程最多 180 秒；不成功则停止。

```bash
# oracle_directory：索引中原有授权共享系统目录，包含上述私有数组。
# result_directory：新结果叶；其父目录先放置通过的 preflight_before.public.json。
python probe_owned.py \
    --oracle "${oracle_directory}" \
    --output "${result_directory}"
```

`--oracle` 指定既有共享状态输入，`--output` 必须是尚不存在的新叶。私有 `expected.json` 保留实际输入定位；公开 [before](bindings/preflight_before.public.json) / [after](bindings/preflight_after.public.json) 记录全部大小和 SHA，不公开该定位文件。复跑需使用已授权的原始输入映射，不能从公开标量恢复数组。

求解器固定原有 `tolerance=1e-3`、`maximum_iterations=500`，用实际相对残差自然停止。尺寸、迭代数和地址对齐均不作为输入特例；保存参考只用于事后比较。生产 Python/CLI 没有新开关。

## 4. 原实现与运算上下文

原软件为既有 FSL 6.0.7.4 / FNIRT 2203。DotProduct/NormFrobenius 是内部函数，没有独立官方 CLI。本轮只对已存 r/z 发出一次自写算术 executable 的 stdio 请求；gdb 在 BLAS 内核入口停止，没有启动原 FNIRT/观察器/优化器，也没有产生新的原生算术评分。

[dispatch.public.json](bindings/dispatch.public.json) 实际观察到 `ddot_k_COOPERLAKE + 19`，两输入 stride 都为 1；长度 1177、地址模 64 为 0/32 是这次观察的元数据。自有代码没有按这些长度或地址编写分支。

[context.public.json](bindings/context.public.json) 绑定原头文件、ELF、CPU 和现有依赖：

- `armawrap/function_dotproduct.hpp` 的 DotProduct 调用 `arma::dot`；`op_dot_meat.hpp` 的 FP64 direct_dot 在该 BLAS 分支调用 ddot。
- 实际 contiguous COOPERLAKE 内核：四组八路 FMA 累加，随后按固定顺序折叠；余 16 项使用四组四路 FMA，尾部完整四项块独立乘法后顺序相加，余下标量使用 FMA。算法分块由输入长度计算。
- `functions_base.hpp` 的 NormFrobenius 定义为 `sqrt(SumSquare)`；SumSquare 经 square/accu，当前非 fast-math 分支使用偶/奇两条独立平方累加链。这里没有调用 dnrm2。前序记录的 dnrm2 `dladdr` 只绑定库身份，不能当作实际范数内核执行证明。

参考 OpenBLAS SHA 为 `8aaf756b5d22aa633640fb37029aeb8f5ae8136085c92ce8ad66d7bfe1cedcfe`。实际 CPU 为 Xeon Gold 6418H。这里没有证明其他 BLAS CPU selector、非单位 stride、大向量线程归约或其他尺寸均与原生逐位一致；93 行没有单独覆盖 `n<=32` 的两链点积路径。

## 5. 精度、时间与依赖证据

| 门 | 实测结果 |
| --- | --- |
| 93 行 rho / 分母 / alpha / 更新前相对范数 / 更新后相对范数 | 各 93/93 逐位一致，共 465 个标量；最大差 0 |
| 相对范数单列 | 186/186 逐位一致；没有存档的绝对原生范数标量，因此未做绝对范数原生门 |
| 自有动态 solve3 | 原门 `1e-3` / 500，69 轮自然停止 |
| 逐轮 r/z/p/q/r_after | 69×5×1177 个 FP64 元素逐位一致 |
| 动态标量 / 最终解 | 全部逐位一致；最终解不同 bit 数 0 |
| 终点递推相对残差 | `0.0002717867671587692` |
| 自有 CSC 重算真实相对残差 | `0.0002717867671592058` |
| 运行前后绑定 | 各 503/503 通过，全部输入/17 源文件及运行器字节相同 |
| 耗时 | 合同含 JIT 0.904 s；合同+动态+诊断共 6.482 s，未做性能对照 |

[runtime.public.json](results/runtime.public.json) 记录 NumPy 1.26.4、Numba 0.61.2、llvmlite 0.44.0。现有环境还包含 SciPy 1.17.1；主页 Conda 已声明 NumPy、SciPy、Numba，llvmlite 是 Numba 现有依赖。已安装包 metadata 为 BSD 系列许可，没有新增依赖、编译器安装或原软件资源再分发。

自有 dot/norm 不调用 BLAS；诊断误差统计可使用现有 Conda NumPy 算术库。合同后、动态前读取的一次 `/proc/self/maps` 快照没有 FSL DSO，公开内容只保留库 basename 和 SHA；`LD_LIBRARY_PATH` / `LD_PRELOAD` 均为 null。自有 JIT 汇编包含 `vfmadd`，记录了汇编 SHA；这不是逐条指令运行跟踪，也不是全程系统调用审计或干净部署验收。

没有新增 MRI 输出，所以本叶提供真实残差 CSV，沿用前序真实数据报告的影像来源，不制作模拟脑图。

## 6. 版本、失败与生产边界

实测 main 为 `4f3267abac486314636d9b081a85b78833e62167`；旧输入绑定来自 `6f624040…`，本轮没有把旧捕获改标为新 main。17 个当前 FNIRT 文件均与旧绑定相同，其中 registration 为 `caab8ffc…`，optimizer 为 `903d5031…`。完整源 SHA 见前置报告和 [manifest.json](manifest.json)。

保留 [FAILURES.json](FAILURES.json) 和两个运行日志：首个计算启动因 CPU PATH 中没有 git，停在前置 HEAD 读取，未执行候选数学。v2 改用标准库读取 HEAD/ref；归约与求解源码没有改变。更早的目录重定向顺序失败也保留，没有对应科学计算。

本轮证明自有有限算术可在该共享系统替代参考库算术桥。前序 current CPU 组装仍有 RHS 相对差 `8.3642e-7`，FSL 顺序 Jte 控制后为 `7.3657e-7`，H 相对差约 `1.0285e-9`。本叶没有新组装、完整优化轨迹、拓扑或最终 warp 门；不能将 69 轮共享系统复现称为整个 CPU 优化器或完整配准修复。生产 CPU/GPU 路径均未改动。

## 7. 参考与复核入口

- [前序共享系统与原生算术桥报告](../fnirt_shared_followup_20261006/README.md)：冻结状态、失败 LOCAL 桥及未解决组装差。
- [原始 PCG 共享状态报告](../fnirt_pcg_shared_state_20261004/README.md)：原生向量与 trace 来源。
- [FSL FNIRT 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/index.html)。实际安装源码身份由 context SHA 约束。
- [OpenBLAS 官方代码库](https://github.com/OpenMathLib/OpenBLAS)、[Numba 官方代码库](https://github.com/numba/numba)、[llvmlite 官方代码库](https://github.com/numba/llvmlite)。
- Jenkinson M, et al. FSL. *NeuroImage* 2012;62:782–790. doi:10.1016/j.neuroimage.2011.09.015。
