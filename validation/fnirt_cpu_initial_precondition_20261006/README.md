# FNIRT CPU Stage4：初始预条件与归约首差

## 1. 功能与实际结论

本轮只从 accepted solve3 checkpoint 读取半梯度、半对角线，执行当前 Torch PCG 与自有 CPU 候选的循环前原始语句。没有 callback、PCG 循环、evaluate、linearize、H、官方进程、影像或 GPU 计算；生产源码不变。

两方向 SHA 均与 Stage3 首个 PCG callback 的已存方向逐位相同。实际 inner floor 没有改变元素；**reciprocal×rhs 与直接 division 产生 332 个 FP64 位模式差异，每个最大 1 ULP，maxabs1.7763568394002505e−15**。同 effective weights 下直接除法与候选方向逐位相同，所以此 saved state 的首个方向分叉来自预条件表达式的舍入顺序。另有同一 z 的 rho 和同一 rhs 的 norm 末位差异。

这些结果定位初始局部分叉，没有证明 Stage3 最终 step 相差6.9665%的唯一原因，也没有解释当前对官方 RHS 的7.3657e−7相对差；不证明数学 bug 或精度改善，**不接入默认运行时**。完整原记录见 [summary](run/initial/summary.public.json)、[机械门](ACCEPTANCE.public.json) 和 [解释](INTERPRETATION.public.json)。

## 2. Python 调用、输入和输出

科学清单 SHA 为 `7f5cdc7572aac6b27752d4ca47bffc1be3bd2087f4c751ce1b1e02fcbcc130e5`；[freeze](source/freeze.public.json) 自排除11个文件。worker 是 [initial_precondition.py](source/initial_precondition.py)，[prefixes.py](source/prefixes.py) 从已绑定的两个函数提取第一个循环之前的原 AST 语句，保留计算顺序，只在末尾返回诊断字段。局部 Torch proxy 记录调用，不全局 patch Torch；未执行任何循环体。

输入为私密 NPZ 的 `gradient_half` / `diagonal_half`：CPU FP64、各 `[1177]`，复制后恢复记录 stride，两个逻辑值 SHA 与 Stage3 桥一致。checkpoint68数组、2647508B、SHA `78abee8589e6e026fede946136c9e78589817b01a1d55d2085aee19758ceae0e` 留在服务器，仅上述两项读取；Stage1/Stage3 public scalar 用于来源和方向 SHA 门。没有加载参考解或官方轨迹。

共同受控 LM 输入按生产公式派生：

```python
# checkpoint 自产半量，保持 CPU FP64 / no-grad。
gradient_half = tensors["gradient_half"]
diagonal_half = tensors["diagonal_half"]

# tau=.001 是本轮受控比较值，未证明是历史 solve3 的实际 damping。
lm_tau = 0.001
floor = torch.finfo(diagonal_half.dtype).eps * diagonal_half.abs().mean().clamp_min(1)
damping_diagonal = diagonal_half.clamp_min(floor)
rhs = -gradient_half
preconditioner = (1 + lm_tau) * damping_diagonal

# 原 Torch 前缀还 clamp preconditioner，然后 reciprocal×rhs。
# 自有前缀直接 rhs/preconditioner；两原前缀均计算一次 rho、一次 rhs norm。
```

输出只有原 public summary、source/input/flags 后置检查、前后 preflight、调度和 exit 收据。诊断没有生成或返回数组。参数与上限保存在 [expected](source/expected.public.json)；首方向 SHA 不匹配即停止后续 cross reduction。

## 3. 调用与资源

Root 与 peer 审查冻结后授权唯一派发；controller189857 / start_ticks534431758，enqueue1791242617.6349792。19000s common CPU lock 等待、19600s controller、180s child，20,000,000,000B address-space cap。nodecw7 同8个物理核 `32,36,40,44,48,52,56,60`，OMP/OpenBLAS/MKL/Numba8、Torch8/interop1；GPU invisible，CUDA 未初始化。

cuDNN TF32=True、matmul TF32=False、grad、默认 dtype 前后不变；22 source/input 绑定和保存/派生逻辑值前后精确。原始科学目录拒绝覆盖，输出700/600。六锁 INDEX 原 prepared/queued/completed 收据保留，completed JSON SHA `4c63be5eb3e8daf5494880d1b496a415e34ce22dab29ce86a1145dbe623905f8`，MD SHA `4b7dd816a480ce036c620930b8cf32f575cc69782115c88613d4f870c786a01f` 是该次更新身份，后续任务可再更新全局索引。

本地只读复核入口不导入 NumPy/Torch/Numba，也不重新执行原 worker：

```bash
PYTHONDONTWRITEBYTECODE=1 python validation/fnirt_cpu_initial_precondition_20261006/verify_report.py
```

原冻结 plan 的 preparation 状态是历史字段；实际状态以 [launch](run/launch.public.json)、原 summary 和三 exit0为准。再次科学运行需新的范围，本叶不派发第二次。

## 4. 原软件范围

这是 PCG 内部循环前的预处理，没有原软件独立 CLI。本轮没有官方 FNIRT、observer、native arithmetic bridge 或 native 同系统 oracle。候选沿用已发布自有 Dot/Norm 的 NumPy/Numba/llvmlite 环境，无新增依赖、无安装 FSL 链接。公共目录不包含原软件源码、二进制、许可证、凭据或私密数组。

前后 `/proc/self/maps` FSL DSO 文件身份快照为空；仅证明记录时的身份，不代表全时段系统调用审计，也不把旧 `dnrm2` 文件身份升级为实际 kernel 调用证明。生产 GPU 路径和精度策略不变。

## 5. 精度、计数与时钟

| 同一 saved half g/diag 与受控 tau=.001 | 实际结果 |
| --- | --- |
| Stage3 两首方向 SHA 桥 | 各逐位相同 |
| inner floor 改值 / 改 bits 数 | 0 / 0 |
| reciprocal×rhs vs division | 332 /1177 bits 不同，max1 ULP |
| division vs同 effective weights division | 0 bits 不同 |
| 同 old z 的 rho：Torch / owned | 1.5627520580425893 / 1.5627520580425889 |
| 同 new z 的 rho：Torch / owned | 1.562752058042589 / 1.5627520580425889 |
| 同 rhs 的 norm：Torch / owned | 5.227043232336545 / 5.227043232336546 |
| 原前缀 / dot / norm 调用 | 2 /4 /2 |
| callback、PCG、evaluate、linearize、native、image | 全0 |
| science / controller / after-preflight exit | 全0 |

rho 的输入是同一 rhs 与各已绑定 z，cross 控制区分预条件表达式和归约；完整 FP64 little-endian hex 保存在 scalar ledger。ULP 是浮点位模式距离，不能写成 voxel 误差。

worker through-post-bind/pre-summary `2.448967095464468s`，imports/binding `1.5133079132065177s`，maxRSS398008KiB。包含首次导入/JIT的有限诊断时钟，不是两个算法速度比、配准耗时或 CPU 加速 benchmark。本轮未新读 MRI，因此没有新脑图；真实输入与既有配准图来源沿用前阶段报告。

## 6. 版本、失败和下一步

Stage1 已通过 saved solve3 的2g/2diag精确桥；Stage2 已恢复有限真实 callback；[Stage3](../fnirt_cpu_matrixfree_pcg_20261006/README.md) 是同 callback 的旧49/new53自然停止控制。本轮补充 Stage3 第一方向分叉的预条件和 scalar 来源，没有重复求解，也没有收紧容差。

冻结前 helper Python3.9生成的 AST 缺 postponed-annotation flag，静态定义编译失败；未调用数学、未上传或派发，原记录见 [PREPARATION_HISTORY](source/PREPARATION_HISTORY.public.json)。完成后的第一次纯 INDEX CLI 忘记 `--phase`，argparse退出2且未写索引；修正参数后完成元数据登记，未重跑科学。失败与实际完成状态见 [STATUS_AND_FAILURES](STATUS_AND_FAILURES.public.json)。

27项原 metadata、107162B 分3个≤45000B块收回后逐全文件 SHA 重验；输入数组留在服务器。下一优先是当前对官方 RHS 的7.3657e−7残差出处，按已存同参数/输入/schema逐项审查 Jte，而不是增加 PCG次数或容差。本轮尚未授权新的组装、官方 capture、callback 或 solve。

## 7. 源码与参考

成熟来源是 FNIT [optimizer.py](../../src/fnit/fnirt/optimizer.py)、[registration.py](../../src/fnit/fnirt/registration.py)、[spline.py](../../src/fnit/fnirt/spline.py)。17生产源码本轮不变；两个自有候选源与已发布 `fnirt_cpu_current_replay_20261006` 字节一致。全部源码/输入 provenance 见 expected 与 raw bindings。

原软件：[FSL FNIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt.html)。PCG参考：Hestenes MR, Stiefel E. Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 1952, 49:409–436。本文仅报告已保存真实科学状态上的局部分解，不代表完整非线性配准已经等价。
