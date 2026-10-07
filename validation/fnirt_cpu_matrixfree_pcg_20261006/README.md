# FNIRT CPU Stage3：真实恢复 callback 上的两臂 PCG

## 1. 功能与实际结论

本报告记录同一已恢复真实 level state 上，当前 Torch PCG 与自有 CPU 候选的一次有限比较。两臂均使用成熟 optimized matrix-free callback，保持生产源码和 GPU 路径；没有 MRI 预处理、evaluate、linearize、新 H 组装、CSC action 或官方进程。

旧 PCG 自然停止于 **49 轮**，候选于 **53 轮**；两臂的递归残差和同 callback 复核的实际相对残差均低于声明的 `1e-3`。两自产 step 的相对 L2 差为 **6.9665%**，最大绝对差 `0.7198678639661988`，单位是混合位移系数/强度参数坐标，不能写成 voxel 位移。没有原生同系统解，结果不证明精度改善或算法 bug；**不接入默认运行时**。

完整原始记录见 [summary.public.json](run/stage3/summary.public.json)，机械验收见 [ACCEPTANCE](ACCEPTANCE.public.json)，解释见 [INTERPRETATION](INTERPRETATION.public.json)。Stage1 是已通过的 g/diag 精确桥；Stage2 是已通过的有限 callback 恢复诊断；本轮是两自产求解控制，均不代表完整非线性配准已等价。

## 2. Python 调用、输入和输出

科学源码冻结 SHA 为 `7ee1eb753514693ff7f00c64cdea43c12fcfb2d83293b717c8db3deeef9166fd`；[清单](source/freeze.public.json) 自排除 10 个文件。worker 为 [stage3_pcg.py](source/stage3_pcg.py)，候选为已发布字节相同副本 [pcg_cpu_candidate.py](source/pcg_cpu_candidate.py) 和 [own_reductions.py](source/own_reductions.py)。

输入仅为 accepted solve3 checkpoint 及其 Stage1/Stage2 标量报告、当前完整 g/diag、当前存档 H。H 只核每臂 `unit[0]` 恢复列，不用于求解。checkpoint 68 数组的逻辑值和记录字节 stride 均显式恢复，两臂独立 storage/cache；原 pointer、alias graph 和原 Python cache 对象未持久化。输入绑定见 [expected](source/expected.public.json)。没有读取原始 MRI、native A/RHS/解、旧 paired 解或参考轨迹。

生产 LM 分支的数量是半 g、半 diag 和半 H callback，本轮使用同一公式：

```python
# 同一 checkpoint 的自产半梯度和半对角线，CPU FP64 / no-grad。
gradient = tensors["gradient_half"]
diagonal = tensors["diagonal_half"]

# .001 是本次明确声明的受控 LM 比较值，未证明是历史 solve3 damping。
tau = 0.001
floor = torch.finfo(diagonal.dtype).eps * diagonal.abs().mean().clamp_min(1)
damping_diagonal = diagonal.clamp_min(floor)
rhs = -gradient
preconditioner = (1 + tau) * damping_diagonal

def damped(direction):
    return mature_half_matvec(direction) + tau * damping_diagonal * direction

# 每臂初值为零，容差和最多迭代数不变；两候选保持各自完整算法。
step, pcg_report = solver(
    damped, rhs,
    diagonal=preconditioner,
    tolerance=1e-3,
    max_iterations=500,
)
```

`effective_lambda=9049.463427795125` 是已包含在 callback 内的弯曲正则化权重，含义不同于 LM `tau`。两臂 floor 同为 `4.846049694038592e-16`，rhs/damping/preconditioner 值哈希相同。旧 PCG 内部还对 preconditioner 再 floor，再 reciprocal×residual；候选直接 division，dot/norm 和 beta 表达式也按原候选保留。这是完整算法差分，不能后来把差异只归因 dot/norm。

输出是原始 public summary、preflight、退出和调度记录。两私密 step 各为 little-endian FP64 `[1177]`、9416B、mode600；仅 SHA/schema 收回，数组留在服务器。摘要记录每次 callback 的 direction/action SHA、阶段、实际停止报告、true residual 和 immutable/source/flags 后检查。

## 3. 调用与资源

协调者审查同一冻结清单后授权唯一派发；无数值重试。`source/PLAN*` 和 `freeze.public.json` 保留冻结时的 preparation 状态；实际执行状态以 [launch](run/launch.public.json)、三个 exit 和原 summary 为准，不能把历史计划状态视为任务仍待执行。

controller173178 / start_ticks534104746；19000s 公共 CPU lock 等待、19600s controller、180s science child，20,000,000,000B address-space cap。nodecw7 physical cores 为 `32,36,40,44,48,52,56,60`；OMP/OpenBLAS/MKL/Numba8、Torch8/interop1。GPU invisible，CUDA 前后均未初始化；既有 cuDNN TF32=True / matmul TF32=False / grad / 默认 dtype 状态前后不变。输出700/600，实际 runner 永不覆盖已存在科学目录。

本地只读复核入口如下，不导入 NumPy/Torch/Numba、不运行数值：

```bash
PYTHONDONTWRITEBYTECODE=1 python validation/fnirt_cpu_matrixfree_pcg_20261006/verify_report.py
```

as-run worker/runner 仅用于理解已完成执行；再次运行需要新的任务范围，不应对原 leaf 派发第二次。

## 4. 原软件范围

PCG 是 FNIRT 内部线性系统步骤，没有原软件对本 restored Python callback 的独立 CLI。本轮未启动官方 FNIRT、原 observer 或官方算术桥，也未读取 native 同系统解。项目候选自有 Dot/Norm 使用已有 NumPy/Numba/llvmlite 链，无新增依赖、无安装 FSL 链接；完整进程中的成熟 Torch/NumPy 库仍按现有 Conda 环境加载。

前后 `/proc/self/maps` 的 FSL DSO 文件身份快照均为空；这是记录时的快照，不是全时段系统调用审计，也不把旧 `dnrm2` 文件身份取证升级为实际调用证据。项目默认 CPU/GPU 源码不变。

## 5. 精度、实际计数与时钟

| 当前同一 restored half matrix-free LM 系统 | 当前 Torch | 自有候选 |
| --- | ---: | ---: |
| 自然 PCG 轮数 | 49 | 53 |
| 递归相对残差 | 0.0007101809160834593 | 0.0006769837850846045 |
| 同 callback 实际相对残差 | 0.0007101809160836206 | 0.0006769837850851172 |
| 实际绝对 L2 残差 | 0.0037121463511494562 | 0.00353862351223074 |
| callback 数：unit + PCG + true residual | 51 | 55 |
| unit0 恢复列不同 bits | 0 | 0 |
| 两自产 step 比较 | 1177 bits 不同；相对 L2 6.9665%、max0.7198678639661988 | 同左 |

共同 rhs L2 为 `5.227043232336545`。实际残差按 `rhs-damped(step)`、共同 Torch norm 复核；不把递归 norm 与 actual norm 混成两个不同科学门。`converged=True` 是 PCG 停止语义，不是外层 LM accepted、非线性停止或完整配准精度结论。

严格 scope 全满足：PCG2，callback106（上限1004）；evaluate/linearize/gradient/Bending构造/CSC/new H/native0。27 source/input 绑定前后精确；68×2 immutable 逻辑值、rhs/damping/preconditioner 和 flags 前后不变；两个 unit 门都在任何 solve 前通过。两 cache 均独立，packed layout `[2,0,1]`、copy1548288B、scratch `[3,24,28,24]` / stride `[672,28,1,2016]`。

worker through-post-bind/pre-summary `4.10488339792937s`；imports/binding `1.9187008999288082s`，restore/factory `0.08507795631885529s`。首次 unit 的当前 cache/JIT 观察为 `1.2946888022124767s`，候选 unit 为 `0.0023142052814364433s`。两 solver 顺序观察 `0.0617000563070178s / 0.7058983212336898s`，包含不同 JIT 状态，**不计算速度比**。maxRSS436076KiB。这是有限线性诊断，无新 MRI 脑图或完整配准耗时。

## 6. 版本、首差与下一步边界

既有 [materialized H 回放](../fnirt_cpu_current_replay_20261006/README.md) 是正 full-g RHS、strict CSC action 的旧53/new49；本轮是负 half-g RHS、真实 half callback 的旧49/new53，不能视为同一浮点求解系统或用轮数倒转定位唯一原因。

保存 ledger 显示两个 unit 的 direction/action SHA 相同，而首个 PCG callback 的 direction 已不同（当前 `e63284f6…` / 候选 `b3fea96e…`）。初始 direction 的生成先于 rho dot 的结果使用，所以首个 vector 分叉定位到 old inner-floor→reciprocal×rhs 与 candidate division 的预条件路径。此点 inner floor 是否实际改值尚未单独量化；最终差异还包含归约和后续更新，不能把首差写成最终误差唯一原因。

详细假说与一个尚未执行的最小诊断建议见 [NEXT_DIAGNOSTIC](NEXT_DIAGNOSTIC.md)。没有追加 callback、solve、tighter tolerance、direct solve、eigen、官方或 MRI；默认接入仍不批准。原 transport 和退出记录完整，所有元数据按45000B块回收后重验全文件 SHA；未下载数组。状态见 [STATUS_AND_FAILURES](STATUS_AND_FAILURES.public.json)。

## 7. 源码与参考

成熟来源为 FNIT 的 [registration.py](../../src/fnit/fnirt/registration.py)、[optimizer.py](../../src/fnit/fnirt/optimizer.py)、[spline.py](../../src/fnit/fnirt/spline.py) 和 [_normal_cpu.py](../../src/fnit/fnirt/_normal_cpu.py)；17 生产源 SHA 本轮不变。Stage1/Stage2 自有报告与 helper 是恢复来源，FMA 归约候选来自已发布 private CPU validation 叶；本叶不包含官方源码、二进制、数组、许可证或凭据。

算法参考：Hestenes MR, Stiefel E. Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 1952, 49:409–436。对应原软件内部实现范围见既有 FNIRT 功能说明和前阶段 source bindings；本轮不重跑官方参考。
