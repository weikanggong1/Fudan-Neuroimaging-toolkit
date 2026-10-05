# FNIRT Stage3：已恢复真实 callback 的有限 PCG 比较计划

## 1. 目标与当前状态

本轮只准备计划、源码和冻结清单，尚未上传、登记、排队或运行数学。目标是比较当前 Torch PCG 与此前独立验证的自有 CPU PCG 候选，在同一个已恢复的真实 matrix-free 系统上的自然停止与实际残差。生产源码、GPU 分支、默认行为和容差均不修改。执行须由协调者另行审查授权。

Stage1 提交 `0b691f32bdc47dc68f30516daf7d4e9f369e3235` 已用一次 evaluate / linearize 桥接当前完整 g、diag；checkpoint 共 68 个数组。Stage2 提交 `e2c13cb75f103e372618cc51189312d2a73c5797` 已恢复字节 stride / 逻辑值，并验证七个方向的 optimized/reference callback 位相同。三列 unit 门及七个方向不代表所有方向与 materialized H 位相同，也不代表 PCG 或完整非线性估计已等价。

## 2. 输入、公式和参数

所有输入以 `expected.public.json` 的大小和 SHA-256 绑定；checkpoint / 原数组留在服务器，不下载。输入限于 accepted solve3 checkpoint、其 producer summary、通过的 Stage2 summary、当前完整 g / diag，以及当前存档 H。H 只检查每臂一个 unit 方向的恢复列；不用于本轮求解。solve3 是第二 accepted 参数；solve2 仅是既有预处理来源的历史说明。本轮不读取 MRI、native 解、native A/RHS、旧 current PCG 解或参考轨迹。

实际生产接口是 `registration.py` 的 LM 求解分支，`linearize` 返回半梯度、半对角线、半 H 的 callback。按同一接口计算：

```python
# checkpoint 中当前自产状态，所有 PCG 向量为 CPU FP64。
gradient = tensors["gradient_half"]
diagonal = tensors["diagonal_half"]

# 本次明确声明的受控 LM 比较值；未证明是历史 solve3 的实际 damping。
tau = 0.001

# 保留生产 floor、乘法顺序和符号，不把完整 H/g 的旧回放直接套进来。
floor = torch.finfo(diagonal.dtype).eps * diagonal.abs().mean().clamp_min(1)
damping_diagonal = diagonal.clamp_min(floor)
rhs = -gradient
preconditioner = (1 + tau) * damping_diagonal

def damped(direction):
    return mature_half_matvec(direction) + tau * damping_diagonal * direction
```

`effective_lambda=9049.463427795125` 是 checkpoint 的弯曲正则化权重，已包含在成熟 callback 内；它与 `tau` 的 LM damping 含义不同。本轮不重算该权重。

| 参数 | 声明值与含义 |
| --- | --- |
| `tau` | `.001`，本次受控 LM 比较值；不声称历史实际值 |
| `tolerance` | `1e-3`，相对未预条件递归残差自然停止 |
| `max_iterations` | `500`，不强制达到旧 69、53 或 49 轮 |
| 初值 | 两臂均为零；不读取任何旧解 |
| `rhs` | 同一 checkpoint 的 `-gradient_half` |
| `diagonal` | `(1+tau)*damping_diagonal`；两臂值逐位相同 |
| callback | 两臂各自独立恢复的同一成熟 optimized CPU half action |
| dtype / device | PCG 向量 CPU FP64 / no-grad；保存的辅助数组按原 dtype 恢复 |

## 3. 两臂与有限门

`current_torch` 直接调用当前 `optimizer.py` 原 PCG（SHA `903d5031…`），CPU 分支仍使用 reciprocal × residual、Torch dot / norm 和原 beta floor。`owned_cpu` 使用已发布的 private `pcg_cpu_candidate.py` 和 `own_reductions.py` 字节相同副本；它使用 division、项目自有 FMA Dot、偶/奇平方归约 Norm 和其原 beta 表达式。比较是完整候选算法差分；不能说只替换 dot / norm。候选不调用或链接安装 FSL / BLAS，不增加依赖；NumPy、Numba、llvmlite 来自主页 Conda 已有链。

两臂均独立恢复 68 逻辑值和记录的字节 stride；原 pointer / alias graph / cache 对象没有持久化。每臂新建一个成熟 `SpatialNormalCPU` cache，初始 packed/scratch 为 None。两臂 checkpoint storage 分离，immutable operand 值前后哈希相同；只有成熟 normal 的 packing/scratch 内部可变。

按以下顺序执行，首门失败即停止，不重试：

1. 预检全部 source / input / freeze；接受 Stage1 / Stage2 的已绑定结论，确认 CPU8、FP64 / no-grad、CUDA 未初始化。
2. 恢复两个独立缓存；每臂 `2*gradient_half`、`2*diagonal_half` 与已有完整数组逐位相同；两臂 rhs/damping/preconditioner 值逐位相同。
3. 每臂仅一个 `unit[0]=1` 的 bare callback：`2*half_matvec(unit)` 必须逐位等于已有 H 的第零列。两个 unit 门都通过后才能求解。
4. old / new 各调用 PCG 一次，按原停止规则自然终止。每次真实 callback 记录方向 / 输出值 SHA 与阶段；最多 500 次迭代 callback。非正或非有限 denominator 的原停止语义保留，不把未接受迭代错算为 accepted iterations。
5. 每臂一次同一 damped callback 计算 `rhs-damped(step)`；两臂统一用 `torch.linalg.vector_norm` 报告 true relative residual。递归 residual 的原归约保持各臂差异。
6. 记录两自产 step 的 bits / max / relative L2；旧 Torch step 只是本轮控制臂。无 native 同系统 oracle，无更新参数、接受/拒绝 LM 步、outer optimizer 或最终配准输出。

上限为每臂 `1 unit + 500 PCG + 1 true residual = 502 callbacks`，合计 1004；PCG 两次；evaluate、linearize、gradient、Bending 构造、新 H、CSC action / solve、native、MRI、GPU 均为零。若 PCG 返回未收敛，则保留自然轮数和 true residual，不自动做更紧 tolerance 或额外 solve。

## 4. 将来的调用与调度

没有原软件的独立同内部状态 CLI；不启动官方 FNIRT。只有协调者批准该 freeze 后，才将自有文件上传新的 canonical leaf，并在六锁内登记；`run_stage3.sh` 是准备入口，目前未运行。其布局为：

```text
workspaces/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1
runs/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1
```

同 nodecw7 八个 physical cores `32,36,40,44,48,52,56,60`，OMP/OpenBLAS/MKL/Numba8、Torch8 / interop1。先公共 CPU lock、再 leaf 内锁；outer wait19000s / child180s / controller19600s，20,000,000,000B address-space cap。保持环境 prefix，清除 loader overrides，GPU invisible；不更改全局 TF32 flag。x86_64 / FMA 只限定本次有证据的候选使用范围，不依据地址、被试或维数选择算法。

worker 入口 `umask(077)`，输出目录700、文件600；禁止已有输出目录后重新运行。runner 无论 science 成功或失败都做 after preflight，并保留原 science RC。worker finally 记录所有已经恢复的 immutable operand / derived system 后哈希、flags / CUDA / DSO snapshots；失败不标 accepted。

## 5. 输出、计时与判断

将来输出 `stage3/summary.public.json`、before/after preflight 和三个 exitcode。两 step 的私密 FP64 文件只公开 SHA / 大小 / mode / schema，不下载或发布数组。summary 分列递归 residual、统一 Torch norm 的 true residual、自然轮数、callback 数、输入 / source / flags 不变和缓存 metadata。

`valid_bounded_diagnostic` 仅代表有限比较完整、有限值和边界门满足。即便两臂报告自然收敛，也不等于数值改善、原软件等价或生产可接入。无官方同系统解，只能说明本次 current 系统上的差异。old→new 单次顺序时钟包含缓存差异、不是 ABBA；不得报告速度比。

## 6. 版本与未执行项

Stage1 单次恢复 / 完整 g,diag 精确桥已通过；Stage2 七方向有限 callback 审查已通过；Stage3 当前仅准备。两候选源副本不改，生产 17 源 SHA 不变。最新 Git HEAD 仅作为现场身份，不能替代源码 / cache / 状态绑定。

不得自动新增 assembly、native run、MRI、direct solve、eigen、tighter PCG、LM 接受步骤或 production patch；结果交协调者后再决定下一步。

## 7. 来源与许可

成熟实现来自本 FNIT 仓库 `src/fnit/fnirt/{registration,optimizer,spline,_normal_cpu}.py`。callback restore helper 逐字复用已通过 Stage2 的自有 helper；候选源来自 `validation/fnirt_cpu_current_replay_20261006/source`。Stage3 不复制官方代码、对象文件或二进制，不读取参考轨迹；此前原生归约只属于隔离 benchmark。数学算法为预条件共轭梯度；本轮只记录两条已绑定实现的真实行为，不更改科学停止门。
