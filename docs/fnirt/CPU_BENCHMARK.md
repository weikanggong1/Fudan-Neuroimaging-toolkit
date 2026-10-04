# FNIRT CPU 官方对照：覆盖范围与复现

[功能、参数和 Python/CLI 用法](README.md) · [统一配对脚本](../../tools/benchmark_multimodal_cpu.py)

## 输入、输出与计时

输入为完整真实 3D T1、GM 或 FA，目标模板、可选 FSL scaled-mm 仿射矩阵和目标网格二值掩膜。
每例保存 `cout` 系数、`iout` 重采样图和 `jout` 非线性 Jacobian；可另存包含 affine 的完整 pull Jacobian。
影像与输出保持 float32，既有优化器系数与法方程保持 float64。

历史配对测试在 nodecw10 使用相同 CPU 亲和性和 1/8 线程预算。新采样与 SCG 候选 v27 改在 nodecw8 做完整输出检查及新的官方配对；两节点的时间分别记录，不能合并成同一配对。最新功能覆盖各记录一次完整观察，明确标 `single_observation`；旧源码重复协议及已完成 clock 保留版本来源，不用作最新源码的重复统计。
端到端包含导入、读取、完整优化和全部声明输出的保存；已导入 API 的读取、计算、保存另列。
官方程序的实际 CPU 时间和线程利用率须实测，设置 8 线程上限不等于程序实际使用 8 线程。
不得缩小网格、降低 `miter`、删掉采样级别或减少输出以满足速度目标。

### 在新进程复现线程预算

Numba 的池上限须在首次导入前设置。下面的完整调用同时限制 PyTorch、Numba 和 BLAS；CPU1 将四个环境值及 `torch.set_num_threads` 均改为 1。调用方另按配对清单设置相同 CPU 亲和性。

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python - <<'PY'
import torch
torch.set_num_threads(8)          # 与上述 Numba/BLAS 上限相同。
torch.set_num_interop_threads(1)  # 与本次独立 benchmark worker 相同。
from fnit.fnirt.standalone import run_fnirt

result = run_fnirt(
    input="moving.nii.gz", reference="reference.nii.gz",
    affine="moving_to_reference.mat", config="default",
    cout="warp_coeff.nii.gz", iout="warped.nii.gz", jout="jacobian.nii.gz",
    device="cpu", execution="optimized",
)
PY
```

只执行 `torch.set_num_threads(8)` 不能保证使用本次并行 CPU helper：当 `NUMBA_NUM_THREADS` 池上限大于 PyTorch 预算时，法方程、平滑和弯曲能量 helper 保守使用串行分支；已经导入 Numba 后再修改环境不能替代上述新进程设置。生产函数不自行修改全局线程预算。

现场完整 `--help` 和选项解析均确认 FSL 6.0.7.4 `fnirt` 不支持 `--nthr`、`--threads`、
`--nthreads`、`--num_threads`。因此设置公共线程环境与相同亲和性，并记录进程 CPU 时间；
不把环境上限写成原程序实际线程数。旧 `0,4,…,28` 核组发现另一 FNIT 作业重叠，
旧时间保留为共享核组观察；FNIRT/InvWarp 正式补充使用 NUMA2 `34,38,42,46,50,54,58,62`，CPU1 取 `34`。
同组配对串行并共用 `fnirt-invwarp.cpu-timing.lock`，其他功能可在互不重叠的核组计算。
私有完整 T1/GM/TBSS 及参数覆盖分配到已释放的 Space 组 `66,70,74,78,82,86,90,94`、FLIRT 组 `33,37,41,45,49,53,57,61` 和 ConvertWarp 组 `65,69,73,77,81,85,89,93`，CPU1 取各组首核，同组共用其原计时锁。每个完整配对内两方使用同一亲和性，各组互不重叠。节点仍有其他不限核研究作业，结果是共享节点条件下的观测。

## 功能覆盖矩阵

| 完整真实数据 case | 预设与优化功能 | 必须检查的输出/指标 |
|---|---|---|
| T1 无配置默认值 | 4 级、LM、全局多项式及乘性 bias、隐式零掩膜、10 mm 样条 | 三输出，脑内重采样 Pearson/MAE、支持区 Dice、系数与 Jacobian |
| T1 六级 | 官方 T1 schedule、5 个强度系数、50 mm bias、末级固定强度、显式 reference mask | 同上；全部 6 级与完整迭代预算 |
| 真实 GM | `global_linear`、关闭隐式掩膜、前三层关闭/末层开启 reference mask、Jacobian 0.2–5 | 三输出；真实 GM 网格和组织支持区 |
| 真实 FA/TBSS | 6 级、前四层 LM、后两层 SCG、三个进程的 coefficient/intensity 交接、10→2 mm warpres | 三输出；全部 s1/s2/s3、50/25 迭代预算，含 affine Jacobian 可选 |
| 固定真实输入的受支持参数变体 | identity/显式 affine、LM/SCG、隐式掩膜开关、reference mask 开关、强度模型/阶数、warp/bias 分辨率、lambda/ssqlambda、Jacobian range | 每项实际 case 和已解析配置须进入 manifest；未运行项标 `not_run` |

参数覆盖依据当前 `FNIRTConfig` 字段和独立 CLI；不能把上述四个默认预设运行完成称为所有参数组合均已验证。
本轮正式速度使用 `execution="optimized"`；`reference` 执行分支保留作数值回归，未另做完整官方 CPU 计时。
`arbitrary .cnf`、输入 mask、通用外部 `inwarp/intin`、DCT、二次样条、局部强度模型、membrane energy、`refderiv`、`refout/intout/fout` CLI 当前未实现，标 `unsupported`。
TBSS 预设内部三阶段交接已实现，不能据此宣布通用外部 `inwarp/intin` 接口可用。
原生参照 adapter 的多进程交接只覆盖首进程估计强度、后续进程固定强度的日程，包含当前 TBSS 预设。自定义 `process_stages` 若在后续进程再次开启 `estimate_intensity`，adapter 会明确拒绝，避免重复使用首阶段的旧强度文件；这类任意分阶段日程没有纳入官方对照覆盖。
`SCG + 当前层 estimate_intensity=True` 的联合求解尚未实现；SCG 案例只覆盖已支持的固定强度层，
不把原软件可以执行的组合误列为 FNIT 已验证功能。

## 配对 adapter

[`benchmark_multimodal_cpu_fnirt.py`](../../tools/benchmark_multimodal_cpu_fnirt.py) 提供 `run_case`、`reference_command`、`reference_outputs`、`compare_case`。
`preset` 选择 `default/gm/t1/tbss`，`overrides` 对应现有配置字段；不会修改采样或迭代数。
`accuracy_mask` 可单独指定精度统计脑区；否则使用 `refmask`，再否则统计 reference 的正值前景。
没有显式脑掩膜时，该区域称“reference 正值前景”，不冒称脑内。

```json
{
  "id": "fnirt_t1_subject01",
  "adapter": "tools/benchmark_multimodal_cpu_fnirt.py",
  "input": "/absolute/path/T1_brain.nii.gz",
  "reference": "/absolute/path/MNI152_T1_2mm_brain.nii.gz",
  "affine": "/absolute/path/T1_to_MNI.mat",
  "refmask": "/absolute/path/MNI_mask.nii.gz",
  "preset": "t1",
  "overrides": {},
  "full_pull_jacobian": false
}
```

官方命令从已解析配置逐项传参，TBSS 拆为原软件所需的三个完整 FNIRT 进程，保留必须的磁盘交接。
官方软件仅在隔离 benchmark 运行，生产 API 不调用 FSL。
FNIRT 官方线性插值参数为 `--interp=linear`；它对应 FNIT 的 trilinear 采样。
首轮公开 CPU 参照发现 adapter 曾误用 `trilinear` 拼写，原版解析失败，现已修正；
该失败尝试保留，未计入任何中位时间或精度统计，生产 FNIRT 算法和 CUDA 路径没有改动。

六级原版 T1 现场还发现 FSL 6.0.7.4 的默认 `applyinmask` 只有四项，初始解析抛出 `Mismatch between --subsamp and --applyinmask options`，但该二进制的 stderr 打印乱码。独立 GDB 在初始异常处核对了字符串；adapter 对每个原生进程显式传与其级数相同的全 1 `--applyinmask`，保持官方原本启用的默认，且不提供新的输入 mask。TBSS 每进程 `--minmet` 为单一 `lm/scg`，不是 CSV；通用同进程混合方法不能冒称与原版同一命令。完整日程、warp 分辨率和迭代数没有修改。失败参照 clock 不计入数值表。

GM 的末级 SCG 分支显式使用 `process_stages=(1,1,1,2)`：前三层完整 LM，末层完整 SCG；双方都在该边界交接 float32 系数与 10 位强度参数。原生工具 v4 在入口拒绝没有对应进程边界的混合 minimizer，逐进程保存 wall/user/system，计时结束后记录系数和强度文件 SHA。原 v3 的未声明边界尝试被拒绝，失败 clock 不进入结果；其他已完成单 minimizer 分支保留其实际工具版本。

修正后的原生工具独立冻结并记录 SHA；[`benchmark_fnirt_cpu_reference_adapter.py`](../../tools/benchmark_fnirt_cpu_reference_adapter.py) 允许原生命令生成使用该工具目录，同时候选计算仍从声明的不可变 runtime 源码启动。该包装器不改计时范围或图像输入。旧失败协议与产物保留版本来源，不覆盖为成功记录。

## 2026-10-04 CPU 修改

CPU Gaussian 平滑将每个体素的 offset 循环融合到一次执行，复用两个图像缓冲；
每个 offset 仍按原顺序进行 double 乘加后写回 float32，float64 输入保持 double 存储。
没有改用普通卷积，也没有改变线程设置、平滑核、边界和停止条件。
CUDA 仍使用原 Triton/参考分支；需要梯度的 CPU 张量仍保留可微参考实现。
专项回归覆盖 FP32/FP64、多通道、非连续布局、边缘、极小值、有符号零、masked smoothing 和线程预算。

CPU bending 能量逐场保留原 dense 展开、乘法、平方、sum 和六项相加顺序，
只让一个导数场存活，并在该场上复用乘法/平方存储；没有改成 Gram-dot 能量。
这避免六个包含完整样条支持区的大场同时存活。CPU 梯度调用、`reference` 与 CUDA 保留原实现。
v19 进一步将乘常量和平方融合为单次遍历，仍显式保留每一步 FP32/FP64 舍入；只改变逐元素遍历，不更改原 tensor.sum 或项间相加顺序。

CPU Jacobian limiter 先筛选超出范围的角点索引，再按原 z→y→x→corner 顺序更新。
筛选保留原标量 double 阈值比较；NumPy determinant、alpha 的逐步 float32 舍入、
相邻角点共享导数的更新次序、FFT 和迭代条件均不变。CUDA 仍走原 Triton/回退分支。
专项回归核对阈值相邻浮点值、NaN/inf 和重叠角点的全部位模式，真实输入另做完整输出验收。

CPU `optimized` 法方程另将逐体素的 3×3 FP64 乘加合并为一次循环，并复用每次线性化的 dense 缓冲。
每行仍依次加三个空间项，再加可选强度交叉项，最后除以 mask 体素数；禁用 fastmath，保持原舍入顺序。
循环按实际场的连续维度遍历，保留 separable einsum 生成的 Y 连续布局和原 adjoint 输入布局。
系数空间投影、PCG 迭代和停止条件不变。完整 deformation、global scale 与 T1 联合强度的矩阵乘回归逐位一致。
这条分支使用环境已有 Numba，延迟到 CPU 无梯度调用才导入；CUDA、`reference` 与需要梯度的 CPU 调用保留原路径。
不改调用方的线程设置；Numba 默认池大于 PyTorch 预算时使用串行内核，也不初始化超预算的池。
该完整候选已通过下述完整真实输入的旧版/新版输出检查；正式核组的官方速度检查另行记录。

最新 CPU 候选在每次新 linearization 内，将固定 weight/cross 数组逐位复制为 dense 场的连续空间布局，后续 PCG 复用。几何在缓存命中前核验，每个算子独立持有常量，不跨层复用；已有连续布局无需复制。逐元素 FP64 运算次序、投影和 GPU 支路不变。完整 CPU profile 单列各层额外复制字节数与进程峰值 RSS；布局专项、线程预算、完整矩阵乘及实际 CUDA 路径回归共 136 项通过，正式完整数据门槛另行记录。

后续 CPU normal SIMD 仅对无重叠的 dense 物理视图，每次加载三个通道与九个 FP64 权重的八个体素，再按原顺序执行乘加、可选 scale 项及除法，最后保存三通道。显式 LLVM 向量运算不启用 fastmath 或 FMA；所有源数据在写入前加载，缓冲区别名安全。尾部、非连续场及非有限值块走原 scalar 算式，CUDA 和可微路径保持原分支。105 项专项回归覆盖布局、1/8 线程、完整矩阵乘、有符号零、溢出/下溢和 NaN payload；该改动没有新增低精度存储。

候选 v27 将 CPU float32 三线性采样的值、有效 FOV 和所需体素梯度合并计算，仍保留原 y→x→z 乘加顺序、每步 float32 舍入、零填充及 `1e-8` 边界条件。不能复用 FLIRT 的不同插值算式。需要梯度、forward AD、`torch.func` 变换和不适用的输入回到原张量分支；CUDA 不导入这条 CPU helper。SCG 的 CPU 梯度另外跳过调用方丢弃的 bending energy 与总 cost；真实 cost、SSD、有效 lambda、最终评估和 CUDA 分支保持原计算。83 项采样专项和 33 项 SCG 专项通过，完整预设和参数功能的输出检查及新官方配对另行记录，不能由专项测试推断已完成真实速度目标。

[`profile_fnirt_cpu.py`](../../tools/profile_fnirt_cpu.py) 读取相同完整 case，统计展开、投影、采样、平滑、
bending、线性化、Jacobian 检查、limiter 和 FFT 的调用次数及累计时间。
各时间包含子函数，不能相加；instrumented profile 不进入配对速度表。
诊断使用单独的共同锁和正式组以外的核，节点与 NUMA 差异只用于定位热点，不作官方性能结论。

### 已完成的完整轨迹检查

公开 ds000114 T1 的完整 default schedule 在 CPU95、1 线程共同诊断锁下执行，
包括输入读取与三个 NIfTI 的保存。候选 v17 的 `cout/iout/jout` 完整文件 SHA-256
均与优化前 v9 相同；v19 与 v20 的三个完整文件 SHA 也相同，调用次数和数值轨迹保持一致。最新函数专项回归共 216 项通过，
其中包含本地 RTX 3060 的既有 CUDA 路径测试；H100 性能另做同源验收。
[无路径数值与源码哈希报告](assets/cpu-profile-20261004.public.json)记录全部热点。

| 完整真实 profile | v9 原 CPU 路径 | v15 平滑/能量/法方程复用 | v17 角点筛选 | v19 单遍逐元素能量 | v20 固定 weight 布局 |
|---|---:|---:|---:|---:|---:|
| instrumented API（含读写） | 337.458 s | 271.397 s | 202.685 s | 165.238 s | 183.613 s |
| Gaussian，4 次 | 44.190 s | 11.468 s | 11.690 s | 10.857 s | 11.454 s |
| bending energy，210 次 | 96.282 s | 47.096 s | 55.396 s | 35.165 s | 44.617 s |
| Jacobian 约束，4 次 | 83.926 s | 87.914 s | 11.110 s | 7.273 s | 9.529 s |
| 完整三个输出文件 SHA 与 v9 一致 | 基线 | 是 | 是 | 是 | 是 |

以上为各一次完整诊断，函数钟互相包含，不可相加；CPU95 所在 NUMA 与正式 CPU34 组不同。
该表用于说明实际热点及数值保持，不计算与官方程序的速度倍数，也不代替正式配对统计。
v20 法方程 1,755 次调用累计 36.624 s；50 个线性化算子的布局准备累计 0.686 s。单算子额外固定数组最大 64,989,288 字节，进程峰值 RSS 为 1,513,148 KiB（约 1.44 GiB）。与 v19 的整次 profile 钟受共享节点负载影响，不能据这两次诊断宣布端到端提速；[v20 原始数值报告](assets/cpu-profile-v20-20261004.public.json)保留源码哈希、每层复制记录和输出校验。

### 六层 T1 的独立热点诊断

同一完整真实六层 T1 在 CPU95、1 线程运行源 v20，保存系数、重采样、非线性和完整 pull Jacobian 四图。四个完整文件 SHA 与正式 CPU66 单次输出一致；峰值 RSS 为 1,371,236 KiB，包含完整求解和读写的 instrumented API 为 362.182 s。[聚合报告](assets/cpu-profile-t1-v20-20261004.public.json)保存源码 SHA、调用计数和四图校验，不公开个体影像或路径。

| 六层 T1 的函数范围 | 调用次数 | 累计 inclusive 时间 |
|---|---:|---:|
| 联合形变/强度矩阵乘 | 7,247 | 303.814 s |
| 空间 3×3 法方程乘积 | 7,247 | 158.905 s |
| 系数反投影 | 20,559 | 48.093 s |
| 系数展开 | 14,107 | 23.367 s |
| 三线性采样 | 201 | 20.509 s |
| bending energy | 200 | 18.364 s |
| Gaussian 平滑 | 6 | 12.025 s |
| Jacobian 约束 | 6 | 0.458 s |

这些函数钟互相包含，不能相加；该六层 T1 与上面的公开四层 default 是不同 case。空间法方程乘积仍是具体 CPU 热点，不能以 default 已达到单次速度目标替代六层 T1 的验收。

### strict FP64 SIMD 的完整数值门槛

纯 1D 遍历试验没有带来完整 T1 收益，未采用。随后 explicit vector8 路径在同一 CPU95 核的完整 default 与六层 T1 检查中，三个/四个完整输出文件 SHA、数值 QC、PCG 停止记录和调用次数均与固定源 v20 相同；只排除 QC 中实际运行钟 `elapsed_seconds`。[完整聚合门槛报告](assets/cpu-normal-simd-v2-gate-20261004.public.json)记录两文件源码 SHA、额外布局内存和完整校验。

| 完整诊断范围 | 固定源 v20 | strict FP64 SIMD v2 |
|---|---:|---:|
| default instrumented API（含三图读写） | 156.358 s | 150.129 s |
| default normal，1,755 次 | 35.213 s | 5.307 s |
| 六层 T1 instrumented API（含四图读写） | 359.173 s | 222.051 s |
| 六层 T1 normal，7,247 次 | 160.306 s | 18.018 s |
| 六层 T1 峰值 RSS | 1,363,176 KiB | 1,367,020 KiB |

每个算子额外布局复制最大均为 64,989,288 字节，新增物理 flat 数组只引用既有存储。上表为完整 instrumented 观察，不含新的原版 clock；新版本的官方单次配对另行记录。

### 历史 normal SIMD v2 的 default 与六层 T1 完整配对

严格 FP64 SIMD v2 源分别完成 default 和六层 T1 的 CPU1/8 官方配对；每项完整读取、求解和保存三图/四图，计时范围没有删减。[历史正式无路径报告](assets/cpu-default-t1-normal-simd-v2-20261004.public.json)保存源码、工具、输入和输出 SHA，逐图误差、脑区指标、CPU 利用率及 RSS。该表在 nodecw10 运行，不能与下面 nodecw8 候选时钟构成旧/新速度比。

| 完整预设 | 线程上限 | 原版完整进程 | FNIT 完整进程 | FNIT 已导入 API（含读写） |
|---|---:|---:|---:|---:|
| default | 1 | 185.953 s | 141.044 s | 138.928 s |
| default | 8 | 182.828 s | 65.595 s | 63.571 s |
| T1 六层 | 1 | 248.883 s | 213.245 s | 210.541 s |
| T1 六层 | 8 | 251.124 s | 138.275 s | 136.047 s |

四项均为共享节点上的一次完整观察。default 两种预算的完整三图 SHA 与 v20 相同；T1 CPU1 的完整四图与固定 v20 轨迹相同。与独立官方求解器的图像/Jacobian 差异仍分别记录，不能以 CPU 优化前后的一致性代替官方精度。

本轮真实 CPU 官方精度、端到端/分步骤耗时和 GPU 性能回归由统一报告记录；在报告完成前不把局部测试作为真实 benchmark 或加速证据。
先前真实 FSL 差异见[历史验证页](../../validation/fnirt/README.md)，仍须单列，不能归为本次 CPU 缓冲优化。

### 最新 v27：nodecw8 的八项主要完整配对

[主要配对报告](assets/cpu-primary-v27-node8-20261004.public.json)保存冻结 v27 的八项新完整官方配对及对应完整输出检查。default 为已核验的公开 CC0 T1；其余真实病例只公开聚合数值。每项两方同一 CPU 预算和亲和性，完整网格、层数、迭代预算和三张/四张声明图像均保留。GM 末层 SCG 包含两方完整的进程边界交接。[完整功能报告](assets/cpu-functional-v27-node8-20261004.public.json)另收齐 13 项预设/参数分支的 26 项完整候选输出检查；八项主要功能有新官方时钟，其余分支保留原精度参照，不将旧时钟算作新配对。

| 完整预设/功能 | 线程上限 | 原版完整进程链 | FNIT 完整进程 | FNIT 首次已导入 API（含读写） | 本次速度目标 |
|---|---:|---:|---:|---:|---|
| default | 1 | 181.041 s | 149.123 s | 146.556 s | 达到 |
| default | 8 | 172.209 s | 57.680 s | 55.674 s | 达到 |
| T1 六层 | 1 | 224.018 s | 293.713 s | 291.493 s | 未达到 |
| T1 六层 | 8 | 221.801 s | 169.214 s | 166.933 s | 达到 |
| TBSS 六层、三个进程阶段 | 1 | 836.237 s | 795.217 s | 793.072 s | 达到 |
| TBSS 六层、三个进程阶段 | 8 | 831.938 s | 400.189 s | 398.292 s | 达到 |
| GM 四层、末层 SCG | 1 | 548.107 s | 143.569 s | 141.424 s | 达到 |
| GM 四层、末层 SCG | 8 | 556.218 s | 63.277 s | 60.727 s | 达到 |

均为共享节点上的单次完整观察，没有 warmup 或稳定中位数。线程上限 8 的原版实际平均使用约一个 CPU 核，FNIT 实际使用量在报告内单列。T1 单线程的 FNIT user/system 为 277.890/13.699 s，平均约 0.993 核；该差距不能归为大量非 CPU 等待，也不能用历史较快结果替代。首次 API 仍包含在函数内发生的 feature import、缓存装载及必要编译；现有采样磁盘缓存早于本项完整检查，命中/回退和编译的精确贡献由独立诊断记录。

每个预算的完整文件、数组、header 和 affine 均与该预算旧 normal SIMD v2 输出一致；旧完整输出 adapter 未保存 QC/停止轨迹，因此这里不宣称旧/新整个流程的 QC/停止记录相同。官方数值差异仍存在：T1 的脑内图像相关系数约 0.9991–0.9992，TBSS 约 0.9990–0.9997，公开 default 约 0.956，GM 末层 SCG 约 0.971–0.974。完整逐图与脑区误差见报告，不能仅凭相关系数认定与原版数值等价。

### v27 的完整功能输出验收

13 项功能各在 1/8 线程上限完成完整运行，共 26 项：公开 default、默认 identity 且无显式 mask、六级 T1、强度阶数 2/3/4、各向异性 warp/bias 分辨率、GM 预设、GM 隐式 mask、GM `ssqlambda=0`、GM 未压缩输出、GM 末级 SCG 和 TBSS 三进程完整日程。三张或四张全部声明图像均保留，共 102 个保存文件。

[独立源码、输入和文件复核](assets/cpu-functional-v27-node8-source-input-output-audit-20261004.public.json)重新读取这些完整保存文件，核对旧新 SHA、数组位模式、shape、dtype、header、affine 和有限值；全部通过。每项两预算的原输入和参数合同相同，八项新官方配对的输入 SHA 与对应输出门相同。26 项新完整 QC 均存在；旧整个流程 QC/停止轨迹缺失的范围仍明确标为不可用，不能补称相同。真实末阶段轨迹和完整 T1 旧新 QC 的额外证据见下两节。

验收绑定注册器 `caab8ffccfd47225018d15803e0acee70c05b0212a22a4b720b034454c75e4da` 和 CPU helper `a7e10ff4cc33edecc17b6ca378a5cfabeeb8f1fd8c41213efa06a1d16ae62368`。当前组合源 v28 保留这两个文件的相同字节；没有据此宣称全部 CPU 速度目标或官方数值等价成立。

组合源 v28 保留这两个 FNIRT 文件字节。[最新 H100 完整六级 TBSS 回归](../../validation/multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json)收齐四个进程的 16 次完整保存，三阶段交接和 `(5,5,5,5,50,25)` 原迭代预算均保留。所有四图的完整值、header/extensions、affine 逐位一致；对应 warmup/重复位置的 allocation 两版相同，warmup 为 3,564,715,520 B、三个重复各为 3,565,562,368 B。共享 H100 两组 API 中位数为基线 14.866/13.365 s、当前 14.854/14.102 s；原 v27 的 TBSS partial 与 setup 失败单列历史，不能把这两个独立协议合成一组时钟。

| H100 完整读写 API | 优化前 FNIT，两组 warm+3 中位数 | 当前 FNIT，两组 warm+3 中位数 | 两版对应调用的峰值 allocation |
|---|---:|---:|---|
| [default 四级，v27](assets/cuda-final-v27-20261004.public.json)；注册器/helper 与当前 v28 相同 | 16.945 / 16.702 s | 16.737 / 16.868 s | warmup 1,181,705,216 B；三个 measured repeat 各 1,181,704,704 B |
| [TBSS 六级/三阶段，v28](../../validation/multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json) | 14.866 / 13.365 s | 14.854 / 14.102 s | warmup 3,564,715,520 B；三个 measured repeat 各 3,565,562,368 B |

每例四个进程各执行一次完整 warmup 与三次测量，共 16 次完整保存；default 核对三输出，TBSS 核对四输出。GPU 对照优化前 FNIT，CPU 官方表对照 FSL，不混用两个参照或范围。H100 默认 TF32、allocation 上限 20 GB；测量来自共享设备，不将上述波动称为稳定提速。

### T1 单线程的同节点旧/新诊断

为判断 T1 单线程差距是否来自新采样器，在 nodecw8 CPU32 使用同一完整 T1、原六级预算和四图输出，分别执行冻结 v26 与 v27 各一次完整已导入 API。[独立聚合报告](assets/cpu-t1-sampler-diagnostic32-node8-v27-20261004.public.json)记录了实际采样命中与编译缓存；这次没有运行原版，不替代上表官方时钟。

| 完整 T1 诊断范围 | v26 | v27 |
|---|---:|---:|
| instrumented API，含四图读写 | 333.439 s | 283.067 s |
| 三线性采样，201 次 | 23.111 s | 4.559 s |
| 空间法方程，7,247 次 | 32.835 s | 22.990 s |
| 系数反投影，20,559 次 | 63.348 s | 65.748 s |
| 系数展开，14,107 次 | 63.847 s | 45.688 s |

v27 的 201 次采样均命中融合 helper，回退为 0；唯一已用 Numba signature 发生一次磁盘 cache hit，cache miss 为 0，未新增编译。Torch/Numba 均为 1 线程。全部四图文件、数组、header、affine 及去掉实际运行钟后的完整 QC/停止记录相同。真实证据不支持采样回退、重编译或采样变慢的解释；反投影与其他 PCG 算子仍占用计算时间。上述各 inclusive 钟互相包含，不能相加；一次诊断不能消除正式 T1 单线程未达速度目标的事实。

### v27 的完整 SCG 阶段检查

[nodecw8 聚合报告](assets/cpu-scg-cost-skip-stage3-node8-20261004.public.json)使用同一真实 TBSS 原版第二阶段 checkpoint，执行完整 `182×218×182` 网格及原定 25 次末阶段 SCG 预算。两方都使用已逐位核验的 CPU 采样 helper，参数按生产 `_pack/_unpack` 的 FSL x-fastest 顺序排列；只比较保留或跳过被梯度调用方丢弃的代价计算。该诊断包含最终评估和保存，不是整个六级 TBSS 流程，也没有新的官方速度比。

| 完整阶段诊断 | 保留梯度中未使用的代价 | 跳过该代价 |
|---|---:|---:|
| 求解及最终评估 | 595.556 s | 423.703 s |
| bending energy 调用数 | 78 | 27 |
| bending energy 累计 inclusive 时间 | 324.621 s | 130.063 s |
| gradient / normal 调用数 | 51 / 51 | 51 / 51 |
| 峰值 RSS | 1,900,724 KiB | 1,898,356 KiB |

全部八组完整张量 SHA、25 次接受迭代、停止状态、cost 历史、每次 cost 的 SSD、每次 gradient 的 SSD 与有效 lambda 相同。inclusive 时间互相包含，不能相加。早期诊断曾误用 C-order flatten/reshape，参数与 gradient 的顺序不对应；该版本已保留为无效诊断，不计入这里的轨迹或性能结论。v27 的整个 TBSS、其他预设和参数功能以完整输出检查及 nodecw8 新官方配对验收。

## 公开 CC0 T1 补充与脑图

[`benchmark_fnirt_invwarp_cpu_public.py`](../../tools/benchmark_fnirt_invwarp_cpu_public.py)
只接受仓库 `examples/data/sub-01_T1w.nii.gz` 的已公布 SHA-256。
输入来自 OpenNeuro ds000114 v1.0.2，CC0 许可已按原数据集元数据核验；使用去面部衍生文件，
保留完整 `256×156×256` 网格。官方 FLIRT 的完整 12DOF/corratio 配准生成共享 affine，
其准备时间单列；FNIT 与官方 FNIRT 都使用完整 default schedule 和相同 `cout/iout/jout` 输出。
目标为现有 FSL MNI152 2 mm 全头模板，精度统计使用其脑掩膜。模板本身不随本轮报告再分发。

```bash
# candidate-root：冻结候选源码；baseline-root：含已校验公开示例的固定基线。
# fsl-dir：隔离参照安装，仅 benchmark 使用。
# output-dir：新的完整产物目录；lock-file：FNIRT/InvWarp 组内计时共用的锁。
python tools/benchmark_fnirt_invwarp_cpu_public.py \
  --candidate-root /absolute/path/frozen_candidate \
  --baseline-root /absolute/path/frozen_baseline \
  --fsl-dir /absolute/path/FSL \
  --output-dir /absolute/path/new_public_run \
  --cpuset 34,38,42,46,50,54,58,62 --threads 1,8 \
  --repetitions 3 --inverse-single-observation \
  --lock-file /absolute/path/cpu-timing.lock
```

FNIRT 每个预算先完整 warmup 一对，再以 AB/BA 顺序执行三对；不同时运行两方。
这里的 `--inverse-single-observation` 将原版耗时较长的完整 native 反场各预算记录一次，
保留完整求解和全部读写，不统计中位数。
`report.public.json` 只保存指定数值字段、源码/输入 SHA 和许可来源，原始路径及输出留在私有清单。
`plot_fnirt_invwarp_cpu_public.py` 对同一公开输入再次检查 SHA，读取已完成最后一对输出，
绘制 warped T1、非线性 Jacobian、差异和固定官方前向场的反场差异；不重新配准、不引用私有病例脑图。

```bash
# public-run：上述完整配对成功后的目录；public-input：同一已校验的公开 T1。
# threads：选择已完成的预算及其最后一对；output-dir：PNG/PDF 和绘图范围记录。
python tools/plot_fnirt_invwarp_cpu_public.py \
  --public-run /absolute/path/new_public_run \
  --public-input /absolute/path/frozen_baseline/examples/data/sub-01_T1w.nii.gz \
  --threads 8 --output-dir /absolute/path/public_figures
```

绘图只使用主页环境已有的 nibabel、NumPy、matplotlib。双方共用标尺；差异显示上限和切面位置写入
`figures.public.json`，按各向异性体素的物理尺寸显示，不做额外重采样。
`--fnirt-suite /absolute/path/latest_complete_suite.private.json` 可选择同一公开输入的已完成新版 FNIRT 结果；
输入 SHA 和 suite 的输入绑定都会再次核验。完整误差以数值报告为准，不由颜色范围推断数值等价。

## 原软件、许可与参考

- 官方 [FNIRT 参数与算法说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)。
- 原实现 [FSL FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)；相关原始源码及哈希见 [`_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md)。
- Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, TR07JA2 (2007)。
- 派生实现和隔离参考受 [FSL Software Licence 6.0](../../licenses/FSL-6.0.txt) 约束；本轮不复制发布参考影像或官方二进制。
