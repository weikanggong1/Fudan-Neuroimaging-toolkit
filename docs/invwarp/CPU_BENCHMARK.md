# InvWarp CPU 官方对照：覆盖范围与复现

[函数完整用法](README.md) · [统一配对脚本](../../tools/benchmark_multimodal_cpu.py)

## 输入、输出与功能矩阵

输入是完整真实配准产生的 dense 位移或 FSL intent-2007 三次样条系数，以及反场输出的原始影像网格。
返回/保存 `reference.shape + (3,)` 的 float32 NIfTI；几何沿用 reference，相对位移 intent=2006，绝对场 intent=0。

| 输入 | 约定/功能 | 对照输出与指标 |
|---|---|---|
| 真实 TBSS 或 T1 dense 相对场 | 显式 relative、intent-2006 auto、relative 输出 | 完整反场；指定脑掩膜内向量误差 mean/median/p95/max mm |
| 相同真实场的 absolute 表示 | 显式 absolute、absolute 输出 | 相同网格反场；所有几何/有限值、forward∘inverse 残差 |
| 真实 FNIRT intent-2007 系数 | auto/显式 relative；嵌入 affine；relative/absolute 输出 | 系数展开、affine 求逆、完整反场及脑内残差 |
| 相同真实输入的约定组合 | relative→absolute、absolute→relative | 任何 absolute 输出追加官方 `convertwarp --rel --absout`；relative 输出直接保存原版反场；完整链计时 |
| 相同真实输入的完整迭代预算 | `iterations`、`tolerance_mm` | 保留全部网格和迭代上限；实际迭代数和收敛状态 |

FNIT 是全局仿射初始化的 fixed-point 求逆；FSL 使用其原生求逆算法、正则化和 Jacobian 约束。
2026-10-04 在 nodecw10 实测的 FSL 6.0.7.4 `invwarp` 不接受 `--niter`，尽管官方网页曾列出这一选项。
因此本版参照保留官方内部默认停止设置，FNIT 单列其 `iterations/tolerance_mm`，不能称相同迭代预算或停止规则。
只有独立核验过该选项的其他程序版本，adapter 才通过 `resources.invwarp_supports_niter=true` 传入该 flag。
FSL `regularise/jmin/jmax/noconstraint` 当前没有 FNIT 对应选项，标 `unsupported`。

## 配对 adapter 与计时

[`benchmark_multimodal_cpu_invwarp.py`](../../tools/benchmark_multimodal_cpu_invwarp.py) 接收以下 case：

```json
{
  "id": "invwarp_tbss_dense_relative",
  "adapter": "tools/benchmark_multimodal_cpu_invwarp.py",
  "reference": "/absolute/path/native_ref.nii.gz",
  "warp": "/absolute/path/forward.nii.gz",
  "accuracy_mask": "/absolute/path/native_brain_mask.nii.gz",
  "warp_convention": "relative",
  "output_convention": "relative",
  "iterations": 30,
  "tolerance_mm": 0.01
}
```

在 nodecw10 固定相同 CPU 亲和性和 1/8 线程预算；完整功能记录包含读取和反场保存。相同完整原生命令的复用只提供精度参照，另列新官方时钟是否存在。
Python 导入/冷进程与已导入 API 时间分列；官方实际 CPU 时间和线程利用率另查。
FSL 6.0.7.4 实际解析不支持 `--nthr/--threads/--nthreads/--num_threads`；线程预算是环境上限和相同亲和性，
不是官方实际使用的线程数。正式公开补充使用 NUMA2 `34,38,42,46,50,54,58,62`（CPU1=34），
受 FNIRT/InvWarp 专组共同锁协调，同组串行；其他功能只使用不重叠核组。
节点其他不限核作业保留，结果标为共享节点观测。
`accuracy_mask` 决定脑内聚合统计；未提供时只称 reference 正值前景。
场内有效位置的 forward∘inverse 残差和场外覆盖比例分别记录。
既有官方数值差异见[真实 DWI 报告](../../validation/invwarp/README.md)，不能用优化前后的逐位一致代替官方精度验收。

私有真实 T1/TBSS 的 14 项功能覆盖接管已完成 ConvertWarp 的 `65,69,73,77,81,85,89,93` 核组，CPU1=65，同组共用原 ConvertWarp 计时锁；公开 T1 对照仍绑定上述 CPU34 组。两组原生输出的缓存签名包含亲和性，不能跨组复用为新参照。

### 14 项真实功能的完整 1/8 线程结果

固定反场源 v17（与 v20 同文件）和修正后的原生工具 v2，共 28 项完整记录已完成：18 项有新官方完整配对，10 项仅复用已核验的精度参照，官方新时钟为空。[无路径完整报告](assets/cpu-functional-v17-corrected-v2-20261004.public.json)逐项保存输入/源码/程序 SHA、输出几何及有限值、脑区误差、composition、API 时间、CPU 时间和 RSS。每次执行为一次完整观察；表中原版 absolute 输出包含完整 `invwarp + convertwarp` 链。

| 完整真实分支 | CPU1 原版 / FNIT（s） | CPU8 原版 / FNIT（s） | 指定脑区 p95 向量差（mm，CPU1/8） |
|---|---:|---:|---:|
| T1 系数，auto → absolute | 78.03 / 55.90 | 76.52 / 19.55 | 0.03361 / 0.03361 |
| TBSS 系数，auto → absolute | 25.20 / 8.51 | 11.78 / 3.93 | 0.00659 / 0.00659 |
| T1 系数，relative → absolute | 复用精度参照 / 52.60 | 复用精度参照 / 16.36 | 0.03361 / 0.03361 |
| T1 dense，auto → absolute | 583.62 / 46.58 | 574.95 / 18.98 | 1.78314 / 1.78314 |
| T1 dense，absolute → absolute | 复用精度参照 / 44.35 | 复用精度参照 / 19.33 | 1.78314 / 1.78314 |
| T1 dense，relative → absolute | 608.23 / 51.73 | 572.81 / 14.59 | 1.78114 / 1.78114 |
| T1 系数，auto → relative | 44.71 / 59.04 | 40.16 / 18.41 | 0.03361 / 0.03361 |
| TBSS 系数，auto → relative | 6.72 / 8.14 | 6.72 / 3.94 | 0.00659 / 0.00659 |
| T1 系数，relative → relative | 复用精度参照 / 50.71 | 复用精度参照 / 17.32 | 0.03361 / 0.03361 |
| T1 系数，最大 60 次 | 复用精度参照 / 98.33 | 复用精度参照 / 31.60 | 0.03360 / 0.03360 |
| T1 系数，未压缩输出 | 37.29 / 50.42 | 35.89 / 15.30 | 0.03361 / 0.03361 |
| T1 dense，relative → relative | 558.53 / 48.52 | 535.00 / 15.75 | 1.78115 / 1.78115 |
| T1 dense，auto → relative | 复用精度参照 / 48.26 | 复用精度参照 / 18.08 | 1.78115 / 1.78115 |
| T1 dense，absolute → relative | 546.52 / 47.89 | 548.21 / 18.23 | 1.78315 / 1.78315 |

所有输出有限且几何一致。relative 系数分支在 CPU1 的完整进程仍慢于原版，未压缩 T1 亦有差距；有新原版时钟的 CPU8 分支均为 FNIT 更快。复用精度参照的行没有新的原版 clock，不能计算新配对速度比。dense 分支的约 1.78 mm 脑区 p95 差异显著大于系数分支，两套求逆算法尚不能称为数值等价；更小的 composition 残差也不能替代这项独立对照误差。

例如 T1 dense auto→absolute 的指定脑区 mean/median/p95 为 1.05007/0.81593/1.78314 mm，最大 311.440 mm，有效场覆盖约 99.83%。完整报告保留这些尾部差异；输出有限和较小的场内残差不能视为严格官方精度门槛通过。

该例的显式脑掩膜包含 1,397,640 体素。场内有效覆盖为原版 99.83794%、FNIT 99.83343%；有效区 forward∘inverse 残差 median/p95 分别为原版 0.75035/1.61167 mm、FNIT 0.00001129/0.00003013 mm。完整网格的逐分量误差另存 `accuracy`，脑掩膜内向量误差另存 `adapter_accuracy`，两者统计范围和单位分别保留。

## 2026-10-04 CPU 修改与 bug 修复

CPU 迭代复用 FP64 残差/矩阵乘缓冲并就地更新坐标，保留原算式、最大迭代数、停止阈值和 float32 输出。
CUDA 路径仍执行原算式。专项回归固定旧求逆循环，覆盖 dense relative/absolute、带 affine 系数、两种输出约定、提前收敛与迭代上限，逐位比较输出及实际停止次数。

原接口允许 NaN/inf `tolerance_mm` 进入迭代，且只记录最大次数。
现在拒绝非有限/非正阈值及非正整数 `iterations`；新增 `qc["iterations_used"]` 和 `qc["converged"]`。
保留原 `qc["iterations"]` 作为最大预算，合法既有调用兼容。

原版参照另修正一个约定错误：FSL 6.0.7.4 的 `invwarp.cc` 只将 `--abs/--rel` 交给输入读取器，最后保存 relative FNIRT 场；系数场还会忽略 dense 输入选择。直接将该输出当 absolute 会产生约 159 mm 的坐标伪差，旧比较已标无效。新协议对所有 absolute 输出运行完整原生求逆，再运行 `convertwarp --rel --absout`；两个进程的 wall/user/system 和读写均保留。该修正只影响隔离 benchmark adapter。

[`benchmark_invwarp_cpu_reused_reference.py`](../../tools/benchmark_invwarp_cpu_reused_reference.py)记录原 runtime 与独立工具冻结的 SHA。只有除输出名外的完整 argv 链、输入 SHA、原程序及动态依赖 SHA、环境、线程预算、亲和性和输出格式全部相同，才复用已完成原生输出。`reference_reused=true` 时官方新时钟为空，只提供精度 oracle；新链不能命中旧错误协议的缓存。

真实 CPU 速度与 GPU 性能回归由统一报告记录；局部 solver 回归不代替端到端 benchmark。

### 单独的采样开销诊断

完整真实 T1 系数反场另在 CPU95、1 线程试验预先保存 CPU 采样源布局，并跳过迭代中丢弃的有效域标志计算；最终有效域仍完整计算。该实验的输出文件、实际停止次数、残差与最终覆盖率和原路径一致。[聚合诊断报告](assets/cpu-skipvalid-diagnostic-20261004.public.json)记录源码 SHA 和完整结果校验。

两次 instrumented API 为 58.330 s 和 56.896 s，峰值 RSS 为 1,679,424 / 1,674,268 KiB。有效域检查从 32 次降为最后 1 次；插值本身的 32 次累计钟为 13.107 / 13.101 s，基本相同。这是共同诊断锁下的完整旧/新观察，既没有新的原版 clock，也不足以关闭该 T1 CPU1 官方速度差距；该实验没有替换上述正式功能队列的冻结源。

后续采用相同源布局复用和最终有效域检查，并对 CPU prepared-source 的 FP64→FP64 坐标归一化保序融合；每一步运算仍为 FP64，F32 和 CUDA 路径保留原实现。历史源 v23 在完整真实 T1 系数、原始网格及未压缩输出的 CPU1/8 官方配对中，每个预算先完整 warmup，再运行三对交替进程；独立热 API 也执行一次完整 warmup 和三次完整读写。[v23 正式报告](../applywarp/cpu_normalization_all23_official_20261004.public.json)记录两方进程中位数 35.917/49.101 s（CPU1）和 35.813/18.293 s（CPU8），FNIT 热 API 中位数 39.208/15.832 s。CPU1 速度差距仍保留，不能将诊断改进称为全面达到目标。

该源的 32 次采样、保存输出与停止状态和旧版一致；[v23 H100 回归](../../validation/multimodal_cpu_20261004/gpu_final_v23_20261004.public.json)核对 16 次完整保存的全部位模式与 metadata，峰值 allocation 不增加。上述 v17 功能矩阵没有被新 clock 覆盖，仍保留原版本来源。

后续源 v26 对 repeated prepared-source 的有限 FP64 坐标、严格对角 FSL scaled-mm 反矩阵及正零平移使用 CPU 静态分支，其余矩阵、布局、可微与 CUDA 保留原乘积。[完整 CPU 旧/新门槛](../applywarp/cpu_static_diagonal_20261004.public.json)分别核对 1/8 线程全部输出文件 SHA、30 次实际停止、solver QC 和有效覆盖；没有改变求逆算法或降低精度。[v26 H100 回归](../../validation/multimodal_cpu_20261004/gpu_inv_mni_v26_20261004.public.json)另核对每例 16 次保存的完整反场值、header、extensions 与 affine，峰值 allocation 保持 1,275,725,824 B；两组基线 API 中位数 3.962/3.902 s，候选 3.935/3.916 s。GPU 时间来自共享 H100，不把短时钟波动写成稳定加速比。

### 历史源 v26 的完整官方配对

[v26 正式报告](../applywarp/cpu_static_diagonal_all26_official_20261004.public.json)在 `35,39,43,47,51,55,59,63` 核组（CPU1=35）完成两种预算的完整 warmup 加三对交替进程，热 API 另完成一次完整 warmup 和三次读取、求逆及保存。输入为同一完整真实 T1 系数，输出未压缩 NIfTI；两方每对使用相同亲和性、线程上限和输入，保存完整 `162×215×180×3` 反场。

| 线程上限 | 原版完整进程中位数 | FNIT 完整进程中位数 | FNIT 热 API 中位数（含读写） |
|---|---:|---:|---:|
| 1 | 35.583 s | 55.584 s | 53.733 s |
| 8 | 35.732 s | 16.542 s | 14.259 s |

CPU1 仍未达到速度目标，CPU8 更快；全部 18,808,200 个坐标值有限且几何相同。相对原版，全网格分量 MAE/RMSE/max 为 0.10954/0.32135/7.02610 mm；指定脑掩膜 1,397,640 体素内向量差 mean/median/p95/max 为 0.01723/0.01008/0.03361/7.31428 mm。原版与 FNIT 的脑区有效覆盖均为 99.83343%；composition 残差与独立算法误差另列，不能据此称严格数值等价。历史 v17 的完整功能矩阵和 v23 的重复计时仍保持原源码来源。

### 最新源 v28：nodecw8 完整官方配对

[v28 正式报告](assets/cpu-allocation-v2-node8-official-20261004.public.json)使用同一完整真实 T1 系数，读取、求逆并保存全部 `162×215×180×3` 未压缩反场。原版和 FNIT 在 nodecw8 的同一核组、相同 1/8 线程上限下完成一次完整 pair warmup 和三对交替进程；已导入 API 另执行一次完整 warmup 与三次完整读写。候选的四个 InvWarp、ConvertWarp、ApplyWarp 与归一化模块 SHA 与组合冻结源 v28 相同；nodecw10 的历史时钟另列。

| 线程上限 | 原版完整进程中位数 | FNIT 完整进程中位数 | FNIT 热 API 中位数（含读写） | 速度目标 |
|---|---:|---:|---:|---|
| 1 | 35.878 s | 50.899 s | 48.571 s | 未达到 |
| 8 | 36.679 s | 21.661 s | 18.441 s | 达到 |

该版减少 CPU prepared-source 的临时分配，保留两步 FP64 舍入与原停止量；旧新完整输出文件、QC 和实际 30 次停止一致。原版实际使用约 0.98–0.99 核，8 是两方相同的资源上限。相对官方的误差与上述 v26 完整系数分支相同，求解器和停止合同仍不同。CPU1 的完整进程和热 API 均未达到速度目标；单次分配诊断不替代这组正式时钟。

[v28 优化与分步骤说明](CPU_ALLOCATION_20261004.md)给出完整参数、命令、源码校验、阶段时钟和 CUDA 正确性范围。v17 的 28 项功能记录保持其原冻结来源，其中 18 项为新官方完整配对，10 项仅复用严格核验的精度参照；不能将它们标为 v28 新官方配对。

## 公开 CC0 T1 反场与脑图

[公共补充脚本](../../tools/benchmark_fnirt_invwarp_cpu_public.py)将已校验的 OpenNeuro ds000114
去面部 T1 经完整 FNIRT 配准；两套反场求解器都读取同一官方前向系数，保持原始 T1 的完整输出网格。
FNIT 的最大 30 次固定点修正和 0.01 mm 阈值与原版内部停止条件分别报告。
本轮公开反场在 1/8 线程预算各记录一次完整观察，包含反场 NIfTI 保存；不称中位数。
CPU1 原先计划 warmup 加三对，因原版完整 native 网格求解耗时较长，改为保留已开始的第一完整对，
结束后取消冗余重复；原协议、取消原因和 partial 日志保留。CPU8 另做官方与最终候选的完整单对。
InvWarp 本身的时间不包含此前 FLIRT/FNIRT，前两阶段单列，不把整条准备链误算为反场耗时。

公共补充脚本的 `--inverse-single-observation` 复现这个完整单次协议；FNIRT 仍按 `--repetitions` 重复。
对于相同官方参照，只有命令除输出名外、输入 SHA、程序 SHA、线程上限和亲和性全部相同才能复用输出。
复用必须记录 `reference_reused`，只作为精度参照，不把历史时钟写成新配对时间。

脑图只从这例公开输入生成，展示原版/FNIT 的反场位移模长和向量差（mm）。
其显示区域为去面部 T1 正值前景，包括邻近颅骨；未提供原生脑掩膜，不冒称“脑内”。
全网格差异、该前景内分位及 forward∘inverse 的有效场内残差在数值报告分别保留。

## 原软件与参考

- [官方 invwarp 参数说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#invwarp)。部分网页描述为输入与输出选择；本次以现场 FSL 6.0.7.4 和[已附原实现](../../src/fnit/_vendor_fsl/sources/fnirt-2203.0/invwarp.cc)核验的 relative 输出为准。
- [FSL FNIRT 原实现](https://git.fmrib.ox.ac.uk/fsl/fnirt)；Andersson, Jenkinson & Smith, TR07JA2 (2007)。
- [FSL Software Licence 6.0](../../licenses/FSL-6.0.txt)。官方程序只用于隔离参考，不作为 FNIT 生产依赖。
