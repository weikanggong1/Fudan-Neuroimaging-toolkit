# FP64 CPU 查询网格归一化：真实逆场诊断

## 1. 功能与采用范围

本次优化共享采样子函数 `_sample_linear` 的 CPU FP64 查询网格准备。真实 FNIRT 系数场展开后仍为 FP64；完整逆场调用该采样器 32 次。融合核逐次保留 `2 * coordinate / (size - 1) - 1` 的 FP64 舍入，按原顺序反转坐标轴，再直接写入 FP64 网格。单体素轴写正零，不改变插值、边界、迭代或停止规则。

仅 CPU、无梯度、连续 FP64 坐标，且重复采样已准备 `prepared_source` 时采用融合核。一次性系数 ApplyWarp/ConvertWarp 保留原 FP64 张量准备。非连续布局、负位视图、其他 dtype、梯度和 CUDA 保留张量路径；普通 float32 网格准备恢复已验证的原实现。Numba 已在主页 Conda 和 Python 依赖中，总计算线程不超过调用方的现有预算。

## 2. Python 调用与输入输出

公共 API 和参数保持原样。输入是完整 3D 输出参考网格，以及 FSL dense pull 场或 FNIRT 三次系数场；输出为参考网格上的三分量反场、有效场内比例及 QC。

复现本页 8 线程 CPU 配置时，在启动 Python 进程前设置线程环境；随后在脚本中设置 PyTorch 预算。环境变量须先于首次 NumPy、PyTorch 和 Numba 导入生效。

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 python /path/benchmark_invwarp.py
```

```python
import torch

torch.set_num_threads(8)  # 与进程启动前的 Numba/BLAS/OMP 上限一致
torch.set_num_interop_threads(1)  # 新进程中在计算前设置；本次 worker 使用该值
from fnit.invwarp import TorchInvWarp

reference_nifti_path = "/path/native_T1w.nii.gz"  # 完整反场输出网格
forward_coefficient_path = "/path/T1w_to_MNI_coeff.nii.gz"  # FNIRT pull 系数场
inverse_output_path = "/path/MNI_to_T1w_warp.nii.gz"  # 三分量相对位移，单位 mm
inverse_model = TorchInvWarp(device="cpu")
inverse_result = inverse_model(
    reference=reference_nifti_path,
    warp=forward_coefficient_path,
    warp_convention="auto",  # 根据系数 intent 解释输入
    output_convention="relative",  # 保存相对位移
    iterations=30,  # 最大固定点修正次数
    tolerance_mm=0.01,  # 最大修正分量停止阈值
)
inverse_result.save(inverse_output_path)
```

全部输入、参数及输出字段见 [InvWarp](../invwarp/README.md)；普通影像重采样见 [ApplyWarp](README.md)。模型和输出 dtype 未因本次优化降低。

FP64 helper 使用调用方的现有 PyTorch 预算。若进程首次导入后固定的 Numba 线程上限高于该预算，helper 采用保守串行路径；本次 benchmark 明确设置 `NUMBA_NUM_THREADS=1/8` 与对应 PyTorch 预算相同。复现 1 线程时，将上述四个环境变量和 `torch.set_num_threads` 一并改为 1。

## 3. 命令行调用

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
fnit invwarp --ref /path/native_T1w.nii.gz \
  --warp /path/T1w_to_MNI_coeff.nii.gz \
  --out /path/MNI_to_T1w_warp.nii.gz \
  --rel --niter 30 --device cpu
```

## 4. 原软件调用

```bash
invwarp --ref=/path/native_T1w.nii.gz \
  --warp=/path/T1w_to_MNI_coeff.nii.gz \
  --out=/path/fsl_MNI_to_T1w_warp.nii.gz --rel
```

FSL 6.0.7.4 不接受 FNIT 的 `--niter`；它使用不同的求逆算法及约束。本页表格比较同一 FNIT 求解器的优化前后，最新原版精度与时钟由 [InvWarp CPU 报告](../invwarp/CPU_BENCHMARK.md)单列。

## 5. 完整真实输入的精度与时间

nodecw10 上使用同一完整病例及相同物理 CPU 亲和性，保持 30 次最大修正、0.01 mm 阈值和完整读写。每项各一次 instrumented API 观察；入口模块导入排除，调用内部的惰性 Numba 导入及 JIT 包含。步骤时间已包含于完整 API，不能重复相加。

| CPU 上限 | 优化前完整 API | FP64 融合完整 API | 优化前 32 次归一化 | 融合 32 次归一化 | RSS 峰值 KiB，前 → 后 |
|---|---:|---:|---:|---:|---:|
| 1 | 51.4119 s | 39.3233 s | 12.8681 s | 3.5055 s | 1,674,496 → 1,745,304 |
| 8 | 18.4087 s | 13.0287 s | 4.2754 s | 1.3804 s | 1,674,264 → 1,747,340 |

两预算均满足输入哈希一致、全部保存文件 SHA-256 一致、QC 和有效场内比例一致，实际运行 30 次且 `converged=False`。完整 MRI 的首个查询网格含 18,808,200 个 FP64 值，1/8 线程逐 uint64 位核对均零差。上述时间对应 v3/v4 快照；最终版本的 FP64 入口和 kernel 与 v4 逐字相同；重复 prepared-source 调用沿用该路径，默认单次 FP64 调用恢复已验证的张量表达式。

普通完整公开 T1 与 105 帧 DWI 在 1/8 线程下各完成优化前后全文件比较，四对 NIfTI 文件和输入哈希均一致，包含所有保存 metadata。工具未捕获普通入口的独立 QC/mask，因此不把这两项列为已通过。最终 float32 准备计算与已验证版本相同；被放弃的 float32 原型时钟保存在 [机器可读报告](cpu_normalization_20261004.public.json)。私有影像仅公开聚合结果；本页不新增单被试脑图，公开脑图例见 [ApplyWarp](README.md)。

### 冻结 all_v23 与官方的完整对照

另在物理核 `35,39,43,47,51,55,59,63` 运行最新原版时钟。逆场排除一组完整 warmup，保留三组交替官方/FNIT 完整进程；FNIT 的已加载 API 另做一次完整 warmup 和三次完整读写。小型 ApplyWarp 各只有一次完整观察，未建立稳定速度比。

| 完整操作 | 1 核官方 / FNIT 进程 | 8 核官方 / FNIT 进程 | FNIT 已加载完整 API，1 / 8 核 |
|---|---:|---:|---:|
| 完整 coefficient 逆场 | 35.9168 / 49.1014 s | 35.8129 / 18.2925 s | 39.2076 / 15.8320 s |
| 公开完整 T1，trilinear float | 2.4926 / 4.5451 s | 1.9962 / 3.0519 s | 未单独重复 |
| coefficient T1，trilinear | 0.4979 / 2.7069 s | 0.4354 / 2.4824 s | 未单独重复 |

1 核逆场和两项小型 ApplyWarp 的完整进程仍高于官方。官方线程上限与 FNIT 相同，但不据此假定 native 程序实际并行使用八核。双方求逆算法与停止规则不同；逆场脑区内向量差的中位数为 0.010075 mm、p95 为 0.033606 mm。有效区域的 forward/inverse composition 残差中位数为官方 0.008819 mm、FNIT 3.80×10^-7 mm。全网格最大分量差仍为 7.0261 mm，不能用脑区聚合误差替代全网格合同。

两项 ApplyWarp 的数值误差与原 50 组报告一致。公开 T1 输出的 dtype、网格、单位与 qform/sform 相同；coefficient T1 输入约定下，FNIT 保留 `unknown/unknown` 单位而官方保存 `mm/sec`，其余所核对的 dtype、网格、qform/sform 和保存缩放一致。报告保留该 metadata 差别。

完整数字、实际 CPU 使用、双方输出合同、版本与源码 SHA-256 见 [all_v23 官方对照聚合](cpu_normalization_all23_official_20261004.public.json)。旧诊断时钟继续单列，不与最新官方进程时间混用。

本地定向回归 160 项通过，新增单次与 prepared-source 的 1/8 线程分派检查；同时覆盖包含 FP64 NaN payload、无穷值、溢出、符号零、单体素轴、非二次幂除数及边界 ULP、1/8 线程、零拷贝布局、梯度、CUDA 路由，以及默认设备为 `meta` 时的显式 CPU 分配。

### all_v23 的短热点诊断与最终收敛

真实 coefficient T1 只进行一次 `[3,91,109,91]` FP64 网格准备，未传入 `prepared_source`。1 核融合准备为 0.5561 s、原张量准备为 0.0366 s；8 核为 0.4186 s、0.1180 s。两预算的完整保存文件均一致。这些是各一次 instrumented 观察；调用内部的首次 helper 导入和缓存载入包含在准备时钟内。最终分派据此仅在重复 prepared-source 查询使用融合核，一次性 FP64 保留原实现。

完整逆场另外经过一次完整 warmup 后进行一次 profile；完整输入、30 次求解与保存均保留，profile 输出与 warmup 的全部文件一致。instrumented API 为 41.7071 s；32 次 `PullField.sample` inclusive 为 33.7386 s，其中采样 body 14.9388 s、归一化 2.8743 s。CPU operator self 时间包括静态矩阵乘法 5.1732 s、平移 add 4.8701 s、base+field add 2.9363 s，以及停止检查 abs 2.2375 s/max 0.6840 s。系数展开仅 0.0441 s。inclusive 时钟彼此包含，不能相加；该 profile 不与原版 CLI 计算速度比。

真实系数场的 `scaled_inverse` 严格对角且没有平移，embedded inverse 有六个非零非对角项和非零平移。短诊断数字见 [聚合记录](cpu_normalization_all23_diagnostics_20261004.public.json)。

### prepared CPU 静态对角坐标优化

`_PullField.sample` 的静态坐标准备仅在已准备重复采样源、CPU 连续 FP64 无梯度查询、严格对角矩阵、正零平移及有限查询时，使用逐轴乘法再加原平移。输出重新分配，保留每次 FP64 舍入。非有限值保留原矩阵乘法中的 `0 × NaN/Inf` 语义；剪切、非零或负零平移、其他布局、梯度、CUDA 和普通单次调用沿用原矩阵乘法。embedded inverse 的原矩阵乘法保持原样。

完整首个实际查询的 18,808,200 个 FP64 值在 1/8 线程下均逐 uint64 位一致；完整逆场随后各做一次同源 old/new 观察。全部保存文件 SHA-256、输入哈希、QC、有效区域比例及停止次数一致，两者实际均运行 30 次且未收敛。

| CPU 上限 | 优化前完整 instrumented API | 静态对角优化后 | RSS 峰值 KiB，前 → 后 |
|---|---:|---:|---:|
| 1 | 57.2806 s | 54.7925 s | 1,745,268 → 1,745,412 |
| 8 | 15.3450 s | 15.3322 s | 1,736,980 → 1,736,876 |

这是共享节点上的各一次完整观察，入口导入排除、函数内惰性导入/JIT 计入；不据此给稳定速度比或与旧官方时钟拼成新对照。新定向回归共 187 项通过，包含实际 CUDA、有限全位模式、非有限值、符号零、溢出、梯度和全局默认设备合同。源码与全部聚合门槛见 [静态坐标记录](cpu_static_diagonal_20261004.public.json)。

### 最新冻结26的正式原版对照

在相同物理核35及`35,39,43,47,51,55,59,63`，完整原版/FNIT进程各排除一次warmup，保留三次交替配对；独立的已加载API另做完整warmup和三次读入、计算及保存。完整MRI网格和30次最大迭代保持原样。

| CPU上限 | 官方/FNIT完整进程中位数 | FNIT已加载完整API中位数 | 完整进程速度目标 |
|---|---:|---:|---|
| 1 | 35.5828 / 55.5842 s | 53.7330 s | 未达到 |
| 8 | 35.7324 / 16.5424 s | 14.2592 s | 达到 |

原版与FNIT使用不同求逆算法和停止规则，精度合同继续单列。全18,808,200值有限，全网格最大分量差7.0261mm；声明脑掩膜区域的向量差中位数0.010075mm、p95为0.033606mm。最新源码、完整重复时钟、实际CPU使用与原版精度见 [冻结26正式公报](cpu_static_diagonal_all26_official_20261004.public.json)。表格给出最新实测状态；all23历史时钟及old/new单次诊断保持各自范围。

### 最新 H100 source28 门槛

最新 [GPU28 完整报告](../../validation/multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json)绑定当前 Inv/Convert/Apply 与 normalizer 的实际 SHA。完整系数反场 16 次保存的 18,808,200 值、header、extensions、affine 均与起始 FNIT `1d31e7b` 逐位一致，各调用 peak allocation 均为 1,275,725,824 B。两组完整 API 中位数为基线 4.298/4.094 s、v28 3.870/3.993 s；包含全部读取、求逆与保存，排除进程启动和入口导入，函数内惰性导入计入。同期完整六层三阶段 TBSS 的四张输出和对应显存峰值也相同，两例共 32 次完整保存。GPU 使用同 UUID、TF32、20 GB 上限，输入和源码前后 SHA 通过；共享时钟不作稳定速度结论。最新 nodecw8 CPU1/8 官方完整进程中位数为 35.878/50.899 s、36.679/21.661 s（FSL/FNIT），单核仍慢；[源 v28 专页](../invwarp/CPU_ALLOCATION_20261004.md)保留协议、精度及分步骤记录。

### 历史 H100 source26 门槛

root 在同一 H100、共同 GPU 锁和 20 GB allocation 上限核对起点 main 与冻结26。完整逆场的 18,808,200 个值在全部 16 次保存中，array、header、extensions 和 affine 位级一致；每次 peak allocation 均为 1,275,725,824 B。两组 warm API 中位数为基线 3.9618 / 3.9023 s、候选 3.9351 / 3.9157 s。运行前 GPU 被其他作业占用 61,606–66,014 MiB，利用率 97–100%；这里只报告共享 GPU 观察。完整记录见 [GPU26 Inv/MNI 公报](../../validation/multimodal_cpu_20261004/gpu_inv_mni_v26_20261004.public.json)，此前 GPU23 保留原历史范围。

## 6. 更新与诊断记录

| 阶段 | 结果 |
|---|---|
| 初始 float32 原型 | 真实系数逆场未经过该入口；保留 `failed_capture` 诊断，不宣称加速。 |
| FP64 v3/v4 | 完整逆场 1/8 线程文件、QC、停止次数一致；修复新缓冲必须显式跟随输入 CPU 设备。 |
| float32 完整影像观察 | 冷 JIT 产生额外准备成本；该分支未采用，恢复原张量实现。 |
| 最终版本 | 仅保留重复 prepared-source 的 FP64→FP64 CPU 融合，单次 FP64 恢复张量准备；GPU 数学、float32 准备、求解预算与低精度策略不变。最终源码哈希和实际计时快照分开记录。 |
| prepared CPU 静态坐标 | 仅有限、严格对角、正零平移的重复 CPU 查询改为逐轴运算；完整逆场 1/8 线程全部文件、QC 和停止次数一致，正式原版计时另记录。 |

## 7. 实现与参考

- [FP64 CPU helper](../../src/fnit/applywarp/_normalization_cpu.py)、[CPU dispatch](../../src/fnit/applywarp/core.py)、[位级回归](../../tests/applywarp/test_cpu_normalization.py)。
- [完整诊断工具](../../tools/benchmark_applywarp_normalization_cpu.py)、[公共聚合报告](cpu_normalization_20261004.public.json)。
- [静态坐标准备](../../src/fnit/convertwarp/core.py)、[异常与路由回归](../../tests/convertwarp/test_cpu_diagonal_pull.py)。
- [原始 InvWarp 算法、FSL 原实现及文献](../invwarp/README.md)。
