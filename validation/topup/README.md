# TorchTOPUP 真实数据验收

这里将同输入完整估计、固定官方参数渲染和首层梯度检查分开记录。运行时实现见[组件说明](../../docs/topup/README.md)；验证脚本读取独立保存的 FSL 输出，不调用 FSL。输入与输出 NIfTI 留在调用方的私有目录，公开报告只包含数值汇总、输入与代码 SHA-256，不包含绝对路径或病例标识。

## 2026-10-02 验收范围

- [同输入完整估计](report.matched_20261002.public.json)：同一真实 `104×104×72×2` b0 对、采集参数和默认九层 `b02b0.cnf`；FNIT 自行估计场与运动，参照 FSL 6.0.7.4 的独立结果。场图固定信号区 RMSE `0.010799 Hz`；固定原软件脑区的 iout RMSE `12.849295`、最大绝对误差 `1037.257813`。九层均用完默认预算，共接受 85 步，未报告收敛。
- [三次独立进程重复](repeats_20261002.public.json)：同一病例的三次运行，源码哈希、精度统计和运动误差相同；API 读算写中位数 `7.094846 s`，内部同步计算中位数 `6.368049 s`，含启动和统计的完整进程中位数 `10.82 s`。主要精度报告保存第 2 次，API 时间 `6.704100 s`。
- [两次原生 FSL 重复](reference_repeats_20261002.public.json)：同一真实 pair、采集参数及默认 `b02b0.cnf`，CPU 16–23、8 线程；外层同步 subprocess 墙钟中位数 `211.110510 s`，两次输出文件哈希相同。没有把完整参考链 CPU 0–7 的单次时间混入中位数。
- [固定官方参数](fixed_parameters_20261002.public.json)：读取官方保存的 coefficient/movement，不估计参数，只渲染同一输入。脑内 iout RMSE `0.011135`，支持区 Dice `1`；脑内 Hz 场 RMSE `4.866082×10⁻⁷ Hz`，两幅全 FOV Jacobian RMSE 约 `2.00×10⁻⁸`。保存系数已转 float32、movement 已转文本，原软件输出使用内部参数，比较包含该落盘边界。
- [首层梯度](initial_gradient_20261002.public.json)：默认首层、零场、零运动；有效体素数与 FSL 同为 88,400，SSD `924.964677585` 对应原软件日志 `924.965`。1,695 个梯度值相对 L2 差 `4.569587×10⁻⁶`；首个接受 LM 步 cost `629.979685552` 对应原软件 `629.98`。
- [真实坐标 sampler](sampler_20261002.public.json)：固定同一真实图像系数与由官方场/运动生成的源坐标，对比融合 CUDA sampler 与独立 double tensor 逐 tap 数学参照；两帧全部强度、三个空间导数和 valid mask 精确相同。暖同步 wall 时间比分别 `89.002×/91.043×`，仅表示 tensor 参照与融合 sampler 的比，不代表 FSL 或完整 TOPUP。

当前实际验收源码 `core.py` SHA-256 为 `d6b9838ca62ffeaa32b608a860520fc3feb5e66582064303a6de47f199e2b8e8`，CUDA helper 为 `ee19a764849bda80137312ed3ab8f1bf0aaef5e13f852f49f435e511b40d2427`。估计、重复与 sampler 报告直接记录两项哈希；固定参数和梯度报告直接记录 core 哈希。估计、固定参数与梯度三项的输入 SHA-256 相同：`f701b4ef97e39f3e821030d8c630f575a1410ec380627f26f11312005968c774`。当前实现已经删除未调用 helper 和旧 float CUDA 分支。

本地[104 项组合回归](../dmri_pipeline/regression_synthstrip_topup_20261002.public.json)通过（13.61 s）：Python 3.11.16 / PyTorch 2.5.1，其中 20 项 CUDA sampler 测试运行于 RTX 3060，其余为 CPU 接口和数学合同检查。这些小测试不作为真实精度或性能 benchmark。

## 独立同输入估计与统计

准备好同一 b0 对和独立原软件输出后运行：

```bash
PYTHONPATH=src python validation/topup/benchmark_matched.py \
  --imain /private/input/B0_AP_PA.nii.gz \
  --datain /private/input/acqparams.txt \
  --official-dir /private/official/topup \
  --output-dir /private/fnit/topup \
  --brain-mask /private/official/nodif_brain_mask.nii.gz \
  --report report.matched_20261002.public.json \
  --device cuda:0 \
  --threads 8 \
  --memory-limit-bytes 20000000000
```

| 参数 | 输入或默认值 | 作用 |
|---|---|---|
| `--imain` | 必填 `[X,Y,Z,2]` NIfTI | 同一对真实、已选中的 AP/PA b0，作为两实现的固定输入。 |
| `--datain` | 必填两行四列文本 | 三个方向分量与总读出时间（秒）。 |
| `--official-dir` | 必填目录 | 独立 FSL 输出：`fieldmap_fout/iout/jacout_01/jacout_02.nii.gz`、`fieldmap_out_fieldcoef.nii.gz` 和 `fieldmap_out_movpar.txt`。 |
| `--output-dir` | 必填新目录 | FNIT 输出 NIfTI、运动文件和持久化 `measurement.json`；这些图像不公开。 |
| `--report` | 必填新 JSON 路径 | 数值、计时、版本和哈希报告。 |
| `--brain-mask` | 可选 NIfTI | 同输入空间网格的独立原软件 mask，按 `>0.5` 定义固定脑区。 |
| `--compare-only` | 默认关闭 | 仅比较已有输出；已有 `measurement.json` 时读取原计时，不重跑估计。 |
| `--device` | 默认 `cuda:0` | PyTorch 设备；CPU 也可运行，计时必须标明设备。 |
| `--threads` | 默认 `8` | PyTorch CPU 线程数。CPU 核隔离由外部启动环境设置。 |
| `--memory-limit-bytes` | 默认 `20000000000` | CUDA 进程分配预算，bytes。 |

脚本先完成 API 读取、计算、输出写盘并 CUDA 同步，把测量结果写入 `measurement.json`，随后独立计算统计。统计失败时可以用新 `--report` 路径与 `--compare-only` 恢复报告，不重复模型计算。

原软件独立运行使用：

```bash
topup --imain=/private/input/B0_AP_PA.nii.gz \
  --datain=/private/input/acqparams.txt \
  --config=b02b0.cnf \
  --out=/private/official/topup/fieldmap_out \
  --fout=/private/official/topup/fieldmap_fout \
  --iout=/private/official/topup/fieldmap_iout \
  --jacout=/private/official/topup/fieldmap_jacout
```

默认配置的 regrid、平滑、正则权重、subsampling、knot resolution 和 LM/SCG 调度见[组件参数表](../../docs/topup/README.md#实现范围)。以上组件验收从固定 b0 对开始；UKB b0 选择与后续 EDDY、脑 mask、DTI/NODDI、TBSS 的完整原始数据链需另行计时和比较。

## 区域、几何与残差解读

完整估计报告列出全 FOV、固定原始两帧均值 `>100` 信号区及可选固定原软件脑区。iout 两帧的 ROI 标量一起展开计算 Pearson r、MAE、RMSE、p95 和最大绝对差值。形状/有限值不通过时不计算误差；场图和 iout 还要求 affine 在 `atol=1e-5, rtol=0` 下匹配。系数和 Analyze-style Jacobian 的 header 有专门的 TOPUP 约定，报告分别记录 affine、intent 和 shape；不要把其 affine 当作普通脑图的世界坐标。内部 canonical storage 与输入 storage 的区别见[输出合同与实现范围](../../docs/topup/README.md#实现范围)。

固定脑区是主要脑内 ROI。共同支持区与支持差异区可用来定位边界和零值造成的残差，并列报告各区 count、误差与平方误差贡献。对完成的输出执行 `diagnose_output_support.py`，[支持诊断](output_support_20261002.public.json)确认固定脑区的 271,080 个体素在两侧、两帧 iout 中全部非零，脑内非零支持区 XOR 为 0，支持差异贡献的平方误差为 0。共同非零脑区仍得到 RMSE `12.849295`、最大误差 `1037.257813`；全 FOV 的 FNIT-only 非零体素为 7、原软件-only 为 0，均在固定脑区外。该报告没有传入独立几何 mask，非零支持和几何有效性分别解释。

固定参数渲染已接近浮点舍入水平，完整独立估计仍有场、Jacobian、系数、运动和 iout 差异。场与运动的小幅估计差会通过采样位置、局部强度梯度和 Jacobian 调制影响 iout，本次尚未隔离每项贡献，不将残差只归因于 GPU 舍入。CPU 数学测试用于 mask、坐标、导数、GN 和求解器回归，不用作 benchmark。

## 计时与历史结果

当前源码的 H100 三次独立进程运行使用 CPU 8–15、threads 8、CUDA 分配预算 20,000,000,000 bytes。计时段没有本任务自身其他 GPU 作业重叠，原软件 CPU 配准在不同 CPU 核运行；节点未做独占。三次精度统计和运动误差相同，峰值 CUDA 已分配显存均为 `389,119,488 bytes`。

| 计时范围 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 |
|---|---:|---:|---:|---:|
| API 读取＋计算＋NIfTI 写盘 | 17.864061 s | 6.704100 s | 7.094846 s | 7.094846 s |
| 内部同步计算 | 17.130760 s | 5.966458 s | 6.368049 s | 6.368049 s |
| 完整进程，含启动、imports、CUDA 初始化、统计/JSON | 22.13 s | 10.56 s | 10.82 s | 10.82 s |

主要精度报告取第 2 次，API 时间 `6.704100 s`，不是三次中位数。第一轮较慢的原因尚未隔离，不把时间差直接归因于 JIT。

FSL 在同一节点本地存储、同一真实输入上独立运行两次，CPU 16–23、8 线程、不使用 GPU；其他本任务 CPU 作业使用不同核，系统 cache 未清空。两次原生输出文件哈希一致。

| FSL 计时范围 | 第 1 次 | 第 2 次 | 中位数 |
|---|---:|---:|---:|
| 外层同步 subprocess 墙钟 | 216.987430 s | 205.233591 s | 211.110510 s |
| GNU time 原生进程墙钟 | 216.98 s | 205.22 s | 211.10 s |

外层 FSL 中位数相对 FNIT 完整验证进程中位数的观测时间比为 `211.110510/10.82 = 19.51×`，后者多包含误差统计和 JSON 写入。FNIT API 中位数 `7.094846 s` 排除启动和验证统计，不与 FSL 完整进程作同范围时间比。本轮完整参考链 CPU 0–7 上 TOPUP 的单次 `215.4733 s` 单独记录；当前两次 FSL 中位数仅使用 CPU 16–23 的两次 standalone 运行。该比值不沿用旧版时间，且不表示数值等价或完整 dMRI 链加速。

sampler 在每种实现 warmup 两次后，用两个 ABBA block 对相同驻留真实输入计时。两帧 tensor wall 中位数 `0.019326/0.019327 s`，融合 sampler `0.000217/0.000212 s`，时间比 `89.002×/91.043×`；没有 FSL 执行作为该 sampler 对照。计时排除输入 IO、regrid、prefilter、几何准备、JIT warmup 和 JSON 写入；CUDA event 是 stream 区间，包含逐 tensor kernel 启动间隙，不是排除其他开销后的 kernel 求和。完整组件与 sampler 的时间分别报告。

[`report.public.json`](report.public.json) 和 [`SHA256SUMS`](SHA256SUMS) 保留 FNIT 0.14.0/0.16.0 L-BFGS 路径的历史精度与三次计时，不覆盖 2026-10-02 的核心与 sampler。历史图和计时比较位于[组件说明](../../docs/topup/README.md#历史-l-bfgs-版与-fsl-6074-的真实数据对照)；新版没有用历史时间作当前加速比的分母。

## 参考

- 原实现：[FSL TOPUP](https://git.fmrib.ox.ac.uk/fsl/topup)，本仓库源版本 `2203.2`，commit `3e2cb9104e834ce18c10e4b7edddbd500d0c459c`。
- Andersson, Skare & Ashburner (2003), [doi:10.1016/S1053-8119(03)00336-7](https://doi.org/10.1016/S1053-8119(03)00336-7)。
- 源码来源与许可见 [`src/fnit/_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md) 和 [`FSL Software Licence 6.0`](../../licenses/FSL-6.0.txt)。
