# FLIRT 验证资产

[功能、输入输出与参数](../../docs/flirt/README.md) · [当前机器可读报告](gpu_batch.current.public.json)

本轮覆盖真实 b0→T1、T1→MNI152 和 CC0 公开 T1w→T1w；同时记录科学修复前后精度、未融合→融合执行 gate、cold/warm 时间、显存和完整 profile。303 项相关测试通过。每次测量保留实际源码和工具 SHA-256；私有影像仅发布汇总标量，没有输入路径、被试标识、原始矩阵或逐体素数组。

| 文件 | 用途 |
|---|---|
| `gpu_batch.current.public.json` | 本轮三例的修复前后实测、融合 gate、消融、官方 CPU 环境差异及公开图来源。 |
| `report.cpu.current.json` | 历史串行 12-DOF GM 10 例 CPU/FSL 对照，保留原测量源码。 |
| `report.gpu.current.json` | 同一 GM 队列前 4 例历史串行 H100/FSL 对照。 |
| `applyxfm_mni.cpu.json`、`applyxfm_mni.gpu.json` | 原始 MNI152 T1 模板 1 mm↔2 mm 重采样，按历史测量源码保留。 |
| [1→2 mm 矩阵](../../src/fnit/flirt/assets/FSL_MNI152_T1_1mm_to_2mm.mat) | 上述原始模板对应的 scaled-mm 矩阵；模板不随包发布。 |
| `benchmark_applyxfm.py`、`validate_real.py` | 分别重跑模板 applyxfm 和 GM 队列的离线验证。 |
| `plot_public_example.py` | 从已保存的 FSL/FNIT moved 图像画三视图。 |
| `SHA256SUMS` | 报告、脚本、公开图和当前 benchmark 工具的校验值。 |

历史 GM 报告来自串行 `core.py` `56a934…`，不是当前融合版的全队列重跑；CPU 的 `rmsdiff <= 0.05 mm` 为 9/10，串行 H100 为 4/4。对应源码继承记录见[历史范围核对](../runtime_dependencies/flirt_profile_source_equivalence.public.json)。

## 当前精度

矩阵比较在 moving 视野内 13³ 个 world-grid 点计算，双方 scaled-mm 均使用 header `pixdim`。图像比较的固定 mask 是官方输出非零区域；MAE 单位是原始强度。下表为本轮最终版：

| 输入与官方 oracle | 位移 mean / median / p95 / RMS，mm | Pearson | MAE | Dice |
|---|---:|---:|---:|---:|
| b0→T1，FSL 6.0.7.22 CPU A | 0.00199 / 0.00200 / 0.00342 / 0.00216 | 0.9999814 | 13.8556 | 0.999579 |
| b0→T1，FSL 6.0.7.22 CPU B | 0.17397 / 0.17846 / 0.27678 / 0.18629 | 0.9996545 | 57.5962 | 0.998071 |
| T1→MNI152，FSL 6.0.7.22 CPU B | 0.11471 / 0.11274 / 0.20462 / 0.12624 | 0.9999838 | 0.9761 | 1.000000 |
| 公开 T1w→T1w，FSL 6.0.7.4 | 0.01191 / 0.01156 / 0.02319 / 0.01355 | 0.9999965 | 0.5555 | 0.999896 |

b0 在 CPU A 的官方重跑复现原 oracle，矩阵文件与体素逐 bit 相同；CPU B 的官方结果与 A 相差 0.18773 mm RMS。两边入口与所核验 FSL 动态库相同，系统库不同，具体数值差异来源尚未隔离。T1→MNI 的新 oracle 与原 oracle 逐 bit 相同。

公开病例的主要精度 bug 是 affine 范数取整使 1 mm 网格变成 2 mm。仅修正 header 采样距离即可将 RMS 从 0.37208 降到 0.01353 mm。T1→MNI 修复后图像 Pearson 从 0.9968224 提升到 0.9999838，但矩阵 RMS 从 0.12269 增到 0.12624 mm；独立 header/cost/schedule/rounding 消融均保留在报告中。回退本次 Brent 舍入不会改变该最终矩阵。

固定官方 T1→MNI 矩阵后，单独核查重采样，Pearson 为 0.9968247→0.9999963、MAE 为 10.0606→0.4649，确认输出背景及空间信息修复对图像一致性有实际作用。完整注册与此固定矩阵诊断分别记录。

FSL 逐行 float32 坐标递推与当前逐点公式仍有舍入差异。公开真实 T1 的固定矩阵、规则行子采样诊断中，坐标误差 mean/p95 为 0.000414/0.001144 mm，相同子集 cost 差约 2.09×10⁻⁶。它不是全图官方 cost，也没有证明本次 T1→MNI RMS 增加的原因。

## 重跑当前 GPU 对照

[benchmark_flirt_gpu.py](../../tools/benchmark_flirt_gpu.py)读取已保存的 FSL oracle，分别运行 `--execution reference` 和 `--execution batched`；[check_flirt_gpu_parity.py](../../tools/check_flirt_gpu_parity.py)检查矩阵、体素、header、cost、求值次数及已捕获的角度候选。完整命令与计时范围见[功能页](../../docs/flirt/README.md#重跑性能与精度对照)。FNIT 运行时不调用 FSL。

本轮融合与科学修正后的未融合实测通过三例 gate；T1→MNI 的独立 profile 输出与正常输出也通过矩阵文件、体素、header 和 affine 检查。旧版数值经过 header-aware scorer 重新计量时，只新增计量记录，原计时及其源码/工具 hash 保持原值。

## 阶段与 profile

最终融合版无 profiler 的 warm 阶段时间，单位秒：

| 阶段 | b0→T1，H100 | T1→MNI152，H100 | 公开 T1w→T1w，RTX 3060 |
|---|---:|---:|---:|
| reference pyramid | 0.019 | 0.019 | 0.147 |
| angular search，含 8 mm refinement | 0.892 | 0.864 | 1.091 |
| 4 mm local optimization | 0.749 | 0.943 | 2.428 |
| 2 mm local optimization | 0.308 | 0.531 | 0.951 |
| 1 mm local optimization | 0.430 | 0.442 | 2.427 |
| final resampling dispatch | 0.012 | 0.032 | 0.080 |

阶段为 host wall scope；异步准备工作可能在下一阶段完成，重采样行不含随后结果回传。8 mm refinement 已计入 angular search，不重复相加。层级标签是 schedule 请求值；低于输入/reference 最小采样距离时，按 FSL 规则保留可用网格。总 cold/warm、未融合版及旧版的分阶段记录见 JSON 和[功能页速度表](../../docs/flirt/README.md#修复前后速度与精度)。

| 完整 profile 计数 | 公开 RTX 3060 | T1→MNI152 H100 |
|---|---:|---:|
| kernel launch | 49,557 | 45,602 |
| `cudaStreamSynchronize` | 850 | 764 |
| `cudaMemcpyAsync` 总调用 | 2,427 | 2,987 |
| H2D / D2H copy event | WSL 无 device timeline | 747 / 725 |
| CUDA tensor `float()` / `bool()` / `cpu()` | 12 / 0 / 787 | 10 / 0 / 705 |
| profiler 边界显式 synchronize | 8 | 7 |

runtime API、device copy event 与 tensor API 是不同计数，不将 `cpu()` 调用直接当成精确同步次数。WSL 的 kernel 数来自真实 runtime launch；H100 同时捕获 device timeline。上次公开批量 profile 为 114,928 launches、841 stream synchronize，原源码与工具 hash 作为诊断基线单独保留。

CUDA 峰值 allocated / reserved 分别为 b0 2.63/3.08、T1→MNI 1.01/1.45、公开 T1 1.87/2.17 GiB。H100 全卡约 100% 利用率包含其他作业；公开 T1 的 cold/warm 全卡平均为 39.6%/47.9%，包含桌面渲染。profile 会增加计时开销，其耗时仅用于诊断；科学搜索、迭代和范围保持不变，没有引入近似 `fast` 路径。

## 重跑 GM 队列

UKB VBM 保存的 moved 图像包含后续 mask，不能作为 FLIRT 直接输出。先用已有官方矩阵生成独立 oracle：

```bash
flirt -in subject_GM.nii.gz -ref template_GM.nii.gz \
  -applyxfm -init T1_GM_to_template_GM.mat -out official_apply.nii.gz
```

再运行 FNIT，以下目录由验证者按本地数据填写：

```bash
python validation/flirt/validate_real.py run \
  --study-root <deidentified_study_root> --source-root <fnit_source_root> \
  --official-moved-root <fresh_fsl_applyxfm_root> \
  --output-root <candidate_cpu_root> --reference-report <fsl_timing_report.json> \
  --report <new_cpu_report.json> --case-count 10 \
  --device cpu --execution reference --threads 4 \
  --candidate-hardware "Intel Xeon Gold 6418H"
```

队列目录与参考计时记录格式由脚本定义。重跑当前 GPU 改为 `--device cuda:0 --execution batched`，硬件填写实际型号；重跑串行 GPU 对照用 `--execution reference`。CPU/GPU 分别输出报告，不将 10 例与 4 例合成同一队列。脚本每次重新运行候选，避免将旧输出绑定到当前源码 hash。

公开图重画命令：

```bash
python validation/flirt/plot_public_example.py \
  --moving examples/data/sub-02_T1w.nii.gz \
  --reference examples/data/sub-01_T1w.nii.gz \
  --fsl-moved <fsl_oracle.nii.gz> --fnit-moved <batched/cold_0.nii.gz> \
  --figure docs/flirt/figures/flirt_public_current.png
```
