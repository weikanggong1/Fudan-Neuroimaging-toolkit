# TorchMCFLIRT 融合后的完整 volume 复测（`cfb7beee`）

2026-10-01，冻结源码 `cfb7beee202f7e89072faf1a8e69b78d143e451f` 在同一例真实 UKB 数据上完成全部 490 帧的公开 `fMRIVolume_pipeline`。API 同时生成 **preproc 与 clean**，含最终保存为 **707.287 s**。运动校正解码图与冻结 FNIT `1eb9c417` 逐值相同；重新运行完整流程后，MNI clean 与冻结 FNIT 的逐体素时间 r 均值为 **0.992649**，与原 SynthStrip/FSL/ICA-AROMA 同步骤参照为 **0.939162**。

2026-10-02，独立 [TorchMCFLIRT](../../docs/mcflirt/README.md) 在 `8a3f227` 更新矩阵缓存及 CUDA graph；同卡完整 490 帧配对调用从 86.37 / 109.46 s 降至 52.47 / 63.48 s，运动输出逐 bit 保持。本页继续记录上述冻结源码的整链测试，707.287 s 未包含本轮优化。

## 输入、输出和本次配置

输入为同一原始 BOLD/SBRef、匹配的存档 T1、MNI 模板与脑掩膜，以及官方 SynthStrip 权重。BOLD 为 `88×88×64×490`，TR `0.735 s`；T1 为 `162×215×180`，经 nibabel 转换保持体素与 affine。原始影像、逐体素时序、完整日志和路径留在服务器，公开 JSON 仅含聚合指标与 SHA-256。

| 输出 | 实际结构 |
|---|---|
| `clean_native` | 原生 EPI `88×88×64×490`，float32 |
| `clean_mni` | MNI 2 mm `91×109×91×490`，float32；脑掩膜外为零 |
| `preproc_t1w` | T1w scanner-RAS、原生 BOLD 分辨率 `59×75×64×490`，float32 |
| `preproc_mni` | MNI 2 mm `91×109×91×490`，float32 |

四份时序均全部有限，保留 490 帧与原始 TR。preproc 从原始 BOLD 组合运动、BBR 和 T1→MNI 变换后一次插值，保留原始强度基线；clean 包含 100 秒高通、nonaggr AROMA、WM/CSF 与 24 项运动回归。具体文件名、每个 API 参数、Python/CLI 示例见[功能页](../../docs/fmri/README.md#python-调用与参数)。

本次关闭 STC、SDC/GDC、额外带通与全脑信号回归。T1→MNI 为 `registration_backend="fnirt"`，BBR 为 `batched`，FNIRT 为 `optimized`；运动仍为 8/4/4 mm、原 NCC、顺序 Brent 和 `(1,1,1)` 轮数，最终 Constant 三次样条与源整数类型转换保持原规则。融合仅减少 cost 输入准备的 GPU 调度，并复用帧与 COG；独立 CUDA 元素控制和固定真实矩阵 cost 控制见 [MCFLIRT 验证页](../mcflirt/README.md)。没有改用半精度。

解剖缓存配置为 `reuse_anatomical=True`，本次使用新 derivatives 目录，报告的 `anatomical_cache.reused=False`，没有命中旧解剖结果。ICA 自动定阶为 95 个成分，43 次迭代收敛；AROMA 标记 47 个噪声成分。运行环境为共享 H100 PCIe、8 个 CPU 线程、PyTorch 2.5.1/CUDA 11.8，TF32 开启，进程 CUDA 额度 20 GB。

## 完整时间与测量边界

以下均读取[最新 API 报告](mcflirt_optimization_api.public.json)，单位为秒。

| 范围 | 实测 |
|---|---:|
| **公开 volume API，含四份 BOLD 和最终 BIDS 输出保存** | **707.287** |
| pipeline 内部 `total`，末尾最终发布前 | 694.390 |
| FEAT：运动、采样、掩膜准备、缩放、高通和阶段输出 | 177.787 |
| PICA＋ICA-AROMA＋混杂回归 | 163.092 |
| clean MNI 重采样 | 50.424 |
| T1w＋MNI preproc 单次插值与该阶段输出 | 251.851 |
| 验证中间产物捕获，已从上述 API 和阶段计时扣除 | 26.658 |
| 当前进程 CUDA 峰值 allocated / reserved，十进制 GB | 6.503 / 10.775 |

API 不含输入哈希、事后输出检查及比较；包含 API 内读写、权重加载和清理。具名阶段并非全部整链开销，不能通过阶段求和替代 `total`，也不能从 API 减去 preproc 时间推造另一次 clean-only 实测。新目录只保证本次没有解剖缓存命中，进程/编译缓存和共享 GPU 负载仍影响耗时。

原软件[连续 clean 链](matched_native_pipeline.public.json)观测 **2570.468 s**，含各阶段启动、阶段完整性检查及 MELODIC HTML，不含新增 T1w/MNI preproc。原链无计算阶段跳过或复用；输入 preflight 哈希和最终 MNI 检查在外。最后原 applywarp 退出码为 255，但完整输出通过形状、有限值及掩膜外零值检查，报告保留了原退出码。两方输出范围、计时边界和共享负载不同，这些单次结果不直接构成加速比。

独立 MCFLIRT 与完整 volume 分别测试。固定保存后的 FEAT 参考、全部 490 帧时，最新 `cfb7beee` 独立 MCFLIRT 在共享**物理 GPU 0** 的 API 为 **278.454 s**，其中最终采样 **32.010 s**；此前 `7456251` 在共享**物理 GPU 1** 观测 **159.275 s**，采样 **24.830 s**。两次 cost 调用均为 **45,972**，运动 MAT/par 文本及解码校正图对冻结 FNIT 一致；它们不等于此完整 volume 的 FEAT 时间。详见[最新独立报告](../mcflirt/gpu_optimization_latest.public.json)与[前次独立报告](../mcflirt/gpu_optimization.public.json)。

## 与原软件同步骤 clean 链的精度

[最新原软件比较](mcflirt_optimization_native.public.json)核对同一 BOLD/SBRef/T1、模板、脑掩膜和权重哈希。各阶段按报告中的共同脑区计算完整 490 帧；时间 r 为逐体素去均值 Pearson r，排除去均值 RMS≤1e-6 的时序。MNI clean 共同域为 224,707 体素。RMSE 使用该阶段保存的实际强度，没有另拟合尺度、偏移或平滑。

| 阶段 | 时间 r 均值 | 中位数 | RMSE |
|---|---:|---:|---:|
| 运动校正 | **0.999578** | 0.999834 | 8.211 |
| 缩放/高通后的 pre-ICA | 0.999356 | 0.999770 | 11.745 |
| 原生 AROMA | 0.934290 | 0.944640 | 86.928 |
| 原生最终 clean | 0.941029 | 0.951576 | 78.031 |
| MNI 最终 clean | **0.939162** | 0.947900 | 62.025 |

这是原 SynthStrip/FSL/作者 ICA-AROMA 加独立 NumPy 联合回归的 matched-step **clean** 参照，不是 fMRIPrep preproc，也不是 UKB FIX release。本次只比较报告列出的五个时序阶段；历史 `1eb9c417` 的组织、变换和掩膜控制仍留在[历史页](matched_native.md)，不能将其数字改名为此次全链结果。

## 与冻结 FNIT 的整链回归

[冻结比较报告](mcflirt_optimization_comparison.public.json)使用候选与 `1eb9c417febebc8bdd450590d4759454e7160141` 两次保存产物，不重新估计变换或去噪；两侧 derivative 记录的运行时源码 SHA-256 均与提供的冻结源树一致。

原生 RMSE 使用冻结 FEAT 的 99,372 体素掩膜，时间 r 排除常数时序；MNI RMSE 使用冻结输出的 224,882 体素脑掩膜，其中 224,877 个体素可计算时间 r。pre-ICA 的 RMSE 包含边界体素，时间 r 使用共同有效时序。

| 阶段 | 时间 r 均值 | 脑区 RMSE | 解码图逐值一致 |
|---|---:|---:|---|
| 运动校正 | **1.000000** | **0** | **是，全部 242,851,840 个 int32 值** |
| pre-ICA | 1.000000 | 0.429008 | 否 |
| 原生 AROMA | 0.991043 | 31.281605 | 否 |
| 原生最终 clean | 0.992109 | 27.658637 | 否 |
| MNI 最终 clean | **0.992649** | **21.194244** | 否 |

独立完整运行的 EPI 脑掩膜为 99,371/99,372 体素，仅相差 **1 个边界体素**，Dice 为 0.999994968。pre-ICA 全图逐值相同占比为 0.999997982，差异为该边界体素的 490 个值；共同有效脑区时间 r 全为 1。MNI 输出掩膜相差 5 体素，完整清理图仍有下游差异。这里记录独立重新估计的整链结果，没有控制证明一个边界体素是全部后续差异的唯一原因；固定掩膜下独立 MCFLIRT/FEAT 的逐值一致不能推广为完整 volume 逐值一致。

## 最新脑图

图中四行依次为同一 MNI 模板、最新 FNIT temporal SD、原同步骤 clean 参照 temporal SD，以及完整 490 帧时间 r。共同掩膜为 224,707 体素，两份 SD 共用色阶，r 为 −1 到 1；切面和文件 SHA-256 见[图来源](mcflirt_optimization_figure.public.json)。未追加空间平滑。

![cfb7beee 全490帧 FNIRT clean 与原软件同步骤对照](../../docs/fmri/figures/fmri_mcflirt_optimized.png)

## 完整参数复测

激活主页 Conda 环境，在下列源码目录运行。每个 derivatives/capture 目录均应新建，BIDS 中须能唯一选出同一被试的 run 与 T1。`--source-revision` 是运行标签，`--source-root` 与输出 sidecar 的逐文件 SHA-256 才用于核对实际源码。共享服务器需另保存运行时 GPU 状态，单次观测不作为受控速度测试。

```bash
# 候选源码冻结目录；不要将另一个版本的运行结果改名为此版本。
fnit_source_root=/absolute/path/cfb7beee-source
fnit_source_revision=cfb7beee202f7e89072faf1a8e69b78d143e451f
original_bids_root=/absolute/path/original-bids
subject_label=0001
candidate_derivatives_root=/absolute/path/fnit-cfb7beee-fresh
candidate_intermediates_root=/absolute/path/private-cfb7beee-intermediates
candidate_resampling_root=/absolute/path/private-cfb7beee-resampling
mni_template_path=/absolute/path/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz
mni_brain_mask_path=/absolute/path/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz
synthstrip_weights_path=/absolute/path/synthstrip.1.pt
candidate_report_path=/absolute/path/volume-cfb7beee.public.json

# 此次 volume 在物理 GPU 1；对程序呈现为 cuda:0。
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$fnit_source_root/src" \
python "$fnit_source_root/validation/fmri/benchmark_bids.py" volume \
  --bids-root "$original_bids_root" \
  --derivatives-root "$candidate_derivatives_root" \
  --subject "$subject_label" \
  --mni-template "$mni_template_path" \
  --mni-brain-mask "$mni_brain_mask_path" \
  --synthstrip-weights "$synthstrip_weights_path" \
  --registration-backend fnirt \
  --bbr-execution batched --fnirt-execution optimized \
  --device cuda:0 --threads 8 --gpu-memory-limit-gb 20 \
  --source-root "$fnit_source_root" --source-revision "$fnit_source_revision" \
  --capture-intermediates "$candidate_intermediates_root" \
  --capture-resampling-inputs "$candidate_resampling_root" \
  --report-out "$candidate_report_path"
```

本驱动固定 `slice_timing=False`、WM/CSF/24 项运动回归、100 秒高通、nonaggr AROMA、`batch_size=8`、`motion_iterations=(1,1,1)` 和 `random_state=0`。volume 总是保存 preproc 与 clean；`--signal` 仅作用于 surface，不用于选择 volume 的计时范围。上例保持缓存配置默认，但新 derivatives 目录应得到 `reused=False`。

复用已保存的冻结 FNIT 产物，仅计算完整整链差异。下列 `frozen_mni_comparison_mask_path` 取冻结 FNIT 的 MNI 输出脑掩膜，224,882 体素；其 SHA-256 须匹配比较报告的 `mni_comparison_mask_sha256`，本次为 `bc46d8d26647523ad853a5aac5c125e3dc9d6230c90c938a19c60b188da02120`。原始模板脑掩膜仍用于上面的 pipeline 输入，替换比较掩膜会改变 RMSE 的统计域。

```bash
frozen_source_root=/absolute/path/1eb9c417-source
frozen_derivatives_root=/absolute/path/fnit-1eb9c417-derivatives
frozen_intermediates_root=/absolute/path/private-1eb9c417-intermediates
# 此统计掩膜取冻结 FNIT 的 MNI 输出，和上面的原始模板掩膜分别保存。
frozen_mni_comparison_mask_path=/absolute/path/fnit-1eb9c417-derivatives/sub-0001/func/sub-0001_task-rest_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz
frozen_comparison_report_path=/absolute/path/cfb7beee-vs-1eb9c417.public.json

PYTHONPATH="$fnit_source_root/src" \
python "$fnit_source_root/validation/fmri/compare_mcflirt_optimization.py" \
  --candidate-derivatives "$candidate_derivatives_root" \
  --baseline-derivatives "$frozen_derivatives_root" \
  --candidate-intermediates "$candidate_intermediates_root" \
  --baseline-intermediates "$frozen_intermediates_root" \
  --mni-mask "$frozen_mni_comparison_mask_path" \
  --candidate-source-root "$fnit_source_root" \
  --baseline-source-root "$frozen_source_root" \
  --source-revision "$fnit_source_revision" \
  --baseline-source-revision 1eb9c417febebc8bdd450590d4759454e7160141 \
  --report-out "$frozen_comparison_report_path"
```

原软件命令与私有 case 清单字段见[历史 matched-step 方法](matched_native.md#数据协议与原命令)，连续驱动 [`run_native_matched_pipeline.py`](run_native_matched_pipeline.py) 仅用于参照环境。用 [`compare_matched_pipeline.py`](compare_matched_pipeline.py) 的 `--manifest` 配对本次保存产物与原链时，须重新核对 manifest 指向的全部输入/输出哈希，不能沿用旧候选的比较报告。原实现与论文见[功能页引用](../../docs/fmri/README.md#参考文献与原实现)。

## 最近版本与报告索引

| 源码快照 | 实际测量与变化 |
|---|---|
| `1eb9c417` | 原 NCC/Brent/整数转换及 SynthStrip、FAST、PICA 数值修复；历史 FNIRT clean API 1318.04 s，FEAT 973.12 s 包含运动与其余 FEAT 步骤。详见[历史报告](matched_native.md)。 |
| `50eb098` / `ca3df003` | SynthMorph 的完整 preproc＋clean API 1225.840 s；固定 fMRIPrep 输入的 surface 控制另行报告。详见[验证索引](README.md)。 |
| `7456251` | 融合 cost 输入准备、帧/COG 复用；独立 MCFLIRT 159.275 s、45,972 次 cost，在共享物理 GPU 1 上与冻结运动/固定掩膜 FEAT 逐值一致。 |
| `44364a8` | [早期 clean-only 执行快照](mcflirt_optimization_clean_only_44364a8.public.json) API 464.118 s；未生成新增 preproc，不作为当前完整 API 的测量。 |
| `cfb7beee` | 最新完整 preproc＋clean API **707.287 s**；独立 MCFLIRT **278.454 s** 为另一共享物理 GPU 0 上的观测，保留各自边界。 |

- [最新完整 volume API、配置、网格与时间](mcflirt_optimization_api.public.json)
- [最新与原软件同步骤 clean 比较](mcflirt_optimization_native.public.json)
- [最新与冻结 FNIT 的整链比较](mcflirt_optimization_comparison.public.json)
- [最新脑图来源与 SHA-256](mcflirt_optimization_figure.public.json)
- [最新独立 MCFLIRT](../mcflirt/gpu_optimization_latest.public.json) · [前次独立 MCFLIRT](../mcflirt/gpu_optimization.public.json)
- [真实固定矩阵 cost 微基准](../mcflirt/cost_optimization.public.json)：预热后 eager CUDA 约 27.5 ms/次、融合约 4.5 ms/次；只是 cost 测量，完整复测参数见 [MCFLIRT 验证页](../mcflirt/README.md#单次代价函数微基准)。
