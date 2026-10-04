# SynthSR 首差定位与 WMH-SynthSeg 内存修复（2026-10-04）

本轮基于 `f1cbdab10fbfd573c3aa1b3461aa086dab220cfb`，只修改 WMH 子函数、对应测试和说明。SynthSR 生产 `model.py` 保持原 SHA `290d0ae911d8c99f4f87c22936ca497ea45bb57b8f50066b822d9cd10c4661b2`；ELU 原型只在本验证目录中使用。生产没有新增依赖、TensorFlow 调用、低精度或预算调整。

## 最终状态

| 项目 | 真实完整输入的结果 | 接入状态 |
|---|---|---|
| WMH CPU、T1 `crop=True` | nodecw7 官方/候选/候选/官方：全部标签、WMH 概率、33 个 CSV 数值、完整 header/affine 逐值相同 | 接入生命周期优化 |
| WMH CUDA `crop=True` | 硬限额 20,000,000,000 B；三个输出文件 SHA 与旧 GPU 完全相同；57 次实际 cuDNN 卷积 kernel 和三次 CNN 输入 SHA 相同 | 接入已验证的缓冲调度 |
| WMH CUDA 默认 `crop=False` | 原调度完整输出与旧 GPU 文件 SHA 相同；同实例 full→crop→full 状态恢复 | 保留原 forward/decoder/概率调度，**仍未达到 20 GB** |
| SynthSR CPU | 首层 Conv+bias 相同，原 ELU 是第一处分歧；诊断可恢复首层逐值同，第二 Conv 布局及随后 BN 仍需处理 | 保持成熟实现；默认浮点门仍失败 |

最终源码冻结为服务器 `workspaces/smri_cpu_20261004/remaining_20261004/synth/source_v10`。`model.py` SHA `0896ad2ac73e1f2ce110579cfb71719df1a45e9e422034c7127c59a488baf9cc`，`pipeline.py` SHA `b0cfe39ee0c1fc2a070cdba90265e9b6fbc4f652b484ed7b9db307c5a9fc3643`。空间代码 SHA `d81416104064110511ee0f073998b3e19959748c74ad88675d90488a261db8a6` 不变。

## 输入、资源与运行位置

- 真实数据为 CC0 OpenNeuro **ds003138 v1.0.1** 原始 T1，匿名为 `case01`；原网格 224×288×288，间距约 0.8×0.777778×0.777778 mm。输入文件 SHA `afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19`。影像、中间张量和模型只留在服务器。
- WMH checkpoint 大小 **790,531,383 B**，SHA `0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988`；SynthSR general 大小 **106,163,752 B**，SHA `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b`。现场核对 FNIT 固定 Release 资源清单；原作者来源资源复用已核验的私有文件，不随此提交分发。
- 现场读取统一入口 `FNIT/README.md`、`INDEX.json`。源码、输出、日志与转移包分别放统一入口的 `workspaces`、`runs`、`logs`、`archive/transfers` 下的 `smri_cpu_20261004/remaining_20261004/synth`。既有 Conda prefix 和冻结源码保持实体位置。
- CPU：nodecw7，Xeon Gold 6418H；8 物理核 `0,4,8,12,16,20,24,28`，OMP/MKL/Torch 8 线程；同组锁 `runs/smri_cpu_20261004/nodecw7.synth.cpu8.lock`。新完整作业没有继续使用 nodecw10。现场 AVX2、AVX512、FMA 能力用于诊断，硬件有 FMA 不等于 TensorFlow 当前 ELU 实际用了融合乘加。
- GPU：gpucw1 H100、Torch 2.5.1、FP32 网络、原 TF32 策略，无 autocast；GPU0 UUID `GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e`，所有 worker 使用共享 `gpucw1.gpu.lock`。最终完整验证初始物理 free 约 59–62 GB。共享外部进程约 19,664 MiB，利用率波动，时间不能证明稳定性能保证。

## WMH 修改与精度证据

checkpoint 在 CPU strict-load 后一次迁移到目标设备，避免构造时重复驻留 GPU 模型与 checkpoint。无梯度、eval、没有容器观察 hooks 的推理逐个消费既有 GroupNorm/Conv/LeakyReLU，及时释放已用 skip 和 cat；CUDA 按通道块填充同一 nearest+cat 张量。训练、有梯度及容器/global hooks 仍调用原 Module 容器，叶子算子 hooks 仍由其 `__call__` 分派。网络参数、GN、Conv 算子和概率累加次序没有改成近似算法。

GPU crop 提前分配 33 类保留概率缓冲，并在第一次 softmax 后释放第一次预测。大 GroupNorm 后释放 allocator **inactive cache**，让后续原 Conv 能在相同预算内获得原 workspace。中间 v6 完整 crop 虽然跑完，却有 25 个标签差异；实测 Conv 16/35/54 的 kernel 与旧路径不同。v6 不限额控制恢复旧三个文件 SHA；v8 释放上述 inactive cache 后，20 GB 下也恢复同一 57 个卷积 kernel。最终 v10 的[完整数值与 kernel 门](reports/wmh_final_gpu_acceptance.public.json)再次通过。

| 完整 T1 crop | peak allocated（B） | peak reserved（B） | 三文件 SHA / 实际 kernel |
|---|---:|---:|---|
| 原成熟 GPU、隔离不限额参考 | 32,633,949,696 | 45,267,025,920 | 参考 |
| 最终 GPU、20,000,000,000 B 限额 | 18,373,921,792 | 18,438,160,384 | 全相同 / 57 次全相同 |

输出网格 179×224×178。分割 SHA `ca94a9caf118205a426a1dc7ad468b013716bd9b452b8a441f1258a053d1672f`；WMH 概率 SHA `cc4cc54d9bcd0d20162f154b7e6b4c7216c723d8eed5d1537c654d75137cc893`；匿名 CSV SHA `c26284e2fe67368f58e1d771e8a8f83fd21db9c18552b8835fc4efaecfbd6cf7`。CSV 第一列统一为同一匿名输出路径后比较；33 数值列直接比较。

### 默认不裁剪的边界

v8 `crop=False` 在 20 GB 内跑完，但 8,981,504 个体素中有 **1 个标签差异**（背景/CSF），概率 max `0.0004196167`、RMSE `3.2041e-7`，CSV max `0.06 mm³`；Conv 16/35 kernel 不同。[失败门](reports/wmh_nocrop_source8_delta_v1.public.json)完整保留。因此最终 GPU 默认不裁剪采用原始网络和概率调度。最终同实例[full→crop→full](reports/wmh_source10_mode_reuse_uncapped_v1.public.json)三个完整结果均与各自旧 GPU 三文件 SHA 相同，调用后原模式恢复；异常恢复有本地回归检查。

不裁剪原路径峰值 allocated **37,997,219,328 B**、reserved **43,136,319,488 B**。这次检查是明确标注的不限额数值控制，不能算 20 GB 通过。生产没有偷偷放宽限额；在 20 GB 环境需显式使用 `crop=True`，裁剪可能移除视野边缘。直接注册观察 hooks 的原容器路径也不承诺达到优化后的峰值。

### 完整与阶段耗时

CPU 官方/候选/候选/官方为 **172.064 / 77.134 / 83.876 / 107.942 s**；同核组、同输入、冷进程含读图、模型加载、计算与保存，官方采样 RSS 约29.992 GB、候选约16.778 GB。[nodecw7 完整对照](reports/wmh_cpu_node7_abba.public.json)包含精度、load 和全部实际资源设置。节点 load 99–136，故不推为普遍加速比。v5 的 CPU 算术路径与最终版本相同；v8/v10 后续改动是 GPU cache 或 GPU no-crop 分派，没有重复重跑官方 CPU。

以下 crop ABBA 使用无 profiler、无 tensor hash 的冻结 v5 harness；v8 和最终 v10 的优化 crop 算术相同。完整 wall 是 worker 冷进程，含 import、CUDA probe、SHA 核验、报告生成、保存，**不是 CLI 专属时钟**。GNU time 在 flock 内启动，排队时间不计入。官方旧 GPU 两次参考超20 GB；不能称同预算速度比较。

| 顺序 | 完整 worker wall（s） | 构造（s） | API（s） | 保存（s） |
|---|---:|---:|---:|---:|
| 旧、不限额 | 12.85 | 1.266 | 4.842 | 1.433 |
| 新、20 GB | 13.21 | 1.333 | 5.199 | 1.452 |
| 新、20 GB | 13.18 | 1.472 | 5.375 | 1.436 |
| 旧、不限额 | 13.32 | 1.390 | 5.364 | 1.402 |

四次三文件 SHA 全相同；各次 synchronized CNN 内部时间与 GNU time 原记录见 `reports/wmh_source8_crop_abba_*.json`。共享 GPU 43–100% 利用率，不能断言稳定提速或性能不退化。真实中央 96³ 子体积的旧/新输出同 SHA，[最终小视野记录](reports/wmh_source8_subset96_candidate20.public.json)只用于较小输入回归，不替代上述完整原始 T1 验证。

早期 GPU 物理 free 约15 GB 时的大输入 OOM，以及个别初始化失败，保留在 `wmh_large_crop_candidate_v*.json`。allocator hard cap 与物理 free 分别记录；不明初始化错误没有被改记为精度失败或预算失败。最终使用 bare-Torch 初始化及12元素 FP32 kernel probe 后才导入 harness，不改生产 CUDA 初始化。

## SynthSR 第一分歧

固定官方 TensorFlow 2.13.1、同一 HDF5、真实第一 CNN 输入 `[1,1,192,224,256]`。第一 Conv+bias 全 **264,241,152** 个值相同；原 Torch ELU 有 **32,371,573** 点不同，max `5.96046e-8`。[首层报告](reports/sr_first_candidate_v1.public.json)保留布局控制。官方单独 eager/compiled ELU 与记录结果同，排除该处 Conv/ELU 融合假说。

按实际安装的 TensorFlow `relu_op_functor.h` 与 Eigen `GenericPacketMathFunctions.h` 核对 ELU 为 `exp(x)-1`，逐步 FP32、非融合 multiply/add 的 packet 多项式恢复整个第一 ELU 全部逐值同。[完整首 ELU 门](reports/sr_whole_first_elu.public.json)通过，4.504 s 是逐 plane 诊断算术时间；不同版本 Eigen 公式、`expm1` 或只按数学关系替换均不充分。真实中间 plane 上 FMA/addcmul 仍有810点1 ULP差，不能凭 CPU 支持 FMA 就采用融合实现。

实际安装 header SHA：`relu_op_functor.h` 为 `6f3ff2daefa98fb557b04a1f0deb3b197465456d90c2db67f6b8364c94e15935`，Eigen generic 为 `a81ac6fd0a3cb4b21d2b8f0873296b64cd0efe917e318b8e2778f15737431f04`，AVX 为 `fdb6c29674a1cb0d301b1790e641756b083470ca993c1d1c2f421aaaf9752ec1`，AVX512 为 `ab26ccb3807434905f30741f5ec50a3528902f6c4eb60a18c02c7f6bfeef5413`。仅归档身份和自有数值原型，不复制原软件无关源码。

在整个官方第一 ELU 输入上，[第二 Conv 定位](reports/sr_second_candidate_node7_v2.public.json)显示：contiguous 有254,863,410点差，max `9.53674e-6`；输入和权重同时 channels-last 可使第二 Conv+bias 全值同，配合诊断 ELU亦同。随后首 BN 仍有147,511,150点差、max `1.90735e-6`；subtract-first/scale-bias 控制也未全同。因此不能把首 ELU 已匹配写成完整网络已匹配。

一次完整 ELU 原型（source_v7）默认 CPU API **66.815 s**，两次 CNN **32.204/30.146 s**；固定 `rtol=1e-5, atol=1e-3` 仍有 **1,935/9,072,000** 点失败，max `0.0193863`，量化457点差1。[拒绝记录](reports/sr_cpu_rejected_prototype.public.json)保留它。成熟默认仍是3,522点浮点失败、max `0.0191345`，量化527点差1且通过既有量化容差；失败且更慢的原型没有进入默认，也没有改 GPU SynthSR。

## 复现与参数

公开 API、输入/输出结构、所有功能参数、原软件命令和既有真实脑图见 [WMH 功能说明](../../../docs/wmh_synthseg/README.md)与 [SynthSR 功能说明](../../../docs/synthsr/README.md)。原始 WMH 对照由隔离官方 `mri_WMHsynthseg --device cpu --threads 8 --crop --save_lesion_probabilities --csv_vols ...` 执行；候选运行没有调用原软件。

```bash
# 调用方先设置 CUDA_VISIBLE_DEVICES 为指定 GPU，并取得共同 GPU flock。
# candidate_source_path 指向冻结候选 src；每次 output_directory 必须是新目录。
PYTHONPATH="$candidate_source_path" python wmh_gpu.py \
  --input "$input_image_path" --weights "$verified_weight_path" \
  --output-dir "$output_directory" --source-revision "$source_revision" --crop
```

`wmh_gpu.py` 的 `--input` 是真实3D NIfTI，`--weights` 是大小/SHA匹配的 checkpoint，`--output-dir` 保存标签、WMH概率、匿名CSV与 JSON，`--source-revision` 记录冻结身份，`--crop` 选择裁剪。`--profile-kernels` 捕获实际 cuDNN kernel，属于诊断计时；`--diagnostic-hooks` 采样 GN 显存。默认硬限额20,000,000,000 B；`--reference-uncapped` 仅用于隔离旧代码参考，严格核对旧 model SHA；候选不限额诊断还须显式 `--uncapped-control`，报告预算为null。`wmh_mode_reuse.py` 是不限额同实例模式切换控制，参数同前四个且不支持20 GB声明。

`wmh_cpu.py` 另接受 `--reference-seg`、`--reference-lesion`、`--reference-csv`，指向相同真实输入的隔离官方保存参考。`compare_wmh.py` 给出逐标签 Dice、概率 MAE/RMSE/max 和每列有符号体积差。`sr_first_layer.py` / `sr_second_layer.py` 的 `--arm reference|candidate` 分别使用隔离 TF 或生产Torch，真实中间数组留私有目录。`sr_cpu.py --elu-prototype` 才显式加载验证目录原型；生产从不导入它。各脚本 `--help` 给出路径参数。所有阶段数组比较覆盖完整张量，阶段秒数均非完整 CNN benchmark。

本地验证：`PYTHONPATH=src python -m pytest -q tests/wmh_synthseg tests/synthsr`：**42 passed、3 skipped**，跳过项为既有可选 CUDA/权重测试。真实GPU精度门由上述服务器完整结果补足。

## 历史与参考

- v1–v5：生命周期、nearest 分块与概率保留顺序；CPU完整门通过，小视野GPU逐值同；早期完整GPU物理显存受外部进程限制。
- v6：完整crop20 GB跑完但actual kernel/数值改变，拒绝；不限额对照确认inactive缓存/卷积workspace关联。
- v7：只用于失败的SR CPU ELU完整原型，随后恢复成熟SR。
- v8：crop20 GB完整三文件SHA及57kernel通过；no-crop20 GB数值门失败，保留报告。
- v10：最终crop保持v8；GPU默认no-crop恢复原decoder/forward/probability调度，同实例切换验证通过。

WMH 原实现：[FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg/WMHSynthSeg)，Laso et al. ISBI 2024，doi [10.1109/ISBI56570.2024.10635502](https://doi.org/10.1109/ISBI56570.2024.10635502)。SynthSR 原实现：[FreeSurfer SynthSR](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthsr)，Iglesias et al., NeuroImage 2021 / Radiology 2023；ELU 参考 [TensorFlow 2.13.1 functor](https://github.com/tensorflow/tensorflow/blob/v2.13.1/tensorflow/core/kernels/relu_op_functor.h)，实际随安装编译的 Eigen header 身份以上述 SHA 为准。
