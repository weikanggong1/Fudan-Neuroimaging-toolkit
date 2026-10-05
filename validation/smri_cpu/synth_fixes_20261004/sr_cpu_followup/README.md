# SynthSR CPU 精度修复与融合诊断（2026-10-04）

基于已交付 WMH 修复 `626cab69555b2c281245268cf2f948da3a2cbb82`，先在验证目录定位成熟 SynthSR CPU FP32 尾差，再冻结受保护生产候选 `source_sr_v2`。原 `model.py` SHA `290d0ae911d8c99f4f87c22936ca497ea45bb57b8f50066b822d9cd10c4661b2` 是本轮旧参考；候选只新增 CPU 分派，原GPU forward正文 AST相同。项目已有 `numba>=0.59`（`pyproject.toml`、`environment.yml`），无新依赖、生产TensorFlow或低精度。诊断脚本不被生产导入。

## 已完成的真实阶段门

与[上一轮首差报告](../README.md)使用同一个 CC0 原始T1、相同校验的权重及私有中间数组。nodecw7固定8核 `0,4,8,12,16,20,24,28`，Torch/Numba 8线程，沿用同一专属锁。

| 阶段 | 完整真实数组 | 数值结果 | 实测阶段秒数 |
|---|---:|---|---:|
| Numba ELU，`fastmath=False` | 264,241,152 | 全值与官方首ELU相同 | 0.510 |
| 首BN，用官方rsqrt因子验证公式 | 264,241,152 | 线性分开乘加全同；FMA/先减均不同 | 3.218（逐plane诊断） |
| 自有SSE seed+逐FP32 Newton rsqrt | 24通道 | 所有官方rsqrt因子逐值同；Torch rsqrt/1sqrt有13个不同 | 因子规模，不作benchmark |
| 自有rsqrt与Numba BN affine | 264,241,152 | 所有官方首BN逐值同 | 0.610 |

[完整ELU与公式门](reports/sr_numba_elu_bn_node7_v2.public.json)、[自有BN门](reports/sr_numba_bn_node7_v1.public.json)保留算术和helper身份；阶段时钟不是整个CNN。旧逐Torch临时张量首ELU诊断为4.504 s，完整慢原型仍见前一报告。

## 官方实际 BN graph

TensorFlow2.13.1官方 `unet_bn_down_0` 的 `fused=False`，epsilon0.001。实际concrete inference graph 是以下逐步FP32顺序：

```text
inverse = rsqrt(variance + epsilon)
scale = inverse * gamma
offset = beta - mean * scale
output = input * scale + offset
```

[实际节点与属性](reports/sr_bn_graph_node7_v1.public.json)直接从隔离官方 layer 提取；各通道数值因子只在服务器私有NPZ，不随仓库分发。图中定义的是非融合乘加；首BN完整参考同时证实该顺序。TF这个rsqrt与Torch/math sqrt的尾差是当前可复现分歧；独立原型按观察到的packet seed和一步Newton、明确非融合FP32操作核对。SSE intrinsic仅用于本次x86 CPU诊断，不宣称任意硬件实现等价。当前helper在导入时要求Linux x86-64及SSE2，且Torch入口拒绝GPU、其他dtype和有梯度输入；生产SR从不导入此目录。v2仅增加平台/ISA门，算术与v1相同。

[实际JIT证据](reports/sr_isa_node7_v2.public.json)记录ELU/rsqrt的LLVM及汇编SHA：`llvm.fma`、fast/contract算术和汇编FMA指令均为0；rsqrt实际指令为`vrsqrtss`。报告只覆盖自有多项式/Newton主体，不推论无关libm内部。

## 一次完整 CPU 诊断：仍失败，不接入

阶段门通过后执行一次完整同源T1：输入和权重同时channels-last 3D，ELU和BN使用本目录自有Numba原型。原始归一化/翻转/后处理/保存由成熟公开API完成；官方完整输出复用既有保存参考，不重复官方CNN。

| 项目 | 成熟默认 | 本次融合诊断 |
|---|---:|---:|
| 固定 `rtol=1e-5, atol=1e-3` 超门槛体素 | 3,522 /9,072,000 | **172 /9,072,000，未通过** |
| 浮点 max | 0.0191345 | 0.00682068 |
| 浮点 RMSE | 9.82070e-5 | 2.98175e-5 |
| 量化不同体素（max1） | 527 | 192 |
| 量化affine | 官方一致 | 官方一致 |

[完整失败门](reports/sr_numba_full_node7_v1.public.json)保留结果。构造0.541 s、JIT warmup0.706 s、两次CNN10.264/7.600 s、公开API22.288 s、保存NIfTI+NPZ2.021 s。warmup用微小数组仅编译算子，不是benchmark；API不含JIT、保存或posthoc比较。这次worker没有采集完整RSS峰值，不能事后补推峰值，也不能与不同节点/时段官方CLI计算稳定加速比。**未通过原固定浮点门，原型没有进入默认，GPU仍使用成熟原forward。** 首BN后MaxPool全33,030,144值同；24→48卷积raw/ELU全66,060,288值同，关闭MKLDNN并不匹配。随后对两个原始flip CNN输入逐层追踪完整下采样路径；输入和Torch CL重排的规范数组hash与参考逐一比较。

## 文件与参数

- `elu_numba.py`：独立ELU严格FP32、无fastmath、保留CPU contiguous/CL3D布局，拒绝autograd/GPU。
- `bn_numba.py`：诊断rsqrt seed/Newton与非融合affine；当前单volume CPU输入，不是公开API。
- `bn_graph_reference.py`：`--official-script` 指实际官方python脚本，`--weights` 校验原checkpoint，`--output-dir` 为新的私有目录；输出graph JSON和仅服务器使用的24通道因子NPZ。
- `stage_candidate.py`：`--first-dir`、`--second-dir` 指官方第一/第二层数组，`--factors` 指私有因子，`--output-dir` 保存匿名统计。v1在1D因子统计的harness索引处失败，日志保留；v2修复维度后完整通过。此失败未记作精度失败。
- `bn_gate.py`：另接受 `--weights`；先因子门再完整BN门，只有全同才返回complete。
- `cpu_inference_numba.py`、`full_gate.py`：显式CPU诊断forward；后者接受 `--input`、`--weights`、`--reference-dir`、`--output-dir`、`--source-revision`，输出标签强度NIfTI、浮点NPZ和固定门JSON。影像留服务器；仓库仅报告SHA和标量。
- `third_layer.py`：后续firstdiff阶段，`--arm reference|candidate`、`--second-dir`、`--reference-dir`（候选）、`--weights`、`--official-script`（参考）、`--output-dir`；比较完整pool及24→48卷积。参考在隔离TF环境，候选仍Torch，诊断秒数不当作完整推理性能。

`down_trace.py` 使用 `--network-input-dir` 指保存的两个实际CNN输入，另接受 `--arm`、`--weights`、`--official-script`（参考）、`--reference-dir`（候选）、`--output-dir`；TF从原模型提取下采样层输出，候选比较规范数组SHA，首次分歧才输出额外标量和私有数组。该图新增中间输出并剪去上采样部分，会影响算子融合，不能代表原始整图的逐层执行。`isa_evidence.py --report` 保存JIT证据，重编译微小输入只验证编译属性，不能作性能benchmark。

## 原整图的融合证据与参考不变

[`full_graph_reference.py` 的实际执行报告](reports/sr_full_graph_reference_node7_v1.public.json)只执行原始完整模型的最终输出，未提取中间层、关闭remapping或替换算子。两个实际flip分支共22,020,096个网络输出值，全部与冻结参考逐值相同：

| 分支 | 冻结NPY文件SHA-256 | 原整图结果规范数组SHA-256 |
|---|---|---|
| 0 | `8a789b66686deb69ba06aa6551608568eb28680dc2c56005f2b074d3d76f2bf6` | `10f6bac6f3e2de425ba33e649be297b4e157032ebd808429227c064bfda5a7b4` |
| 1 | `5c9b6552623a3a01a7008e4c4ef252b44f105495263e35145f7c99376733e8ef` | `708af41301fabc12b7cb51313bd7eeeef9216a49731f8cbfdc7c59baa1b0f790` |

原整图CPU profiler记录18个中间卷积为 `Elu:_MklNativeFusedConv3D`，末层为 `_MklNativeConv3D` 和独立BiasAdd。该结果保留原参考口径；profiler耗时不作速度benchmark。

新增中间输出的下采样图中，两分支输入hash全同，候选首次不同均为 `unet_conv_downarm_0_1`。但[隔离raw+ELU对照](reports/sr_fusion_compare_node7_v1.public.json)中同层264,241,152个raw与ELU值又全部相同。[融合控制图](reports/sr_fusion_reference_node7_v1.public.json)进一步显示，仅输出前两层的图使第二层变成融合Conv+Bias与独立Elu；关闭remapping又得到另一组不同SHA。因此这些控制只能说明输出提取改变了数值执行路径，不能把该层宣布为原整图的首个分歧，也不能用关闭融合的图替换验收参考。

仅在验证目录测试oneDNN融合ELU的具体FP32公式。[现场runtime/注册op记录](reports/sr_onednn_metadata_node7_v2.public.json)已确认oneDNN 2.7.3、threadpool8、AVX-512，微数组仅让runtime打印版本和post-op能力，不作精度/性能benchmark。该记录和[官方版本源码URL/大小/SHA](reports/sr_onednn_source_provenance.public.json)相互核对，源码实体留在私有outputs。TensorFlow的构建清单还列出可选3.1，其ELU公式相同；本次数值依据是实际2.7.3。

自有 `elu_onednn_numba.py` 使用明确的FP32 FMA，和前面的严格非融合Eigen helper分开。它最初为数值假设，后续实际融合阶段及完整冻结整图门分别通过，结果见下。既有Numba/LLVM依赖足够，无新C++构建或生产TensorFlow依赖。`onednn_fused_layer.py` 用实际注册op执行完整真实24→24层，分别建融合Conv+Bias和Conv+Bias+Elu的独立graph，并捕获profiler；候选从同一TF raw输入和成熟Torch卷积输入分别比较，以区分ELU与卷积累加。`--first-dir` 指已捕获首ELU，`--reference-dir` 为候选所用显式融合参考，其他`--arm`/权重/原脚本/输出参数与上述阶段门相同。该输入只定义隔离算子对照，不能称为原整图此层的实际输入。[三次私有op绑定失败](reports/sr_onednn_harness_failures.public.json)发生在计算前，日志原样保留；重试分别修正合法节点名、显式默认属性和实际注册的MklNameChangeOp kernel label。

后续 `full_gate_onednn.py` 只在阶段门通过后运行，参数为`--input`、`--weights`、`--reference-dir`（同时包含冻结最终影像与两分支CNN输入/输出）、`--output-dir`、`--source-revision`。它保留固定浮点门、量化不同体素及affine门，并独立核对完整CNN输入/输出SHA；50ms `/proc` RSS采样和rusage高水位分别报告构造/JIT/API/保存/比较。为后续hash保留的网络输入/输出view会计入RSS，保留字节数另列；不把比较耗时计入API。隔离融合op的全等与最终完整CNN的全等分别记录，前者不能替代后者。

### 真实完整融合算子阶段门：全值通过

[显式融合参考](reports/sr_onednn_fused_reference_node7_v4.public.json)记录实际oneDNN `brgconv:avx512_core`、输入/输出NDHWC、权重32通道内部block，以及 `eltwise_elu:1` post-op；所有实际数据均为FP32，ISA能力打印中的float16/BF16名称不表示使用低精度。完整数组为`[1,24,192,224,256]`，264,241,152个值。

[候选阶段门](reports/sr_onednn_fused_candidate_node7_v1.public.json)的三项均逐值全同：成熟Torch CL3D卷积raw与TF融合Conv+Bias；该raw经过自有FMA ELU与真实融合Conv+Bias+Elu；TF raw经过同一ELU与该融合输出。ELU规范数组SHA均为`04795563e1d371f3fe1b9306d4c6b717b154c54bf3f3f5391e2286bdacdee980`。

该1GB真实数组的Torch卷积阶段1.303 s，首ELU调用0.996 s（含JIT），第二次warm ELU0.494 s。独立参考的Conv+Bias包装1.440 s、Conv+Bias+Elu包装1.196 s，两者包含profiler；这些时钟仅描述隔离阶段。实际LLVM出现明确FMA，汇编有6个FMA指令，`fastmath=False`，ISA flags及LLVM/汇编SHA已记录；前面的非融合Eigen/Newton无FMA证据仍独立成立。

完成阶段门后才运行一次完整原始T1原型，随后对生产候选完成同源真实整图门及旧新回归。原始整图参考始终未改；GPU继续执行原forward正文。

所有源码冻结在统一FNIT `workspaces/smri_cpu_20261004/remaining_20261004/synth/sr_cpu_followup`；输出在 `runs/.../synth`，日志在 `logs/.../synth`。平台门/ISA证据另冻结在 `sr_cpu_followup_v2`，不覆盖既有v1科学记录所用helper。WMH `source_v10` 和已交付分支保持冻结。

## 最终完整原型与生产验收

一次完整原型的[真实门](reports/sr_onednn_full_node7_v1.public.json)通过：两个完整CNN共22,020,096值逐值且数组SHA同原whole oracle；最终9,072,000浮点值全同、固定`rtol1e-5/atol1e-3`失败0，量化全同。API19.677 s、两CNN8.064/7.289 s、JIT1.370 s、两份保存1.981 s、含评分GNUwall28.96 s；采样API RSS7,480,295,424 B，包括保留网络view176,160,768 B。该原型时钟不替代后续生产ABBA。

### 生产范围与源码绑定

生产候选只修改 `src/fnit/synthsr/model.py`，新增 `_cpu_inference.py` 与 `_cpu_math.py`。复用原Torch conv/pool/interpolate、参数和接口；临时CL3D权重视图不保留副本，参数身份、内容、stride、版本、BN buffers与RNG保持。CPU eval/F32/单volume/no-grad/Linux x86-64/SSE2/FMA/MKLDNN才启用；训练、grad、CPU autocast、注册局部/全局hooks、非原叶模块、其他dtype/平台使用原调用。NumBa仅此CPU路径延迟导入，线程mask不超过调用方既有NumBa与Torch预算，finally恢复；不改变CUDA后端策略。

| 冻结 `source_sr_v2` 文件 | SHA-256 |
|---|---|
| `model.py` | `aa8c577eb728b68b90aa7703153c7156c3bad61e78e5803d0f295b94d60452b5` |
| `_cpu_inference.py` | `2cbcbbf65b87c827bf7fa3665a508a4be5b1844d1de8474b1da1d1a1e430a148` |
| `_cpu_math.py` | `b643d821b1c84d746656deb6809799dbd18a279dc82a000e710dfae9d9add18e` |

生产目录与WMH `source_v10`分开冻结。全部输入/权重/driver/source SHA和边界守卫见[最终汇总](reports/sr_production_source_summary.public.json)。本地与协调者分别运行 `PYTHONPATH=src python -m pytest -q tests/synthsr`，均22 passed/3 skipped（15.31/10.23 s）；小张量测试只验训练、hooks、autocast和状态，数值验收取完整真实数组。

### CPU 生产 ABBA：完整精度

[旧1](reports/sr_production_cpu_abba_node7_v1_old1.public.json)、[新1](reports/sr_production_cpu_abba_node7_v1_new1.public.json)、[新2](reports/sr_production_cpu_abba_node7_v1_new2.public.json)、[旧2](reports/sr_production_cpu_abba_node7_v1_old2.public.json)在同nodecw7核组/8线程/专属锁串行执行。两新臂都保持原CNN输入SHA，完整预测SHA分别为上述`10f6bac6…`和`708af413…`；最终float与uint8逐值全同官方、affine同。NIfTI SHA `dff4ef9d658243a500704c141290c636414e7e871e3d6fccf5b2f211a7328922`，NPZ SHA `fba3de5d788d11a149db26fbda720c8c3d25f69a26cb91083bc33776487e55f3`。旧两臂均保留3,522点浮点失败与527个uint8差1。

| 同driver阶段（s） | 旧1 / 新1 / 新2 / 旧2 |
|---|---|
| 构造 | 0.969 / 0.695 / 0.681 / 0.327 |
| JIT warmup | 0 / 1.277 / 0.246 / 0 |
| 两CNN | 12.299+9.442 / 12.194+12.033 / 12.406+12.731 / 15.070+14.428 |
| 完整API | 25.979 / 28.487 / 29.443 / 33.792 |
| NIfTI+NPZ保存 | 2.063 / 1.897 / 1.911 / 1.921 |
| 评分比较 | 1.082 / 1.273 / 1.247 / 1.070 |
| worker GNUwall，含全部冷导入/JIT/保存/评分 | 33.84 / 36.62 / 36.68 / 39.74 |

旧API50ms采样RSS约10.219GB、新约7.485GB（十进制）；双方都保留176,160,768 B的网络view，用于API之后评分。子CNN嵌套于API，不重复累加。此轮候选相对旧API基本持平，不用较早原型19.677 s替代生产结果。

### 正常 CPU CLI：同机官方对照与冷JIT

[`cpu_cli_abba.sh`](cpu_cli_abba.sh)运行真实公开CLI，仅保存正常NIfTI；每次新进程，候选每次独立空NumBa缓存。无网络wrapper、CNN数组复制、NPZ保存或运行内哈希。GNUwall从flock内启动，含正常导入/模型加载/JIT/推理/写盘，排队不计入；[`collect_normal_cli.py`](collect_normal_cli.py)的后续全图/header/SHA比较不计入。

[完整正常CLI报告](reports/sr_normal_cli_node7_v1.public.json)：官方1/旧1/新1/新2/旧2/官方2为148.68/28.24/29.91/26.77/29.08/75.17 s。中位官方111.925、旧28.660、新28.340 s；GNU峰值RSS分别最大8.780/10.222/7.496 GB。候选与官方连NIfTI文件及header SHA全同。官方GPFS冷导入/缓存波动明显，节点load约24–28；不据此断言稳定4倍加速。旧新正常CLI基本持平，候选RSS约降低26.7%。旧nodecw10时钟仅属历史，不与本次除比。

### CPU 参数分支

[`cpu_function_gate.py`](cpu_function_gate.py)只运行候选，复用原官方保存数组：v1、lowfield、disable_flipping、disable_sharpening各9,072,000个uint8值均全同，保存affine/header以及NIfTI文件SHA同官方；原量化exact≥99.99%、max≤1、MAE≤1e-4门均通过。旧差异分别520/628/668/397点。没有这些模式的原浮点oracle，故不声明其浮点逐值验收。

[参数分支结果](reports/sr_functions_node7_v2.public.json)明确保留评分修正：v1 harness将写盘前float64 affine与官方保存后float32 NIfTI sform直接比，四例raw affine flag为False；[`rescore_function_gate.py`](rescore_function_gate.py)在同一保存边界重新评分，未运行任何CNN。原报告SHA和raw flag仍保留，这不是生产空间错误，也没有放宽数值门。

### 不同真实输入与尺寸的最小扩展

[`cpu_domain_gate.py`](cpu_domain_gate.py)只运行冻结候选，复用既有官方保存结果；同nodecw7核组/8线程/锁。没有重跑原软件，也没有将格式/目录等同输入组合重复重建。[唯一域结果](reports/sr_domains_node7_v1.public.json)保留原gate、参考SHA和最新输出：

| 唯一真实场景 | 完整值数 | 本次结果 | 旧结果 |
|---|---:|---|---|
| 第二幅公开原始T1 | 9,072,000 | uint8/affine/header/NIfTI文件SHA全同 | 563点差1 |
| 已有公开衍生FLAIR | 6,007,862 | uint8/affine/header/NIfTI文件SHA全同 | 399点差1 |
| SynthRAD2023真实HU CT | 11,498,256 | uint8/affine/header/NIfTI文件SHA全同 | 747点差1 |
| 真实64mT MRI，低场模型 | 7,848,000 | uint8/affine/header/NIfTI文件SHA全同 | 610点差1 |
| 真实EPI两帧取首通道 | 7,866,240 | uint8/affine/header/NIfTI文件SHA全同 | 358点差1 |
| 同一真实b0的NPZ/单位affine约定 | 979,200 | 原`rtol1e-5/atol1e-3`全通过；922,432值有尾差，max0.000339508、RMSE1.76756e-5 | 962,849值不同，max0.001571655、RMSE6.87055e-5 |

各NIfTI场景原量化门exact≥99.99%、max≤1、MAE≤1e-4保持；本次实现达到全值相同。NPZ没有affine/header，保留浮点特例和实际非逐值结果。各场景API时钟仅为候选一次功能检查，不与旧节点历史时钟计算提速；不会把此前9参数/7域格式的全部组合改记为本次重跑。

### CUDA 生产完整门与正常 CLI

[旧1](reports/sr_production_gpu_abba_v1_old1.public.json)、[新1](reports/sr_production_gpu_abba_v1_new1.public.json)、[新2](reports/sr_production_gpu_abba_v1_new2.public.json)、[旧2](reports/sr_production_gpu_abba_v1_old2.public.json)在同GPU0 UUID、公共GPU锁串行完成，显式20,000,000,000 B allocator cap（按total_memory为分母）。CUDA仍FP32/defaultTF32，cuDNN9.1/Torch2.5.1、benchmarkFalse/deterministicFalse；CPU数学与分派模块均未导入，实际conv参数stride完全一致。

新两臂及旧重复对旧1：两CNN共22,020,096值、最终9,072,000浮点和uint8、affine/header、NIfTI/NPZ文件SHA全同。CNN规范数组SHA `5cede6cc2e14b5ea791501c7137f524b1e29dad1b2751ab89006a0108eccc38e` 与 `eb61bd3548a2a82d636d5611ffaf63eabeb47bedf0934aad0d3209c2406a05cc`；NIfTI SHA `23634581c27ef5d1f20edbf596ae8fd0892029f89dde43c39d5349799e18ae68`，NPZ SHA `af3b4884a0851c6d141da3450aaf267578e7e0601d11dda6b82e63cacb7371c0`。各臂allocated10,006,443,008 B、reserved13,845,397,504 B完全一致，均小于20GB预算。

GPU旧1对官方CPU另有TF32差异（122,005个uint8体素，max3），只作为原基线，不宣称GPU与官方CPU逐值相同。CUDA不退化门以对应旧GPU参考为准。

| CUDA时钟（s） | 旧1 / 新1 / 新2 / 旧2 |
|---|---|
| CNN原图+翻转，显式同步 | 0.549+0.401 / 0.550+0.405 / 0.533+0.403 / 0.550+0.331 |
| 诊断API，含同样追踪复制 | 7.001 / 6.234 / 6.463 / 7.256 |
| 诊断GNUwall，含私有数组保存和评分 | 19.43 / 16.05 / 15.59 / 17.79 |
| 正常CLI冷bootstrap+NIfTI，无追踪/额外保存/评分 | 12.14 / 11.36 / 11.29 / 11.25 |

正常CLI由[`gpu_cli_abba.sh`](gpu_cli_abba.sh)与[`gpu_normal_cli.py`](gpu_normal_cli.py)调用实际public CLI；小bootstrap仅设置20GB cap、明确初始化及12元素probe，计入GNUwall，不改生产初始化。后续[`collect_gpu_cli.py`](collect_gpu_cli.py)评分不计入正常CLI时钟。[正常CLI报告](reports/sr_normal_gpu_cli_v1.public.json)的NIfTI/header/SHA及显存也全同。外部GPU利用97–100%，因此这些时钟只作配对观察，不能据此保证独占环境性能或稳定加速；数值、原forward与显存不退化证据独立成立。

### 统一路径与复现

诊断源各版本`sr_cpu_followup_v1`至`v9`、生产`source_sr_v1/v2`、CLI`sr_cli_v1`、参数`sr_functions_v1/v2`冻结在FNIT统一`workspaces/smri_cpu_20261004/remaining_20261004/synth`；实际输出在对应`runs/.../synth`，日志在`logs/.../synth`，权重/环境prefix/旧WMH源码未移动。NumBa JIT缓存只在自身私有输出目录；模型、影像、临床ID、许可证和凭据不提交。仓库只有自有脚本、匿名数值、SHA和中文说明。

```bash
# candidate_source_path 为冻结source_sr_v2/src；调用者先取得对应CPU锁和8核预算。
PYTHONPATH="$candidate_source_path" python full_gate_onednn.py \
  --implementation production --input "$input_image_path" \
  --weights "$verified_weight_path" --reference-dir "$frozen_reference_directory" \
  --output-dir "$new_private_output_directory" --source-revision source_sr_v2
```

`--implementation original|prototype|production`选择验证分支，`--input`为真实原始扫描，`--weights`为按固定Release校验的HDF5，`--reference-dir`为只用于posthoc的原冻结CNN/输出参考，`--output-dir`为私有新输出目录，`--source-revision`记录身份。公开生产接口不读官方参考。[功能七节说明](../../../../docs/synthsr/README.md)列每个参数、数据/输出结构、原软件命令、最新及历史对照与既有脑图。
