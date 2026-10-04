# SynthStrip 与 SynthSR：CPU 官方对照

本目录验证公开单例 Python API 和 CLI，使用真实影像。生产 FNIT 仍只运行仓库代码；FreeSurfer 仅用于独立的原软件对照。

## 输入、版本和计时

初始源码冻结为 `1d31e7baaebbb644ab199471f7fe6282721455fd`。默认模型在两例公开原始 T1 上分别按「原版、FNIT、FNIT、原版」运行；其余参数使用同输入的原版/FNIT 功能对照。默认 T1 来自 [OpenNeuro ds003138 v1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1)，不以既有去面示例替代。既有公开 FLAIR 为衍生影像，单列其来源与范围。

主机为 nodecw10。两臂使用 8 线程，并限制到相同 8 个物理核：`0,4,8,12,16,20,24,28`。所有真实计算由协调者的同一 CPU 锁串行调度；记录完整进程墙钟、GNU time、进程树 RSS/线程数/亲和性、节点负载、退出码和输出存在性。等待锁的时间不计入命令墙钟。计时包含解释器启动、导入、模型加载、推理及全部指定输出的写盘。

`profile_inference.py` 另行记录模型构造、预处理、网络前向、重采样、连通域及写盘。它会增加函数包装和张量哈希的开销，其整个进程时间不用于计算 CLI 加速比。嵌套阶段已经包含在父阶段中，不能累加。`--cpu-channels-last` 是验证布局原型；SynthStrip 的生产 CPU 模块 `6b7aeafd` 已通过门槛并在 oneDNN 启用时使用此布局，SynthSR 原型没有通过候选门槛，生产网络保持 contiguous。CUDA 布局均未更改。

## 已确认的问题与控制

原始冻结版本在第一例 T1 上的脑掩膜 Dice 为 `0.9964284923`，有 `20,312` 个差异体素。原始数据解码、官方权重和归一化公式一致；问题来自共享 `new_image` 重写 qform 时重新计算了 `pixdim`。原有 `0.7777777910 mm` 被改成 `0.7777777314 mm`，使 `ceil(288 * voxel_size)` 从 225 变为 224，继而改变 1 mm 网格的中心。

共享修复提交 `dc2fc052` 在数据前三轴 shape 和 affine 不变时保留原 NIfTI 几何字段。第一例修复后的 1 mm 数组与归一化网络输入均逐元素等于官方。两例 T1 和无 CSF 模型的完整 CPU contiguous 输出均逐值等于官方；第一例有 `18,579,456` 个体素，三份输出差异数均为 0。生产 CPU 布局随后完成 26 次 CLI、11 个真实参数场景：脑图与 mask 逐值同，SDT 最大全场景差 `4.3392e-5 mm`，全部通过固定门槛。

SynthStrip 的 CPU 构造原先还会修改同进程其他模型的 CUDA 后端策略。本轮将这些设置限制到 CUDA 构造，CUDA 路径保留原有 TF32 与固定卷积算法设置。该修改与几何修复分别验证。

SynthSR 的 CPU channels-last 原型在第一例真实 T1 上缩短了网络耗时，但与原 CPU contiguous 的量化结果有 `465 / 9,072,000` 个体素相差 1，浮点最大差为 `0.0188980103`。它未通过预先固定的候选门槛，未进入生产实现。

### 完成结果与仍未通过项

| 范围 | 实测结果与匿名报告 |
|---|---|
| SynthStrip 生产 CPU，11 场景/26 CLI | 全部通过 mask/brain 逐值门槛和 SDT 固定容差；[逐项结果](reports/synthstrip_cpu_cli.public.json) |
| SynthStrip 默认两个原始 T1 | 官方/FNIT 完整 CLI 中位数 `45.561/14.901`、`49.199/14.027 s`；官方冷启动波动与卷积另列 |
| SynthStrip CPU layout 的 GPU 回归 | 同修复输入 ABBA 全部三份输出逐值同；组件 allocated/reserved峰值一致，[配对结果](reports/synthstrip_gpu_regression.public.json) |
| SynthStrip GPU TF32 对官方 CPU | 几何修复将 mask 差从 20,315 降至 59；仍未通过 CPU 的逐值 gate，[修复前](reports/synthstrip_gpu_official_before.public.json)/[修复后](reports/synthstrip_gpu_official_after.public.json) |
| SynthSR 当前 CPU 功能 | 9 个模型/参数场景 + 7 个域/格式场景全部通过官方量化容差或相应 NPZ 门槛；[参数](reports/synthsr_cpu_functions.public.json)/[域和格式](reports/synthsr_cpu_domains.public.json) |
| SynthSR 默认 case01 浮点 NPZ | `rtol=1e-5, atol=1e-3` 未通过；3,522/9,072,000 点超门槛、max `0.0191345`，[尾部报告](reports/synthsr_float_tail.public.json) |
| SynthSR 误差隔离 | 两次 CNN 实际输入逐值同，原始预测 RMSE `3.76e-7/4.01e-7`；同一官方预测回放经过 FNIT 后处理，9,072,000 个输出逐值等于官方，[回放控制](reports/synthsr_network_replay.public.json) |
| SynthSR BN 算式原型 | 两种 FP32 算式仍有 3,533/3,702 点超浮点门槛，未接入生产，[结果](reports/synthsr_bn_formula.public.json) |

[阶段观察](reports/cpu_stage_profiles.public.json)与完整 CLI 使用不同计时范围。GPU 设备记录中，整卡 `nvidia-smi` 显存包含其他保留 context，不能代替组件的 Torch 峰值；[设备采样](reports/synthstrip_gpu_device_load.public.json)已按实际目标卡 UUID 过滤。所有公开 JSON 只发布匿名参数标识、资源设置、标量与哈希；完整命令、私有输入、数组和模型留在服务器。

## 功能覆盖

| 功能 | 固定真实输入与检查 |
|---|---|
| SynthStrip 标准模型 | 两例原始 T1，重复完整 CLI；公开衍生 FLAIR另列 |
| `no_csf` | 官方自动选模型与 FNIT 权重目录解析；掩膜、脑图、距离场 |
| `border` | 1、2 mm及大阈值 SDT 扩展分支；大阈值只作功能检查 |
| `fill` | 默认与显式 -1；检查掩膜外填值和原网格 |
| SynthSR 模型 | 通用 v2、v1、低场权重及 v1/lowfield 同时指定时的优先级 |
| SynthSR 翻转与锐化 | 默认、关闭翻转、关闭锐化、同时关闭 |
| 输出格式 | 同次真实推理结果保存 NIfTI/MGZ；SynthSR另保存 NPZ浮点值 |
| 额外输入契约 | 真实多帧、内存影像、NPZ输入及真实 HU CT与低场采集按实际资源补测 |

本轮计划、执行记录和结果分别保存。表中列出需要覆盖的功能，不能把计划当作已经通过的实测。

### 真实域外输入与许可

- 64 mT T1 来自作者的 [Zenodo 15862148 v3](https://zenodo.org/records/15862148)，DOI `10.5281/zenodo.15862148`。从作者 ZIP 的 `64mT data/sub-0001/ses-01/anat/` 目录提取实际 T1 及 sidecar、README、MIT LICENSE；图像 shape 为 `112×136×40`，体素为 `1.6×1.6×5 mm`。
- HU CT 来自作者的 [SynthRAD2023 Task1](https://zenodo.org/records/7260705)，成员 `Task1/brain/1BA001/ct.nii.gz`，实际范围为 `−1023…1874 HU`。它是作者已裁剪、去面并刚体配准的 CT，按衍生影像报告。该资源采用 CC-BY-NC 4.0，图像留在私密验证目录，不随代码发布。
- 实际 EPI 来自 ds003138 的扩散采集。多帧子集、b0、MGZ 和 NPZ 格式的转换保存父输入哈希和派生方法；没有生成模拟体素。
- 大 ZIP 通过范围读取提取指定成员。已核对所取成员的 CRC、文件大小和 SHA-256；整包没有下载，不能声称整包 MD5 已验收。官方发布的整包大小和 MD5 仅作为来源元数据保存。

## 驱动与逐项参数

### 生成完整命令计划

```bash
# 统一服务器入口与尚不存在的私有计划文件。
fnit_server_root=/cwStorage/home/gongwk/Notebook_code/FNIT
private_job_plan=/path/to/fresh_strip_sr_jobs.private.json
python validation/smri_cpu/strip_sr_20261004/make_jobs.py \
  --server-root "$fnit_server_root" \
  --output "$private_job_plan"
```

`--server-root` 指定资源、冻结源码与运行目录的统一根目录；`--reference-wrapper` 可显式指定协调者核验过的原软件环境包装；`--source-path` 指定候选 `src`；`--source-revision` 记录该源码实际版本；`--run-directory` 可指定另一新冻结候选的输出根目录；`--output` 是不可覆盖的 JSON。默认计划为 16 个参数场景、40 个完整进程，不下载数据、不运行计算。真实 CT、64 mT 和格式输入另有补充计划。JSON中的 `jobs` 可交给统一 `queue_runner_v2.py`；原软件环境预检通过后再启动。

### 观察一次 Python API

```bash
# 输入为真实原始 T1，权重字节必须与固定清单一致。
anatomical_t1_image=/path/to/raw_T1w.nii.gz
synthstrip_checkpoint=/path/to/synthstrip.1.pt
candidate_source_revision=1d31e7baaebbb644ab199471f7fe6282721455fd
private_profile_directory=/path/to/fresh_strip_profile
anonymous_profile_report=/path/to/fresh_strip_profile.public.json
PYTHONPATH=src python validation/smri_cpu/strip_sr_20261004/profile_inference.py \
  --feature synthstrip --input "$anatomical_t1_image" \
  --weights "$synthstrip_checkpoint" \
  --weight-sha256 37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33 \
  --source-revision "$candidate_source_revision" \
  --output-dir "$private_profile_directory" --report "$anonymous_profile_report" \
  --device cpu --threads 8 --format-checks
```

| 参数 | 输入或行为 |
|---|---|
| `--feature` | `synthstrip` 或 `synthsr` |
| `--input` | 单个真实影像路径；影像只保存在私有目录 |
| `--weights`、`--weight-sha256` | 固定官方权重与预先核验的 SHA-256；不符即停止 |
| `--source-revision` | 实际运行源码的提交；报告另记录功能文件字节哈希 |
| `--output-dir`、`--report` | 新目录及新 JSON；不会覆盖旧尝试 |
| `--threads`、`--device` | 默认 8 与 CPU；CUDA剖析会同步指定设备并记录显存峰值 |
| `--no-csf`、`--border`、`--fill` | 原样传给 SynthStrip；默认 False、1、None |
| `--ct`、`--lowfield`、`--v1` | 原样传给 SynthSR；默认 False |
| `--disable-flipping`、`--disable-sharpening` | 关闭对应 SynthSR处理，默认 False |
| `--format-checks` | 复用本次结果检查 NIfTI/MGZ；SynthSR另写NPZ，无额外前向 |
| `--cpu-channels-last` | CPU 专用验证布局原型；禁止用于 CUDA；生产 Strip 是否自动启用另记录 |
| `--save-network-arrays` | 保存本次真实前向的输入/输出 NPY 到私有目录；不额外推理，写盘阶段另列 |
| `--cpu-bn-formula` | 仅 CPU SynthSR 诊断，默认 `pytorch`；`subtract_first` 与 `scale_bias` 改变验证实例的 BN 算式，不修改生产源码 |
| `--input-object` | 用 `nibabel.load` 后的内存影像调用公开 Python API；默认传路径 |

API完整输入输出和原软件命令见 [SynthStrip](../../../docs/synthstrip/README.md) 与 [SynthSR](../../../docs/synthsr/README.md)。

### 比较保存结果

```bash
python validation/smri_cpu/strip_sr_20261004/compare_outputs.py \
  --feature synthstrip --case case01_default \
  --reference-image /path/to/original/image.nii.gz \
  --candidate-image /path/to/fnit/image.nii.gz \
  --reference-mask /path/to/original/mask.nii.gz \
  --candidate-mask /path/to/fnit/mask.nii.gz \
  --reference-distance /path/to/original/distance.nii.gz \
  --candidate-distance /path/to/fnit/distance.nii.gz \
  --gate strip_official \
  --report /path/to/fresh_comparison.public.json
```

`--case` 是匿名功能标识；`--reference-*` 与 `--candidate-*` 分别是同输入生成的保存结果；`--report` 必须尚不存在。SynthSR只需要两幅 `image`。比较先检查 shape、affine（最大差不超过 `1e-6 mm`）与有限值，网格不匹配时不计算逐体素误差；dtype单独记录。掩膜报告 Dice、差异体素和体积，空掩膜不能被计为成功脑提取。脑图与距离场同时报告完整网格和掩膜并集的误差；SynthSR完整网格为主，并补充非零并集结果。NPZ按其真实浮点尺度比较，不先转换为uint8。数值门槛不会在看到结果后放宽。

| `--gate` | 验收条件 |
|---|---|
| `measure_only` | 只测量误差，不声明等价；默认值 |
| `strip_official`、`strip_candidate` | 掩膜、脑图逐值相同；SDT `rtol=1e-5, atol=1e-4 mm`；网格、dtype和有限值一致；掩膜非空 |
| `sr_candidate` | NIfTI/MGZ量化输出逐值相同；浮点NPZ `rtol=1e-5, atol=1e-4` |
| `sr_official` | 浮点NPZ `rtol=1e-5, atol=1e-3`；量化容差要求 exact≥99.99%、max≤1、MAE≤1e-4 |

`capture_fnit_strip_preprocess.py` 和 `capture_original_strip.py` 保存实际解码、1 mm 数组及归一化输入，用 `compare_preprocessing.py` 比较。原软件观察驱动以 `runpy` 运行已核验哈希的未修改脚本；只在验证目录依赖原软件。`capture_original_sr.py` 在同次官方推理中分别保存量化图和额外浮点 NPZ，并在原写盘函数会原位乘 2 之前取副本，不重复网络前向。

GPU 回归使用 gpucw1 的同一 H100 和统一 GPU 锁。首先对比修复前后输入；再在两臂均修复几何的条件下单独对比 CPU 后端策略修改，记录实际 TF32、cuDNN、FP32 类型、显存峰值与 `monitor_gpu.py` 采样的利用率。GPU observer 的进程时间与普通 CLI 分开报告。

`replay_sr_network_outputs.py` 接收 `--input`、`--reference-directory`、`--candidate-directory`、`--source-revision`、`--report`，比较两次实际 CNN 输入和原始预测，再将保存预测注入公开 `SynthSR` API。每次输入必须逐值同于捕获输入；不会构建网络、加载权重或执行 CNN，不能用于推理 benchmark。

`collect_cli_results.py` 接收 `--plan`、`--runs-root`、`--feature`、`--output-directory`，并可用 `--comparison-driver` 指定比较脚本。它逐项核对完成状态、nodecw10、8 线程和固定 8 核亲和性，再导出匿名 timing/精度；未完成项明确列为 pending，不执行推理。目录必须全新，不覆盖原记录。

## 来源

- [FreeSurfer SynthStrip](https://github.com/freesurfer/freesurfer/tree/v8.2.0/mri_synthstrip)，Hoopes et al., NeuroImage 2022，[DOI](https://doi.org/10.1016/j.neuroimage.2022.119474)。
- [FreeSurfer SynthSR](https://github.com/freesurfer/freesurfer/tree/v8.2.0/mri_synthsr)，Iglesias et al., Science Advances 2023，[DOI](https://doi.org/10.1126/sciadv.add3607)。
- 权重沿用 FNIT 固定 Release、原文件大小/SHA-256及原许可；本目录不再发布影像或模型文件。
