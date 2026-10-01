# TorchFLIRT：线性配准

[返回首页](../../README.md) · [源码](../../src/fnit/flirt/) · [GPU 批量优化报告](../../validation/flirt/gpu_batch.current.public.json) · [验证记录](../../validation/flirt/README.md)

`TorchFLIRT` 在 FNIT 内实现单被试线性配准和已知线性变换的重采样。运行时主要依赖 PyTorch、NumPy 和 nibabel，CUDA 批量 NMI 另需 Conda 环境已包含的 Triton；不调用 FSL。FSL 只用于离线对照；本次 GPU 优化重新核验的官方版本为 6.0.7.22，较早对照使用的版本保留在各自报告中。

当前公开接口支持以下模式：

- `dof=12, cost="corratio"`：12 自由度仿射配准，对应 FSL `flirt -dof 12 -cost corratio`；
- `dof=6, cost="normmi"`：6 自由度刚体配准，对应 FSL `flirt -dof 6 -cost normmi`。
- `applyxfm=True`：直接应用已知 `.mat`，或根据两张图的 qform/sform 对齐同一世界空间；不执行配准优化。

实现包含 FSL scaled-mm 坐标、8/4/2/1 mm 多层搜索、Brent 坐标优化、correlation ratio、normalized mutual information 和默认三线性输出路径。默认角度搜索遵循 FSL 的 `search 12`，包括 `dof=6` 的初始化；`dof` 决定后续优化和最终变换的自由度。

图像与 cost 使用 float32，矩阵构造使用 double，NMI 直方图使用 float64 累加。CUDA 默认允许 TF32；候选 cost 的坐标与融合采样使用明确的 float32 运算，不使用 TF32 GEMM，并关闭 FMA。输出采样及 `applyxfm` 的坐标也按 float32 系数逐项计算，避免 TF32 矩阵乘法改变采样位置。没有使用 float16 或 bfloat16。与官方 FSL 的误差及测量范围见本页真实数据对照。

## 输入

| Python 参数 | 命令行参数 | 类型 | 含义与要求 |
|---|---|---|---|
| `input` | `-in` | NIfTI 路径或单帧 `nibabel` 空间影像 | moving 图像。必须是有限值的 3D 图像；4D 图像仅允许末维长度为 1。 |
| `reference` | `-ref` | NIfTI 路径或单帧 `nibabel` 空间影像 | fixed 图像。它的 shape、affine 和 header 空间信息决定输出网格。 |
| `output` | `-out` | 可选路径 | 重采样图像的保存位置。`output` 与 `omat` 至少给出一个。 |
| `omat` | `-omat` | 可选路径 | input→reference 的 4×4 FSL scaled-mm 矩阵。 |
| `init` | `-init` | 可选 `.mat` 路径或 4×4 数组 | 配准模式下是优化初值；`applyxfm=True` 时是直接应用的 input→reference FSL scaled-mm 矩阵。 |
| `applyxfm` | `-applyxfm` | 布尔值 | 应用现成变换并在 reference 网格上重采样，不重新估计矩阵；默认 `False`。 |
| `usesqform` | `-usesqform` | 布尔值 | 仅用于 `applyxfm=True`：按两图 qform/sform 的共同世界坐标对齐；默认 `False`，且不能同时提供 `init`。两图均须有有效 qform 或 sform。 |
| `inweight` | `-inweight` | 可选 3D NIfTI | 仅用于 12/corratio：input 网格上的连续体素权重；shape 和 affine 必须与 input 相同。 |
| `refweight` | `-refweight` | 可选 3D NIfTI | 仅用于 12/corratio：reference 网格上的连续体素权重；shape 和 affine 必须与 reference 相同。 |
| `dof` | `-dof` | `6` 或 `12` | 变换自由度。当前只接受 `12/corratio` 和 `6/normmi` 两种组合。 |
| `cost` | `-cost` | `corratio` 或 `normmi` | 优化代价函数，必须与 `dof` 使用上述组合。 |
| `device` | `--device` | PyTorch 设备字符串 | 例如 `"cuda:0"` 或 `"cpu"`；省略时有 CUDA 则使用 CUDA。 |
| `execution` | `--execution` | `auto`、`reference` 或 `batched` | 默认 `auto`：CUDA 采用批量执行，CPU 保留串行参考路径；CUDA NMI 缺少 Triton 时自动使用 `reference`。 |
| `candidate_batch_size` | `--candidate-batch-size` | 正整数，默认 `128` | 每次 GPU chunk 的候选上限；不改变角度样本数量、搜索范围或迭代上限。 |
| `memory_budget_gb` | `--memory-budget-gb` | 正数，默认 `20.0` | 批量临时张量的显存预算，单位 GiB；同时根据设备剩余显存分块，图像每层只保存一份。 |
| `overwrite` | `--overwrite` | 布尔值 | 是否替换已有输出。默认保护已有文件。 |

## Python 单被试调用

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="subject_GM.nii.gz",  # moving 3D 图像，对应 FSL -in
    reference="template_GM.nii.gz",  # fixed 图像和输出网格，对应 FSL -ref
    output="subject_GM_to_template.nii.gz",  # reference 网格上的重采样图像
    omat="subject_GM_to_template.mat",  # input→reference 的 FSL scaled-mm 4×4 矩阵
    init=None,  # 可选初始矩阵；None 表示使用默认初始化和角度搜索
    inweight=None,  # 可选 input 网格连续权重图
    refweight=None,  # 可选 reference 网格连续权重图
    dof=12,  # 12 自由度仿射模型
    cost="corratio",  # FSL correlation-ratio 代价函数
    device="cuda:0",  # CUDA float32，默认允许 TF32；也可写 "cpu"
    execution="auto",  # CUDA 批量执行；reference 可重跑原串行路径
    candidate_batch_size=128,  # 候选分块上限，不减少搜索候选
    memory_budget_gb=20.0,  # 显存预算，GiB
    overwrite=False,  # False 时不覆盖已有文件
)
```

`run_flirt()` 返回 `FLIRTResult`。只需要内存结果时，也可直接调用模型：

```python
from fnit.flirt import TorchFLIRT

model = TorchFLIRT(
    device="cuda:0",  # 计算设备
    angular_search=True,  # 执行默认角度搜索
    dof=12,  # 12 自由度仿射模型
    cost="corratio",  # correlation-ratio 代价函数
    execution="auto",  # CUDA 使用批量候选求值，CPU 使用参考路径
    candidate_batch_size=128,  # 最大候选分块
    memory_budget_gb=20.0,  # 显存预算，GiB
)
result = model(
    moving="subject_GM.nii.gz",  # moving 图像路径或 nibabel 空间影像
    fixed="template_GM.nii.gz",  # fixed 图像路径或 nibabel 空间影像
    init=None,  # 可选 FSL scaled-mm 初始矩阵
    inweight=None,  # 可选 moving 权重图
    refweight=None,  # 可选 fixed 权重图
)
```

## 返回值与磁盘输出

`FLIRTResult` 包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `moved` | `FNITNifti1Image`，是 `nibabel.Nifti1Image` 的子类 | input 在 reference 网格上的图像；按 FSL 规则保留 input 的存盘类型，整数输出强度范围小于 1.5 时转 float32，以保留插值后的 mask 小数。 |
| `matrix` / `fsl_matrix` | NumPy `(4, 4)` 数组 | input→reference 的 FSL scaled-mm 矩阵。 |
| `moving_to_fixed_world` | NumPy `(4, 4)` 数组 | input world-RAS→reference world-RAS 的正向矩阵。 |
| `fixed_to_moving_world` | NumPy `(4, 4)` 数组 | 重采样使用的 reference world-RAS→input world-RAS pull 矩阵。 |
| `qc` | 字典 | 设备、执行路径、TF32 状态、代价函数、搜索层级、评价次数、分阶段墙钟时间和坐标方向。运行时 QC 不把仓库基准解释为当前输入已与 FSL 比较。 |

写盘后的结构为：

```text
subject_GM_to_template.nii.gz  # -out；reference 网格；数据类型按上表的 FSL 规则
subject_GM_to_template.mat     # -omat；4 行×4 列文本矩阵，input→reference
```

配准模式的 qform 和 sform 各自保留 reference 的有效值，缺少一种时从另一种补齐；两种都未设置时使用变换后的 input 空间信息。`applyxfm` 保留 reference 的两种 form 和 code，包括未设置的 code。两种模式都保留 reference header 的 `pixdim`。

FSL scaled-mm 使用 NIfTI header 的 `pixdim` 作为体素大小。视野外填充值采用输出平滑后 input 边缘两层体素的第 10 百分位；存盘类型与整数 mask 的处理规则见上表。

输出先写入同目录临时文件，全部成功后再原子移动到目标位置。没有 `--overwrite` 时，已有文件会使调用停止。

## 命令行调用

独立入口：

```bash
fnit-flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio \
  --device cuda:0
```

统一入口将 `fnit-flirt` 换成 `fnit flirt`，配准参数相同；统一入口还可用 `--threads` 设置 PyTorch CPU 线程数，默认 `1`。`--device cuda:0` 选择第一块可见 GPU，CPU 写为 `--device cpu`。

对应的 FSL 命令是：

```bash
flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio
```

两套命令的 `-in`、`-ref`、`-out`、`-omat`、`-init`、`-inweight`、`-refweight`、`-dof` 和 `-cost` 含义一致。FNIT 另外提供 `--device`、`--execution`、`--candidate-batch-size`、`--memory-budget-gb` 和 `--overwrite`。当前接口不接受 FSL 的其他 cost、DOF、schedule、搜索范围和插值选项。

去脑 b0 到去脑 T1 的刚性配准改用 6 自由度与归一化互信息：

```bash
fnit-flirt -in b0_brain.nii.gz -ref T1_brain.nii.gz \
  -out b0_in_T1.nii.gz -omat b0_to_T1.mat \
  -dof 6 -cost normmi --device cuda:0
```

这里 `-in` 是去脑 b0，`-ref` 是去脑 T1；`-out` 写 T1 网格上的 b0，`-omat` 写 b0→T1 矩阵。对应的原软件指令是 `flirt -in b0_brain.nii.gz -ref T1_brain.nii.gz -out b0_in_T1.nii.gz -omat b0_to_T1.mat -dof 6 -cost normmi`。真实数据对照见下文。

## 已知线性变换：MNI152 分辨率转换

同一 MNI152 世界空间的 1 mm 与 2 mm 模板转换，使用 `applyxfm=True, usesqform=True`。`reference` 决定输出的体素大小、形状、视野和 affine；这一步不寻找新的配准。输入目前须为有限值的单帧 3D NIfTI，使用 FSL FLIRT 默认的三线性插值及下采样前平滑。标签图所需的最近邻路径尚未实现。

```python
from fnit.flirt import run_flirt

mni_1mm = "/absolute/path/MNI152_T1_1mm.nii.gz"  # 输入：原始 MNI152 T1 1 mm 强度图
mni_2mm = "/absolute/path/MNI152_T1_2mm.nii.gz"  # reference：MNI152 2 mm 输出网格
resampled_t1 = "/absolute/path/MNI152_T1_1mm_on_2mm.nii.gz"  # 输出：2 mm 网格上的输入强度
world_alignment = "/absolute/path/MNI152_1mm_to_2mm.mat"  # 输出：input→reference 的 FSL scaled-mm 矩阵
result = run_flirt(
    input=mni_1mm,  # moving 3D NIfTI
    reference=mni_2mm,  # 决定输出网格，不从此图复制强度
    output=resampled_t1,  # 保存三线性重采样图
    omat=world_alignment,  # 保存实际使用的 4×4 FSL 矩阵
    applyxfm=True,  # 只应用变换，不执行配准搜索
    usesqform=True,  # 使两图的 NIfTI 世界坐标重合
    device="cuda:0",  # PyTorch 设备；CPU 可写 "cpu"
    overwrite=False,  # 保护已存在的结果
)
```

同一计算也可直接取得内存结果：`TorchFLIRT(device="cuda:0").applyxfm(mni_1mm, mni_2mm, usesqform=True)`。如果已有由 FLIRT 产生的 input→reference `.mat`，把上面 `usesqform=True` 改为 `init="/absolute/path/input_to_reference.mat"`，保留 `applyxfm=True`；这时直接应用该矩阵。`init` 和 `usesqform` 必须二选一。`-inweight`、`-refweight`、非默认 `-dof/-cost` 只用于配准，不用于 `applyxfm`。

真实 T1 FAST 概率图到 EPI 网格的同输入、同矩阵控制，验证了默认降采样预滤波与上述坐标精度；GPU 保持 TF32 默认开启。精度及 0.8 阈值后的组织掩膜一致性见[fMRI 相同步骤对照](../../validation/fmri/matched_native.md)。

FNIT 命令行与对应原软件命令：

```bash
fnit-flirt -in /absolute/path/MNI152_T1_1mm.nii.gz \
  -ref /absolute/path/MNI152_T1_2mm.nii.gz \
  -applyxfm -usesqform -out /absolute/path/MNI152_T1_1mm_on_2mm.nii.gz \
  -omat /absolute/path/MNI152_1mm_to_2mm.mat --device cuda:0

flirt -in /absolute/path/MNI152_T1_1mm.nii.gz \
  -ref /absolute/path/MNI152_T1_2mm.nii.gz \
  -applyxfm -usesqform -out /absolute/path/fsl_MNI152_T1_1mm_on_2mm.nii.gz
```

对于已有 `.mat`，两者均改用 `-applyxfm -init /absolute/path/input_to_reference.mat`。FSL [FLIRT User Guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/user_guide.html) 与 [FAQ 的分辨率转换示例](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/faq.html) 说明了这两种调用。输出 `.mat` 为 FSL scaled-mm 坐标；虽然 `usesqform` 对应世界空间恒等映射，它的 `.mat` 通常不等于单位矩阵。

本仓库提供官方 FLIRT 直接从上述两张 FSL 原始模板导出的 [1→2 mm 矩阵](../../src/fnit/flirt/assets/FSL_MNI152_T1_1mm_to_2mm.mat)（仅去掉每行末尾空格；44 bytes，SHA-256 `402bef1e8114fd895cd8261fd73fc9e565609f02d90d053de7c52c82535d4d2a`）。对应的输入模板 SHA-256 为 `d1f03e160c2548592a01d98d44d7e0ffa8a8ef58bc9b07d26369e214ba170edb`，reference 模板为 `0585cd056bf5ccfb8bf97a5f6a66082d4e7caad525718fc11e40d80a827fcb92`。矩阵恰好是单位矩阵，因为这对模板的 FSL scaled-mm 坐标一致；其他来源的 MNI152 图像必须检查 affine 和视野，不能直接沿用。模板影像本身不随 FNIT 分发。

下载矩阵文件后，也可以运行：

```bash
fnit-flirt -in /absolute/path/MNI152_T1_1mm.nii.gz \
  -ref /absolute/path/MNI152_T1_2mm.nii.gz -applyxfm \
  -init /absolute/path/FSL_MNI152_T1_1mm_to_2mm.mat \
  -out /absolute/path/MNI152_T1_1mm_on_2mm.nii.gz
```

## `.mat` 坐标约定

FSL `.mat` 不是 NIfTI world-RAS affine。设 input 和 reference 的 voxel-to-world 矩阵为 `W_in`、`W_ref`，对应的 FSL scaled-mm 基为 `S_in`、`S_ref`，FLIRT 矩阵为 `A`，则 world-RAS 正向变换为：

```text
W_ref @ inverse(S_ref) @ A @ S_in @ inverse(W_in)
```

FSL 在 voxel-to-world 线性部分行列式为正时翻转 scaled-mm 第一轴。FNIT 的 `.mat` 读写和 world-RAS 转换使用同一规则。不能把该矩阵直接当作 FreeSurfer LTA 或 NIfTI affine。

## GPU 批量执行

`execution="batched"` 按块计算独立候选。8 mm 的粗角度候选、1,331 个细角度候选及各候选的独立 Brent 搜索共享 GPU 求值；各搜索按原顺序接收自己的 cost。矩阵在 CPU 批量使用 double 构造，Brent 状态与分支保留在 CPU。GPU 通过 Triton 将候选坐标、三线性采样和边缘权重融合为一个 kernel，再计算 cost。4 mm 扰动优化也按此方式执行；2/1 mm 的单条搜索保留顺序依赖。

两条执行路径共享修正后的实现：体素大小改为 header `pixdim`，避免 affine 列范数的微小误差使 1 mm 层被跳过；修正 cost 分箱、邻 bin、零联合熵、CorrRatio 舍入和 schedule 的 float 运算。官方默认 `search 12` 在 8 mm 角度初始化时联合优化共同尺度和三轴平移；6/normmi 随后的候选姿态与最终优化使用 6 自由度。

批量及融合执行保留完整层级、角度范围、候选顺序、剪枝、自由度进阶和停止条件。真实数据 gate 对比修正后的未融合实现，检查保存矩阵、重采样影像、header、cost、求值次数和搜索候选；融合只改变执行方式。CPU 默认使用串行张量路径，有权重的候选采样保留张量实现。

reference grid、分箱排序、图像、权重和 sampling 常量在每层缓存。需要 Brent 分支时，每轮只回传一个 cost 向量。CorrRatio 归约保持候选的 16-byte 起点对齐和参考 CUDA 求和分组；NMI 的概率与熵使用 float32。Triton 采样保留逐步 float32 舍入，融合归约关闭 FMA。

主页 Conda 环境已包含 PyTorch 2.5.1 和 Triton 3.1.0，无需安装 FSL 或新增编译依赖。缺少 Triton 时，CorrRatio 使用张量采样和归约，CUDA NMI 的 `auto` 回到串行 `reference`；显式 CUDA `batched/normmi` 需要 Triton。

运行结果的 `qc["phase_timings_seconds"]` 给出分阶段墙钟时间，`qc["cost_evaluations"]` 给出求值次数。CUDA 异步准备工作可能在下一个阶段完成，搜索阶段包含等待 cost 的时间。`batched_host_result_transfers` 只计批量求值的回传，完整 kernel、copy 和同步计数由独立 profiler 获得。

### 重跑性能与精度对照

[测量脚本](../../tools/benchmark_flirt_gpu.py)读取已经生成的官方 `.mat` 和重采样影像，在 FNIT 运行过程中不执行 FSL。下面两次调用使用相同输入和 oracle，分别测量串行参考与批量路径；12 自由度改用 `--dof 12 --cost corratio`。

```bash
python tools/benchmark_flirt_gpu.py \
  --moving /absolute/path/b0_brain.nii.gz \
  --reference /absolute/path/T1_brain.nii.gz \
  --fsl-matrix /absolute/path/fsl_b0_to_T1.mat \
  --fsl-moved /absolute/path/fsl_b0_in_T1.nii.gz \
  --dof 6 --cost normmi --device cuda:0 --execution reference \
  --output-dir /absolute/path/benchmark/reference --warm-repeats 1 --profile-full

python tools/benchmark_flirt_gpu.py \
  --moving /absolute/path/b0_brain.nii.gz \
  --reference /absolute/path/T1_brain.nii.gz \
  --fsl-matrix /absolute/path/fsl_b0_to_T1.mat \
  --fsl-moved /absolute/path/fsl_b0_in_T1.nii.gz \
  --dof 6 --cost normmi --device cuda:0 --execution batched \
  --output-dir /absolute/path/benchmark/batched --warm-repeats 1 --profile-full

python tools/check_flirt_gpu_parity.py \
  --baseline-dir /absolute/path/benchmark/reference \
  --optimized-dir /absolute/path/benchmark/batched \
  --output-json /absolute/path/benchmark/parity.json
```

`cold` 是新进程内的第一次完整配准，CUDA context 与读图在计时前完成；`warm` 在同一进程重新执行全部搜索，不复用拟合矩阵。时间包含优化、重采样和结果回传，不含读写文件。profile 是第三次独立运行，其耗时不用于速度表。报告记录源码和输入 SHA-256、阶段时间、显存、实际 CUDA runtime 调用计数及同输入 FSL 精度；gate 检查两条路径的保存矩阵、影像与 header、cost、评价次数和已捕获的角度候选。矩阵文本保存 12 位有效数字，文件相同不等于未舍入的 double 矩阵逐 bit 相同。

## 真实数据测量

### 本次定位与修复

- **体素大小与搜索层级。** FSL 从 NIfTI header 的 `pixdim` 读取采样距离。旧代码使用 affine 列范数；公开 T1 的 `1.000000021 mm` 取整成 2 mm，使最后的 1 mm 搜索仍使用 2 mm 参考网格。只修正此处的真实数据实验已将 world-grid RMS 从 0.37208 降至 0.01353 mm，确认它是该病例精度差异的主要原因。
- **代价函数。** 修正 NMI 最大强度的邻格顺序、零联合熵代价及分箱的 `value*b1+b0` 舍入；CorrRatio 恢复官方两次 `1−x` 的 float32 舍入和第二矩乘法顺序。
- **搜索。** 恢复官方 schedule 中零扰动也会执行的矩阵分解／重建，以及旋转扰动、角度阈值、RMS 剪枝和 Brent 停止条件的 float 舍入。8 mm 候选保留自己的调用顺序。
- **重采样。** 矩阵先在 double 中求逆再赋值为 float 采样系数；补齐缺失 qform，保留两种有效 form，按 FSL 处理背景和整数 mask 输出。最终采样移除动态 CUDA 布尔索引引起的同步。

修复后执行真正的 1 mm 层，因此不能将旧版跳过该层的耗时视作相同工作量。以下速度和精度表均记录完整实际运行；CPU 串行与 CUDA 批量共享修复后的科学定义。

### 修复前后速度与精度

本轮重新测量 `cb1748d` 旧批量版、修正科学行为的未融合版及最终融合版。修正后的未融合版与最终版在三份真实输入上，保存矩阵、体素、header、cost 和求值次数均通过逐 bit 检查；12-DOF 的角度候选记录也一致。[当前标量报告](../../validation/flirt/gpu_batch.current.public.json)保留每次实际测量的源码和工具 SHA-256。

| 真实输入与 GPU | 旧批量 cold / warm | 修正、未融合 cold / warm | 修正、融合 cold / warm | 融合峰值 allocated / reserved |
|---|---:|---:|---:|---:|
| b0→T1，6/normmi，H100 | 5.31 / 4.00 秒 | 5.10 / 4.16 秒 | 4.30 / 3.04 秒 | 2.63 / 3.08 GiB |
| T1→MNI152，12/corratio，H100 | 5.56 / 4.39 秒 | 5.48 / 4.37 秒 | 4.43 / 3.29 秒 | 1.01 / 1.45 GiB |
| 公开 T1w→T1w，12/corratio，RTX 3060 | 17.18 / 15.31 秒 | 30.77 / 27.50 秒 | 10.73 / 9.35 秒 | 1.87 / 2.17 GiB |

公开病例恢复真实 1 mm 网格后，融合使相同工作量的 warm 从 27.50 降至 9.35 秒，约 2.94×。旧版使用了错误的 2 mm 参考网格。H100 全卡利用率约 100%，包含其他作业；以上记录共享节点的耗时。RTX 3060 本轮 cold/warm 全卡平均利用率为 39.6% / 47.9%，包含桌面进程。显存是本进程 PyTorch 峰值。

| FSL 对照 | 位移 RMS：旧→修复，mm | 重采样 Pearson：旧→修复 | MAE：旧→修复 | 修复后 support Dice |
|---|---:|---:|---:|---:|
| b0→T1，CPU 环境 A | 0.00955→0.00216 | 0.9999718→0.9999814 | 17.4431→13.8556 | 0.999579 |
| b0→T1，CPU 环境 B | 0.18724→0.18629 | 0.9994895→0.9996545 | 66.7281→57.5962 | 0.998071 |
| T1→MNI152，CPU 环境 B | 0.12269→0.12624 | 0.9968224→0.9999838 | 10.5053→0.9761 | 1.000000 |
| 公开 T1w→T1w | 0.37208→0.01355 | 0.9952013→0.9999965 | 20.5673→0.5555 | 0.999896 |

b0 和 T1→MNI 的官方程序为 FSL 6.0.7.22，本轮重新运行完整命令。b0 在 CPU 环境 A 重现旧 oracle，矩阵和体素逐 bit 相同；环境 B 的官方结果与它相差 0.18773 mm RMS，因此两项对照都列出。已核验的 FLIRT 入口和 23 个 FSL 动态库 SHA 相同，系统数学库不同；造成结果差异的具体环节尚未隔离。官方完整命令用时分别为 A 的 b0 13.96 秒、B 的 b0 14.75 秒和 T1→MNI 16.58 秒，包含进程启动和文件读写。公开例使用已核验 SHA 的 FSL 6.0.7.4 oracle。

矩阵误差使用 moving 视野的 13³ 个世界坐标点，两边 scaled-mm 均按 header `pixdim` 转换。图像 Pearson/MAE 在固定的官方输出非零区域计算；MAE 为原始强度单位。mean、median、p95 及全部指标见[验证页](../../validation/flirt/README.md#当前精度)。三例最终 FNIT cost / 求值次数依次为 `−1.188756347 / 6502`、`0.093537807 / 7468`、`0.212815523 / 8969`；官方默认日志没有同格式的最终内部 cost。

T1→MNI 的图像与输出空间信息明显改善，矩阵 RMS 增加 0.00355 mm，约 2.9%。真实输入消融表明 header 采样距离与 cost 修正会改变搜索轨迹；回退本次 Brent 舍入后矩阵不变。继续使用符合官方源码的规则，并在报告中保留这项小幅退化。修复后各例的 shape、affine、dtype、qform/sform code 检查通过；完整 NIfTI header 与 FSL 的逐 bit 等价没有建立。

### profile 与保留的参考路径

公开病例的完整 profile 为 49,557 次 kernel launch；上次批量实现的实测记录为 114,928 次，原测量源码 hash 单独保留。stream synchronize 为 850 次，上次为 841 次；本轮主要减少 kernel 数，Brent 仍按顺序回传 cost。最新 T1→MNI H100 profile 为 45,602 个 kernel、764 次 stream synchronize、747 次 H2D 与 725 次 D2H copy event。阶段耗时、计数范围及剩余同步见[验证页](../../validation/flirt/README.md#阶段与-profile)。profile 独立运行，耗时不列入速度表。

`execution="reference"` 保留修正后的串行 cost/Brent；`execution="batched"` 执行相同搜索的批量、融合计算。没有新增减少层级或迭代的 `fast` 路径。融合与未融合的执行一致性已经验证；官方 FSL 的逐行坐标递推与 FNIT 逐点坐标公式仍有 float32 舍入差异，因此当前实现仍不声明与 FSL 逐矩阵或逐体素等价。

### 其他已验证数据与重采样路径

**MNI152 模板 1 mm↔2 mm 的 `applyxfm -usesqform`。** 以 FSL 发布的原始 MNI152_T1 1 mm、2 mm 图像为双向输入及 reference，FNIT 与官方 FLIRT 各自从新进程读图、重采样并写出 gzip NIfTI。两边输出的 shape 和 affine 均等于 reference。源码哈希、模板 SHA-256、逐方向矩阵和完整指标见 [CPU 实测报告](../../validation/flirt/applyxfm_mni.cpu.json)、[H100 实测报告](../../validation/flirt/applyxfm_mni.gpu.json)；[重测脚本](../../validation/flirt/benchmark_applyxfm.py)不在 FNIT 运行时调用。

| 方向 | 输出尺寸 | 全体素 Pearson r | 全体素 MAE；最大绝对差 | FSL CPU / FNIT CPU 完整命令耗时 |
|---|---:|---:|---:|---:|
| 1→2 mm | 91×109×91 | 0.99999999997 | 0.000450；1 | 1.22 / 3.06 秒 |
| 2→1 mm | 182×218×182 | 1.0 | 0；0 | 1.36 / 3.58 秒 |

官方 FSL 安装在这台服务器上对两次命令都返回状态码 255；脚本每次先删旧文件，只在新输出存在、gzip 可读且尺寸与 affine 合法时才计算指标。FSL `-version` 也返回 255，因此这里把退出状态保留在报告中，而不将其解释为图像计算失败。时间是当次共享节点的观察值，包含进程启动和文件读写。

H100 的两方向精度与上表 CPU 完全相同，FNIT 完整命令分别用时 5.34 和 4.61 秒。测试时两块 H100 的已用显存约 69 GiB、利用率 100%；这些耗时不用于推断 GPU 加速比。

**串行参考路径的 CPU/FSL 对照。** 以下保留既有真实数据结果及原测量源码 hash，当前批量源码未重新执行全部 GM 病例。GM 图像比较使用 `template_GM > 0` 的 258,990 个体素；b0 比较使用两份非零输出的交集。两者的 mask 与本页最新 GPU 表不同。

| 真实数据与串行路径 | 对 FSL 矩阵位移 RMS | moved Pearson | moved MAE | FNIT / FSL CPU 时间 |
|---|---:|---:|---:|---:|
| b0→T1，CPU，一例 | 0.009469 mm | 0.99999935 | 2.770 | 49.57 / 10.02 秒 |
| GM→group GM，CPU，10 例 | 中位数 0.019811；最大 0.078530 mm | 中位数 0.999985 | 中位数 0.001056 | 中位数 170.21 / 27.70 秒 |
| GM→group GM，H100，前 4 例 | 中位数 0.007118；最大 0.021722 mm | 中位数 0.999806 | 中位数 0.004244 | FNIT 406.86 秒；FSL 前 4 例 24.26 秒 |

b0 的 FNIT 时间不含文件读写，FSL 是完整命令；GM 的时间均含进程启动和读写。测量来自共享节点的不同运行，不计算加速比。GM 的 `rmsdiff <= 0.05 mm` 门限，CPU 为 9/10，原串行 H100 为 4/4。逐例结果见 [GM CPU](../../validation/flirt/report.cpu.current.json)、[GM H100](../../validation/flirt/report.gpu.current.json)和 [b0 CPU/FSL](../../validation/connectome/original_ukb_flirt.public.json)。[源码范围核对](../../validation/runtime_dependencies/flirt_profile_source_equivalence.public.json)仅证明早期 12-DOF 分支继承，不是当前批量版的十例重跑。

## 公开 OpenNeuro 示例

图中使用仓库内 OpenNeuro ds000114 v1.0.2 的 CC0 去面部派生数据，sub-02 T1w 为 input，sub-01 T1w 为 reference。FNIT 是最新批量 GPU 实测输出，参考为 FSL FLIRT 6.0.7.4；两者使用相同强度范围，绝对差单独缩放。

![OpenNeuro T1w：FSL FLIRT 与 FNIT 批量 GPU 配准结果](figures/flirt_public_current.png)

官方输出非零区域内 Pearson 为 0.9999965、MAE 为 0.5555；13³ world-grid 位移 RMS 为 0.01355 mm。cold/warm 为 10.73 / 9.35 秒，峰值 allocated / reserved 为 1.87 / 2.17 GiB。输入、oracle、测量源码和图示的 SHA-256 见[最新报告](../../validation/flirt/gpu_batch.current.public.json)。该 T1w→T1w 病例与 GM 数据集使用不同输入及误差定义。

## 结果解释

当前实现已经对齐 FSL 的输入/输出文件结构、输出网格、矩阵方向和 scaled-mm 坐标合同。数值优化仍会在个别病例落到与 FSL 不同的局部解。需要与既有 FSL 结果逐矩阵复现的研究，应先按自己的图像类型建立同输入验证集，再决定是否采用当前误差范围。

该移植依据 FSL 源码，受非商业 [FSL Software Licence](../../licenses/FSL-6.0.txt) 约束。第三方说明见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。

## 参考文献与原实现

- 参考文献：Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage (2002), [doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- 原实现代码库：[FSL `flirt`](https://git.fmrib.ox.ac.uk/fsl/flirt)。
