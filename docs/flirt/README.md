# TorchFLIRT：线性配准

[返回首页](../../README.md) · [源码](../../src/fnit/flirt/) · [当前 10 例报告](../../validation/flirt/report.public.json) · [公开示例报告](../../validation/flirt/public_example.current.json)

`TorchFLIRT` 在 FNIT 内实现单被试线性配准。候选程序只依赖 PyTorch、NumPy 和 nibabel，运行时不调用 FSL。FSL 6.0.7.4 只用于本页的对照测试。

当前公开接口支持两组参数：

- `dof=12, cost="corratio"`：12 自由度仿射配准，对应 FSL `flirt -dof 12 -cost corratio`；
- `dof=6, cost="normmi"`：6 自由度刚体配准，对应 FSL `flirt -dof 6 -cost normmi`。

实现包含 FSL scaled-mm 坐标、8/4/2/1 mm 多层搜索、Brent 坐标优化、correlation ratio、normalized mutual information 和默认三线性输出路径。CUDA 使用 float32，默认启用 TF32；没有使用 float16 或 bfloat16。

## 输入

| Python 参数 | 命令行参数 | 类型 | 含义与要求 |
|---|---|---|---|
| `input` | `-in` | NIfTI 路径或单帧 `nibabel` 空间影像 | moving 图像。必须是有限值的 3D 图像；4D 图像仅允许末维长度为 1。 |
| `reference` | `-ref` | NIfTI 路径或单帧 `nibabel` 空间影像 | fixed 图像。它的 shape、affine 和 header 空间信息决定输出网格。 |
| `output` | `-out` | 可选路径 | 重采样图像的保存位置。`output` 与 `omat` 至少给出一个。 |
| `omat` | `-omat` | 可选路径 | input→reference 的 4×4 FSL scaled-mm 矩阵。 |
| `init` | `-init` | 可选 `.mat` 路径或 4×4 数组 | input→reference 的初始 FSL scaled-mm 矩阵；随后仍执行优化。 |
| `inweight` | `-inweight` | 可选 3D NIfTI | input 网格上的连续体素权重；shape 和 affine 必须与 input 相同。 |
| `refweight` | `-refweight` | 可选 3D NIfTI | reference 网格上的连续体素权重；shape 和 affine 必须与 reference 相同。 |
| `dof` | `-dof` | `6` 或 `12` | 变换自由度。当前只接受 `12/corratio` 和 `6/normmi` 两种组合。 |
| `cost` | `-cost` | `corratio` 或 `normmi` | 优化代价函数，必须与 `dof` 使用上述组合。 |
| `device` | `--device` | PyTorch 设备字符串 | 例如 `"cuda:0"` 或 `"cpu"`；省略时有 CUDA 则使用 CUDA。 |
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
| `moved` | `FNITNifti1Image`，是 `nibabel.Nifti1Image` 的子类 | input 在 reference 网格上的 float32 图像。 |
| `matrix` / `fsl_matrix` | NumPy `(4, 4)` 数组 | input→reference 的 FSL scaled-mm 矩阵。 |
| `moving_to_fixed_world` | NumPy `(4, 4)` 数组 | input world-RAS→reference world-RAS 的正向矩阵。 |
| `fixed_to_moving_world` | NumPy `(4, 4)` 数组 | 重采样使用的 reference world-RAS→input world-RAS pull 矩阵。 |
| `qc` | 字典 | 设备、TF32 状态、代价函数、搜索层级、评价次数和坐标方向。运行时 QC 不把仓库基准解释为当前输入已与 FSL 比较。 |

写盘后的结构为：

```text
subject_GM_to_template.nii.gz  # -out；reference 的 shape 和 affine，float32
subject_GM_to_template.mat     # -omat；4 行×4 列文本矩阵，input→reference
```

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

统一入口写法相同：

```bash
fnit flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio \
  --device cuda:0
```

这两条命令把 `subject_GM.nii.gz` 仿射配准到 `template_GM.nii.gz`，把重采样图像写入 `-out`，并把 input→reference 的 FSL scaled-mm 矩阵写入 `-omat`。`--device cuda:0` 选择第一块可见 GPU；需要 CPU 时改为 `--device cpu`。

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

两套命令的 `-in`、`-ref`、`-out`、`-omat`、`-init`、`-inweight`、`-refweight`、`-dof` 和 `-cost` 含义一致。FNIT 另外提供 `--device` 和 `--overwrite`。当前接口不接受 FSL 的其他 cost、DOF、schedule、搜索范围和插值选项。

## `.mat` 坐标约定

FSL `.mat` 不是 NIfTI world-RAS affine。设 input 和 reference 的 voxel-to-world 矩阵为 `W_in`、`W_ref`，对应的 FSL scaled-mm 基为 `S_in`、`S_ref`，FLIRT 矩阵为 `A`，则 world-RAS 正向变换为：

```text
W_ref @ inverse(S_ref) @ A @ S_in @ inverse(W_in)
```

FSL 在 voxel-to-world 线性部分行列式为正时翻转 scaled-mm 第一轴。FNIT 的 `.mat` 读写和 world-RAS 转换使用同一规则。不能把该矩阵直接当作 FreeSurfer LTA 或 NIfTI affine。

## 真实数据测量与当前 12-DOF 路径

测试使用 10 例真实 T1w 经 FSL FAST 得到的 GM PVE，以及同一 UKB group-GM template。FNIT CPU、FNIT H100 GPU 和 FSL 使用相同 input、reference 与 12-DOF/corratio 配置。FSL 官方矩阵用于重新运行 `flirt -applyxfm`，因此影像指标比较的是直接 FLIRT 重采样结果，不包含 VBM 后续 mask 或调制步骤。指标在 `template_GM > 0` 的 258,990 个体素内计算。

候选源码 SHA-256 写入[合并报告](../../validation/flirt/report.public.json)。CPU 和 GPU 的完整逐例记录分别见 [CPU 报告](../../validation/flirt/report.cpu.current.json) 和 [GPU 报告](../../validation/flirt/report.gpu.current.json)。这组数据由 `flirt/core.py` `f5315f…` 生成；当前文件为 `ce375d…`。差异位于 6-DOF/normmi 的搜索代价函数和运行时 QC，12-DOF/corratio 使用的 `_DefaultFLIRTEngine` 及 QC 构造前的调用路径 AST 均未变化。逐项 hash 和 AST 指纹见[配置限定的源码等价证明](../../validation/runtime_dependencies/flirt_profile_source_equivalence.public.json)。因此本节数值可继承到当前 **12-DOF/corratio** 路径，但它不是当前 hash 的 fresh 完整真实数据重跑，也不能证明 6-DOF/normmi 数值等价。

| 指标 | FNIT CPU | FNIT H100 GPU，TF32 |
|---|---:|---:|
| 矩阵 RMS 差，中位数 [Q1–Q3] | 0.018468 [0.008390–0.027364] mm | 0.007994 [0.006268–0.025214] mm |
| 矩阵 RMS 差，最大值 | 0.078530 mm | 0.063188 mm |
| `rmsdiff <= 0.05 mm` | 9/10 | 9/10 |
| moved Pearson，中位数；最小值 | 0.999985；0.999835 | 0.999823；0.999597 |
| moved Dice@0.2，中位数；最小值 | 0.999187；0.997356 | 0.997048；0.995527 |
| moved MAE，中位数 | 0.001013 | 0.003812 |
| moved RMSE，中位数 | 0.001735 | 0.005935 |

CPU 和 GPU 都有 1 例超过预设的 0.05 mm 矩阵门限，因此当前实现没有通过 10/10 的矩阵判据，也不声明逐元素或完整数值等价。直接重采样影像仍保持很高的一致性。CPU 的差异小于默认 TF32 GPU，说明 TF32 和不同设备上的归约顺序会影响串行搜索落点。

完整命令时间包括 Python/FSL 进程启动、图像读取、优化、重采样以及矩阵和影像写盘：

| 实现 | 硬件 | 中位数 [Q1–Q3] | 相对 FSL |
|---|---|---:|---:|
| FSL FLIRT 6.0.7.4 | Xeon Gold 6430 CPU | 27.705 [24.867–30.215] s | 1.00 |
| FNIT TorchFLIRT | Xeon Gold 6418H CPU | 82.159 [76.455–88.879] s | 2.97× |
| FNIT TorchFLIRT | H100 PCIe GPU，TF32 | 30.165 [27.574–33.002] s | 1.09× |

H100 相对 FNIT CPU 的中位时间为 0.367，即约快 2.72 倍；它在这组小规模搜索上仍比 FSL C++ CPU 慢约 9%。GPU 峰值 allocated memory 为 487,223,808 bytes（0.454 GiB），峰值 reserved memory 为 870,318,080 bytes（0.811 GiB）。FSL `-applyxfm` 的单独复核中位数为 2.56 s；该数值只包含重采样，不能与完整配准时间直接比较。

所有计时来自共享节点。GPU 计时开始时同卡没有观测到其他计算负载，但有其他进程保留显存；CPU 计时使用另一台同代 Xeon 节点。因此这些时间用于说明当前实现的实际量级，不代表独占硬件吞吐上限。

## 公开 OpenNeuro 示例

下图使用仓库内 OpenNeuro ds000114 v1.0.2 的 CC0 去面部派生数据：sub-02 T1w 为 input，sub-01 T1w 为 reference。两者分别运行 FSL FLIRT 6.0.7.4 和 FNIT CPU 的 12-DOF/corratio 测量源码。该配置可按上述证明继承到当前源码。图中 FSL 与 FNIT 使用相同显示范围，差值图单独缩放。

![OpenNeuro T1w 上 FSL FLIRT 与 FNIT TorchFLIRT 的配准结果](figures/flirt_public_current.png)

该公开 T1w→T1w 示例的 moved Pearson 为 0.995955，normalized RMSE 为 0.018652，矩阵 RMS 差为 0.196141 mm；FNIT CPU 与 FSL CPU 用时分别为 126.06 s 和 40.41 s。它用于复现图示和检查一般 T1w 输入，不属于上面的 GM 门限数据集。输入来源、文件 hash、命令、源码 hash 和完整指标见[公开示例报告](../../validation/flirt/public_example.current.json)。

## 结果解释

当前实现已经对齐 FSL 的输入/输出文件结构、输出网格、矩阵方向和 scaled-mm 坐标合同。数值优化仍会在个别病例落到与 FSL 不同的局部解。需要与既有 FSL 结果逐矩阵复现的研究，应先按自己的图像类型建立同输入验证集，再决定是否采用当前误差范围。

该移植依据 FSL 源码，受非商业 [FSL Software Licence](../../licenses/FSL-6.0.txt) 约束。第三方说明见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
