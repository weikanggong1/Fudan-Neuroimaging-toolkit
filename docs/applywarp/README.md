# TorchApplyWarp：变形场重采样

`TorchApplyWarp` 是 FSL `applywarp` 已验证子集的 PyTorch 实现。运行时不调用
FSL，可在 CPU 或 CUDA 上读取 FSL dense warp 以及 FNIRT cubic coefficient
文件，并把 3D/4D 输入重采样到 reference 网格。

CUDA 运行保持 TF32 matrix/cuDNN 全局开关；FSL 坐标矩阵、坐标网格和 cubic
coefficient 展开显式使用 float64，避免 TF32 改变重采样坐标。影像插值张量与输出
使用所声明的 float32/float64 dtype，不自动使用 float16 或 bfloat16。实际全局开关
记录在 `result.qc["tf32"]`。

## Python 调用

```python
from fnit.applywarp import TorchApplyWarp

warper = TorchApplyWarp(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU
)
result = warper(
    input="subject_GM.nii.gz",  # 输入：待重采样的 3D 图像或 4D 序列
    reference="GM_template.nii.gz",  # 输入：定义输出 shape 与空间几何的参考图
    warp="subject_to_template_coef.nii.gz",  # 输入：dense warp 或 intent-2007 系数
    premat=None,  # 输入：warp 之前的 FLIRT scaled-mm 矩阵
    postmat=None,  # 输入：warp 之后的 FLIRT scaled-mm 矩阵
    interpolation="trilinear",  # 插值：连续图像使用三线性
    warp_convention="auto",  # warp 约定：按 FSL 规则自动判定 relative/absolute
    output_dtype="float",  # 输出：写盘时使用 float32
)
result.save(path="subject_GM_to_template.nii.gz")  # 输出路径：reference 网格重采样图像
```

`result.image` 是 reference 网格上的 NIfTI image，`result.valid_mask` 标记 warp
和输入采样都有效的体素，`result.qc` 记录实际使用的 intent、warp convention、
插值和 dtype。需要一步写盘时使用：

```python
TorchApplyWarp(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU
).run(
    input="subject_GM.nii.gz",  # 输入：待重采样图像
    reference="GM_template.nii.gz",  # 输入：输出参考网格
    output="subject_GM_to_template.nii.gz",  # 输出：写盘路径
    warp="subject_to_template_coef.nii.gz",  # 输入：intent-2007 系数文件
    interpolation="trilinear",  # 插值：连续图像使用三线性
)
```

函数形式从子包导入，避免与 `fnit.applywarp` 模块同名：

```python
from fnit.applywarp import applywarp

result = applywarp(
    input="subject_GM.nii.gz",  # 输入：待重采样图像
    reference="GM_template.nii.gz",  # 输入：输出参考网格
    warp="subject_to_template_warp.nii.gz",  # 输入：dense warp
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU
)
```

## 命令行调用

以下两行执行同一操作：

```bash
fnit-applywarp \
  --in subject_GM.nii.gz \
  --ref GM_template.nii.gz \
  --warp subject_to_template_coef.nii.gz \
  --interp trilinear \
  --datatype float \
  --device cuda:0 \
  --out subject_GM_to_template.nii.gz

fnit applywarp \
  --in subject_GM.nii.gz \
  --ref GM_template.nii.gz \
  --warp subject_to_template_coef.nii.gz \
  --interp trilinear \
  --datatype float \
  --device cuda:0 \
  --out subject_GM_to_template.nii.gz
```

这对应 FSL：

```bash
applywarp \
  --in=subject_GM.nii.gz \
  --ref=GM_template.nii.gz \
  --warp=subject_to_template_coef.nii.gz \
  --interp=trilinear \
  --datatype=float \
  --out=subject_GM_to_template.nii.gz
```

| Python | 本包 CLI | FSL `applywarp` | 含义 |
|---|---|---|---|
| `input` | `--in` | `--in` | 待重采样的 3D image 或 4D series |
| `reference` | `--ref` | `--ref` | 决定输出 shape、voxel size、qform 和 sform |
| `warp` | `--warp` | `--warp` | FSL dense field 或 intent 2007 coefficient；省略时只应用矩阵 |
| `premat` | `--premat` | `--premat` | input → warp source 的一个 4×4 FLIRT scaled-mm 矩阵 |
| `postmat` | `--postmat` | `--postmat` | warp reference → output reference 的一个 4×4 FLIRT scaled-mm 矩阵 |
| `interpolation` | `--interp` | `--interp` | 输入 image 的 `trilinear` 或 `nearest`/`nn` 插值 |
| `warp_convention` | `--rel` / `--abs` | `--rel` / `--abs` | 只控制无专用 intent 的 dense field；`auto` 使用 FSL 的标准差判别 |
| `output_dtype` | `--datatype` | `--datatype` | `char`、`short`、`int`、`float`、`double`；省略时遵循 FSL 默认规则 |
| `device` | `--device` | 无 | `cpu`、`cuda` 或 `cuda:N` |
| 无 | `--overwrite` | FSL 默认覆盖 | 允许 CLI 覆盖已有输出 |

CLI 当前只接收一个 4×4 `premat` 和一个 4×4 `postmat`。FSL 支持的逐帧堆叠
矩阵不在本实现范围内。

## 坐标和变换方向

FSL scaled-mm 不是 NIfTI world-RAS。它按 header voxel size 缩放体素轴；当
voxel-to-world affine 的行列式为正时，还会翻转并平移第一存储轴。dense field
的三个分量是这组三个 scaled-mm 轴上的毫米值。

对 output/reference voxel `x`，设 `Vref` 和 `Vin` 分别为 reference 与 input
的 voxel → FSL scaled-mm 矩阵。本实现执行以下 pull mapping：

```text
q = Vref x
r = inverse(postmat) q

dense relative: s = r + sample(warp, r)
dense absolute: s = sample(warp, r)
intent 2007:    s = inverse(embedded_affine) r + sample(cubic_residual, r)

u = inverse(Vin) inverse(premat) s
output(x) = input(u)
```

因此，从输入到输出描述线性步骤时，顺序是 `premat → warp → postmat`；真正
重采样时从输出点反查输入点，按上式使用逆矩阵。warp field 本身始终用
trilinear 插值；`--interp` 选择的是最后对输入 image 的插值。nearest 使用
FSL 对正坐标的 half-up 规则。

SynthMorph/Surfa 的 `disp_ras` warp 同样可表示 target/output → source/input 的
pull mapping，但其位移分量属于 world-RAS，并携带 source/target geometry。
FSL warp 的分量属于 FSL scaled-mm，FNIRT coefficient 还把 residual、knot
spacing 和 affine 分开保存。两者方向角色相近，数组和 header 不能直接互换。
本模块不直接读取 FreeSurfer `.m3z`、`.mgz` warp 或 `surfa.Warp`。本包
`fnit.synthmorph.convert_warp_to_fsl(warp, moving=..., fixed=...)` 使用原
moving/fixed 几何生成 intent-2006 FSL relative warp，随后可由本模块的
`warp` 参数读取；调用方法见 [SynthMorph](../synthmorph/README.md)。

## 支持的 warp 文件

| NIfTI intent | FSL 表示 | 状态 | 解释 |
|---:|---|---|---|
| 0 或其他 | 4D dense field，最后一维为 3 | 支持 | `--rel`、`--abs` 或 FSL `auto` 判别 |
| 2006 | FNIRT dense displacement | 支持 | 按 FSL 规定始终视为 relative，忽略 `--abs` |
| 2007 | cubic B-spline coefficients | 支持 | 解码 qform offset field size、intent voxel size、pixdim knot spacing、raw coefficient order 和 sform embedded affine |
| 2008 | DCT coefficients | 拒绝 | 抛出 `NotImplementedError`，不做近似 |
| 2009 | quadratic B-spline coefficients | 拒绝 | 抛出 `NotImplementedError`，不做 cubic 替代 |

intent 2007 的运行时解码完全在 PyTorch 中完成，不调用 `fnirtfileutils`。
`fnirtfileutils` 只用于外部验证。

## 输出约定和实现边界

输出 shape、affine、qform/sform code 和空间 voxel size 取自 reference；4D 输入
的时间间隔取自 input。显式整数 dtype 使用 FSL 的向零截断。未指定 dtype 时，
float/double 输入保持 dtype；整数输入在输出动态范围小于 100 时写 float32，
否则保持输入 dtype。

已实现的 FSL 子集包括 reference-grid 输出、dense relative/absolute、intent
2006、intent 2007、一个 premat、一个 postmat、trilinear/nearest 和 FSL dtype
规则。以下 FSL 选项没有实现：`sinc`、输入 image 的 cubic `spline` 插值、
supersampling、`--paddingsize`、`--mask`、`--usesqform` 和逐帧矩阵。CLI 会拒绝
这些选项，代码不会把它们降级成已支持模式。

## 当前真实数据验证

2026 年 9 月 27 日用 1 例去标识化的真实 UKB 格式 dMRI 完成了独立验证。输入是
TBSS 预处理后的 native FA，reference 是 `FMRIB58_FA_1mm`，warp 是 FSL FNIRT
生成的 intent-2007 cubic coefficient；FSL 直接 `applywarp` 输出作为 reference。
验证运行的 `src/fnit/applywarp/core.py` SHA-256 为
`cf6de438f3ac1551804d38682ce3fbb11b8fb042ad881562ebc93aada80f2df5`。这是当时的官方对照快照；2026-10-02 采样计划的当前 FNIT 对照结果见下文。

| 检查 | 结果 |
|---|---:|
| 输出 shape / affine / dtype | 全部一致 |
| union-support voxel | 1,548,144 |
| Pearson r | 0.999999999994 |
| MAE | 3.55e-7 |
| RMSE | 5.87e-7 |
| 最大绝对误差 | 1.18e-5 |
| 有效采样体素比例 | 0.565360 |
| H100 peak CUDA allocation | 1.958 GB |

FNIT 先预热一次，再测三次；每次包含 NIfTI 读取、coefficient 展开、GPU
重采样和输出回传 CPU，不含最后写盘，耗时为 `0.337/0.259/0.364 s`，中位数
`0.337 s`。FSL 6.0.7.4 的三次外部命令包含读取、计算和写盘，耗时为
`6.19/5.11/5.67 s`，中位数 `5.67 s`。两者计时边界不同，因此不据此计算加速比。

![真实 FA 的 FSL applywarp 与 FNIT TorchApplyWarp 对照](figures/applywarp_real_fa_comparison.png)

机器可读结果见
[`report.real.current.json`](../../validation/applywarp/report.real.current.json)，复现脚本见
[`validate_real.py`](../../validation/applywarp/validate_real.py)。报告只发布数据类型和文件
SHA-256，不发布受试者标识。这个结果验证了单例连续 FA、trilinear 和 intent-2007
coefficient；不能外推到其他病例、nearest、dense warp 或未实现选项。

新增的 SynthMorph→FSL intent-2006 dense warp 使用另一对仓库内公开的真实 T1w 验证：
同一 warp 经本包 GPU TorchApplyWarp 与 FSL CPU applywarp 后，非零并集 Pearson
为 `0.999999999967`、MAE 为 `0.001570`；两次含读写计时分别为
`2.795 s` 和 `40.892 s`。这是不同计算设备上的单次测量，完整输出、图像和
复现脚本见 [SynthMorph 的转换对照](../synthmorph/README.md#fsl-warp-转换真实-t1w)。

## 测试

单元测试位于 [`tests/applywarp`](../../tests/applywarp)。当前真实数据复现入口为
[`validation/applywarp/validate_real.py`](../../validation/applywarp/validate_real.py)；它要求调用方显式提供输入、FSL reference、coefficient warp 与输出目录。测试和验证期间可以安装 FSL 生成参照，`TorchApplyWarp` 的正常运行不调用 FSL。

## 2026-10-02：同网格多指标图传播

新增采样计划先准备固定 warp/affine 的坐标，再对每张图独立插值。源 shape、affine、header pixdim 和 FSL scaled-mm 网格须完全相同；参考图还核验完整 header 和 extensions。原 API、float64 坐标计算、float32 插值与输出规则保持兼容。

2026-10-02 在 H100 上，使用既有真实数据的九张指标图与固定 FNIRT coefficient，输出网格为 `182×218×182`，CPU 线程为 8，PyTorch CUDA 分配限额为 20,000,000,000 bytes。每轮交替先后顺序，三轮的九图解码体素、完整 header、affine、有效掩膜及 TBSS standard/skeleton 后处理结果均逐值一致。

| 九图传播计时 | 原逐图调用 | 复用采样计划 |
|---|---:|---:|
| 首轮，含冷启动 | 0.702855815 s | 0.248454907 s |
| 后两轮热调用中位数 | 0.365734295 s | 0.180841511 s |

热调用观察到约 **2.02 倍**速度。新方案的计时包含准备计划、九图采样与输出回传；输入加载、FA 预处理和输出写盘在计时外。这是固定形变的九图传播结果，不代表整条 pipeline 或配准估计的加速比。完整参数和 Python 示例见[组件文档](../../src/fnit/applywarp/README.md)，三轮记录及复现命令见[九图验收](../../validation/dmri_pipeline/map_propagation_20261002.md)，整体进展见[主报告](../../validation/dmri_pipeline/lossless_20261002.md)。

## Reference

- 参考文献：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。 `applywarp` 没有单独的方法论文。
- 原实现代码库：[FSL `fnirt`（含 `applywarp`）](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
