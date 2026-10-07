# FNIRT CPU 采样修复候选：真实体积对照（2026-10-06）

## 1. 功能简介

修正 CPU 三线性采样的运算顺序：不取导数时按原 plain 的差值混合顺序计算；取导数时保留原 partial 中三个左项的 Double 提升及 Float32 存储。在同一真实体积和16,128个坐标上，两模式的值、有效性及三个导数均与原生结果逐位一致。

本目录提供已验证的独立候选 [candidate_cpu_sampler.py](candidate_cpu_sampler.py)。FNIRT 默认注册链尚未接入，GPU代码未修改。实际执行源码与公开源码仅模块说明不同，剩余完整 AST 相同；两个 SHA 分列于 [manifest](manifest.public.json)。

## 2. Python 调用、输入与输出

函数为 `try_sample_cpu(volume, coordinates, derivatives=True)`，只处理 CPU：

| 参数 | 含义及格式 |
| --- | --- |
| `volume` | 连续的 CPU `torch.float32` 三维 `[X,Y,Z]` 体积。坐标以该数组的体素索引为单位；函数不读取或转换影像 affine。 |
| `coordinates` | 连续的 CPU `torch.float32` 数组 `[3,...]`，第一维依次为 X/Y/Z，后续维度是输出布局；与 volume 使用同一体素坐标系。 |
| `derivatives` | 默认 `True`，返回值及三个体素坐标导数；`False` 使用原 plain 顺序、只计算值。 |

返回 `(values, valid, gradient)`：values 为 Float32、shape 等于坐标的后续维度；valid 为相同 shape 的 bool；gradient 为 `[3,...]` Float32，轴顺序 X/Y/Z，`False` 时为 `None`。返回值和导数已经乘以 valid。出界角点沿用成熟 helper 的 clamp 后乘 inside 语义，包括符号零。

非CPU、非Float32、非连续或非strided张量、空数据、非有限值、极端坐标、需要自动微分的输入、dual/vmap或负/共轭视图会返回 `None`，由调用者的原通用路径处理；中间值非有限时也返回 `None`。函数观察每次输入，不缓存图像或坐标。Numba线程不超过当前 Torch 线程预算，调用后恢复原Numba设置。Double仅用于CPU的三个标量表达式，没有Double影像或GPU Double运算。

以下示例要求影像与坐标已经处于同一逻辑XYZ体素坐标系。它不执行世界坐标变换或FSL存储方向转换：

```python
import importlib.util
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

# sampler_source_path：本目录的独立候选；不替换生产注册器。
sampler_source_path = Path(
    "validation/fnirt_cpu_sampler_case_20261006/candidate_cpu_sampler.py"
).resolve()
sampler_spec = importlib.util.spec_from_file_location(
    "fnit_validation_cpu_sampler", sampler_source_path
)
sampler_module = importlib.util.module_from_spec(sampler_spec)
sys.modules[sampler_spec.name] = sampler_module
sampler_spec.loader.exec_module(sampler_module)

# moving_image_path：待采样的三维影像；coordinates_path：同帧体素坐标[3,...]。
moving_image_path = Path("moving_in_query_voxel_frame.nii.gz")
coordinates_path = Path("coordinates_xyz.npy")
moving_array = nib.load(moving_image_path).get_fdata(dtype=np.float32)
coordinate_array = np.load(coordinates_path, allow_pickle=False)
moving_tensor = torch.from_numpy(np.ascontiguousarray(moving_array))
coordinate_tensor = torch.from_numpy(
    np.ascontiguousarray(coordinate_array, dtype=np.float32)
)
# cpu_threads：本次CPU线程预算；compute_derivatives：是否需要三个体素导数。
cpu_threads = 8
compute_derivatives = True
torch.set_num_threads(cpu_threads)
sampling_outputs = sampler_module.try_sample_cpu(
    moving_tensor, coordinate_tensor, derivatives=compute_derivatives
)
if sampling_outputs is None:
    raise ValueError("输入需交给原通用采样路径处理")
sampled_values, sampled_valid, sampled_voxel_gradient = sampling_outputs
```

依赖沿用FNIT Conda中的PyTorch、NumPy及Numba，没有新增安装依赖。

## 3. 命令行调用

这是内部采样候选，没有独立生产CLI。读取聚合结果：

```bash
python -m json.tool validation/fnirt_cpu_sampler_case_20261006/manifest.public.json
```

## 4. 原软件对应

对应 NEWIMAGE 的 `interpolate` 和 `interp3partial`，由FNIRT重采样/导数阶段调用，原内部函数没有单独命令行。本轮复用已捕获的实际原DSO结果，没有重新运行完整 `fnirt`。[自然捕获](../fnirt_natural_sampler_capture_20261006/README.md)与[公式控制](../fnirt_recorded_operand_reference_20261006/README.md)分别记录原delegate身份和同操作数结果。

官方保存的平滑moving只用于隔离验证；按4字节进行X方向置换后，与实际原采样体积的端点SHA相同。本轮恢复该实际体积及记录坐标，不重算平滑。官方数组不会进入生产。

## 5. 真实精度、耗时及可视化

体积含18,579,456个Float32值。16,128个坐标的floor、fraction、derived valid及129,024个角点先通过身份门；其中13,652个角点出界，padding及符号零均一致。current/candidate各调用两次，原生、blur、solver和GPU新调用均为0。

| 对实际原生输出的差异 | 当前helper不同words | 候选不同words |
| --- | ---: | ---: |
| plain value | 2,338 | 0 |
| partial value | 813 | 0 |
| partial gx | 0 | 0 |
| partial gy | 913 | 0 |
| partial gz | 1,520 | 0 |

上述raw与derivedvalid-masked结果分别计量，候选均逐位一致；API的valid与捕获derived valid一致。derived valid不是原caller datamask trace。坐标按XX位模式关联，尚未证明其与历史FNIT目标网格的一一对应；旧2,425word差异的整体原因仍未证明。

| 顺序API观察 | 秒 |
| --- | ---: |
| current，False / True | 1.700351 / 0.008265 |
| candidate，False / True | 1.397349 / 0.007893 |
| worker总时间 | 7.432919 |

每模块首调用包含JIT，第二模式可复用已编译类型；这是一次顺序观察，没有ABBA或额外暖调用，不能据此报告速度倍率。实际Torch/Numba intra线程均8；Torch interop96保持原值，CUDA未初始化，线程/TF32状态前后相同。owned tree峰值RSS605,282,304B，原14份文本逐项size/SHA核验、源码/输入/保存输出身份及树/锁释放门通过；独立复核还核对7个冻结API文件与19项前后绑定。该API过程没有加载FSL DSO。

随后[原checkpoint坐标核对](coordinate_followup.public.json)只构建一次成熟坐标，先复现旧坐标SHA。两侧各16,128个唯一XX位键，却没有相同键；同输入门未过，因此没有运行新采样或RHS。首次差异报告还遇到NumPy整数JSON写出错误，原失败保留，补录只读取已保存整数位模式，没有重跑坐标。公共NIfTI与NEWIMAGE内部体素方向的source-defined对应仍需验证；没有自动反射或按输出寻找坐标匹配。

随后[同目标affine v3控制](affine_coordinate_v3/README.md)已补足这一保存状态的对应关系：16128个坐标位键、顺序输出、mask和全部角点门通过；候选partial值及三导数逐位同。固定lambda的FSL累计序正梯度总相对L2降至8.7145e−10，1177个Double words仍非逐位同，逐块误差见该报告。第一组失败不改写为通过。完整FNIRT的H/PCG、动态缓存、最终配准、其他输入/线程、Tensor fallback语义及后续GPU保护仍需验收，默认实现未替换。本轮没有完整配准脑图；[原完整CPU benchmark及脑图](../../docs/fnirt/README.md#5-最新精度和运行时间)仍绑定原版本。

[随后的除法投影控制](projection_division_v1/README.md)通过48,384个Float32值含符号零的逐位门。FSL累计序正g相对L2降至5.26766e−15，三系数块误差均约1e−15；仍有1,170个Double words尾差，LM序仍约3.62e−8。该结果仅是固定lambda保存状态的CPU投影与g前缀，完整H/求解、动态缓存、最终输出、fallback与GPU保护仍待验收。

[同投影Hessian对角线对照](diagonal_only_v2/README.md)已完成：未加扰及乘1.001后的1177项对角线相对L2分别为7.08786e−16、7.08078e−16；三个392项系数块约1.5e−15至2.4e−15，整体范数由scale项主导。两向量仍各有1166个Double words不同，不能称逐位同。仅执行一次对角线计算，未计算非对角项、完整H或求解；完整缓存、非线性配准及GPU保护仍待验收。

[保存状态的 X0 单列对照](hcol_x0_v1/README.md)已完成：196项合同门通过，组合列59/1177个Double词不同，相对L2为2.94287e−15。该边界方向的候选数据项全部为正零，结果主要验证bending及接口，不能代表非零数据项或完整H。data列含冷JIT为1.362秒，worker为88.437秒；其余导入和核验时间未细分，没有API或端到端加速结论。下一步用原始输入自产内部图像与坐标，核对cf/grad缓存、FSL顺序RHS和连续PCG，再验完整CPU配准及GPU保护。

## 6. 更新及 benchmark 记录

- 既有raw moving不匹配与公式导入失败保留；随后同记录操作数控制定位plain顺序和partial提升差异。
- 2026-10-06：本轮真实体积四API一次通过，当前helper差异精确复现公式参考，候选值/导数逐位匹配；没有重复官方采样或完整配准。
- 当前公开候选保持相同计算AST，未接入默认注册器。同目标坐标、投影、对角及X0单列控制已分阶段核验；原始输入的连续求解、完整真实CPU配准和GPU保护仍待验收。

## 7. 原实现、许可及参考文献

- [FSL NEWIMAGE源码](https://git.fmrib.ox.ac.uk/fsl/newimage)、[FSL FNIRT源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)。FNIT改写遵循[FSL 6.0许可](../../licenses/FSL-6.0.txt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
