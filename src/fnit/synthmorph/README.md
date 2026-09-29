# SynthMorph 源码目录

本目录实现 FreeSurfer 8.2 `mri_synthmorph` 对应的 PyTorch 配准路径。`models.py` 定义网络并读取官方 HDF5 权重，`spatial.py` 实现采样、积分和变换组合，`pipeline.py` 提供 `SynthMorph`、`RegistrationResult` 和 `apply_transform`，`fsl_warp.py` 将联合/非线性模型的 RAS warp 转为 FSL relative warp。运行时不调用 FreeSurfer、TensorFlow、VoxelMorph 或 Neurite。

## Python 单被试示例

```python
from fnit.synthmorph import SynthMorph, apply_transform, convert_warp_to_fsl
from fnit.applywarp import TorchApplyWarp

model = SynthMorph(
    weights="/path/to/weights",  # 权重输入：官方 SynthMorph HDF5 文件所在目录
    device="cuda:0",  # 计算设备；可改为 "cpu"
    model="joint",  # joint 同时估计仿射和非线性变换
    extent=256,  # 网络空间每轴体素数；支持 192 或 256
    hyper=0.5,  # 非线性正则化参数
    steps=7,  # scaling-and-squaring 积分次数
)
result = model(
    moving="moving_T1w.nii.gz",  # 输入：待变换的单帧 3D 图像
    fixed="fixed_T1w.nii.gz",  # 输入：目标图像和 moved 输出网格
    init=None,  # 可选输入：带几何信息的初始 LTA 仿射
    mid_space=False,  # 是否使用初始仿射的中间空间
    header_only=False,  # 仅 affine/rigid 支持只修改头信息
    output_dir=None,  # 可选调试输出目录
)
result.moved.save(path="moving_in_fixed.nii.gz")  # 输出路径：moving 在 fixed 网格的图像
result.transform.save(path="moving_to_fixed.mgz")  # 输出路径：moving→fixed 变换
fsl_warp = convert_warp_to_fsl(
    warp=result.transform,  # 输入：joint/deform 的 fixed 网格 RAS 位移场
    moving="moving_T1w.nii.gz",  # 输入：配准时的 moving 图像
    fixed="fixed_T1w.nii.gz",  # 输入：配准时的 fixed 图像
)
fsl_warp.save(path="moving_to_fixed_fsl_warp.nii.gz")  # 输出：FSL intent-2006 相对位移场
TorchApplyWarp(device="cuda:0").run(
    input="moving_T1w.nii.gz",  # 输入：原 moving 网格上的图像
    reference="fixed_T1w.nii.gz",  # 输入：目标网格
    output="moving_in_fixed_by_fsl_warp.nii.gz",  # 输出：重采样图像
    warp="moving_to_fixed_fsl_warp.nii.gz",  # 输入：刚转换的 warp
    interpolation="trilinear",  # 连续图像使用三线性
    warp_convention="auto",  # intent 2006 自动按 relative 读取
    output_dtype="float",  # 输出 float32
)

labels = apply_transform(
    image="moving_labels.nii.gz",  # 输入：与 moving 几何一致的标签图
    transformation=result.transform,  # 输入：moving→fixed 的带几何变换
    method="nearest",  # 标签图使用最近邻插值
    fill=0,  # 视野外填充值
    dtype="int16",  # 输出数据类型
    header_only=False,  # False 表示重采样数据
)
labels.save(path="labels_in_fixed.nii.gz")  # 输出路径：重采样后的标签图
```

`RegistrationResult` 包含 moving→fixed 图像 `moved`、fixed→moving 图像 `fixed_moved`、正向变换 `transform` 和反向变换 `inverse`。affine/rigid 变换建议保存为 `.lta`，joint/deform 的 RAS 位移场建议保存为 `.mgz`；普通三通道数组缺少 source/target geometry，不能直接替代。`convert_warp_to_fsl` 必须同时获得原 moving 和 fixed 图像；生成的 NIfTI 可直接交给本包或 FSL `applywarp`。

## 命令行与原软件对应

```bash
fnit synthmorph moving_T1w.nii.gz fixed_T1w.nii.gz \
  --model joint --device cuda:0 --weights /path/to/weights \
  -o moving_in_fixed.nii.gz -t moving_to_fixed.mgz \
  --fsl-warp moving_to_fixed_fsl_warp.nii.gz

fnit applywarp --in moving_T1w.nii.gz --ref fixed_T1w.nii.gz \
  --warp moving_to_fixed_fsl_warp.nii.gz --device cuda:0 \
  --out moving_in_fixed_by_fsl_warp.nii.gz

mri_synthmorph register -m joint \
  -o moving_in_fixed.nii.gz -t moving_to_fixed.mgz \
  moving_T1w.nii.gz fixed_T1w.nii.gz
```

第一个位置参数是 moving，第二个是 fixed；`-o` 保存 fixed 网格上的 moving，`-t` 保存 moving→fixed 变换，`--fsl-warp` 保存可由 `applywarp` 读取的 fixed 网格位移场。完整参数、双向输出、apply 命令、源码对应、当前 benchmark 和示意图见[功能说明](../../../docs/synthmorph/README.md)。

0.14 的 12 例真实 T1w 报告由 `pipeline.py` `e680d3…`、`spatial.py` `9c629a…` 生成。当前两文件的 `70e97c…`、`dab615…` 保留 registration linear 路径，关系见[源码等价证明](../../../validation/runtime_dependencies/synthmorph_linear_source_equivalence.public.json)；12 例没有重跑新增的 FSL warp 输出。该旧对照中，GPU moved Pearson 最低 `0.994337`，位移向量平均误差均值 `0.079139 mm`，FNIT/FreeSurfer 完整命令中位数为 `15.780/116.594 s`，FNIT 峰值 CUDA allocation 为 `13.426 GB`；CPU moved Pearson 最低 `0.994810`。新增转换在一对公开真实 T1w 上与 FSL applywarp 对照，Pearson 为 `0.999999999967`，Torch/FSL 含读写时间为 `2.795/40.892 s`。计时设备分别是 GPU/CPU；具体输出、精度、图片和边界见[功能说明](../../../docs/synthmorph/README.md)。
