# SynthMorph 源码目录

本目录实现 FreeSurfer 8.2 `mri_synthmorph` 对应的 PyTorch 配准路径。`models.py` 定义网络并读取官方 HDF5 权重，`spatial.py` 实现采样、积分和变换组合，`pipeline.py` 提供 `SynthMorph`、`RegistrationResult` 和 `apply_transform`，`__init__.py` 导出公开接口。运行时不调用 FreeSurfer、TensorFlow、VoxelMorph 或 Neurite。

## Python 单被试示例

```python
from fnit.synthmorph import SynthMorph, apply_transform

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

`RegistrationResult` 包含 moving→fixed 图像 `moved`、fixed→moving 图像 `fixed_moved`、正向变换 `transform` 和反向变换 `inverse`。affine/rigid 变换建议保存为 `.lta`，joint/deform 的 RAS 位移场建议保存为 `.mgz`；普通三通道数组缺少 source/target geometry，不能直接替代。

## 命令行与原软件对应

```bash
fnit synthmorph moving_T1w.nii.gz fixed_T1w.nii.gz \
  --model joint --device cuda:0 --weights /path/to/weights \
  -o moving_in_fixed.nii.gz -t moving_to_fixed.mgz

mri_synthmorph register -m joint \
  -o moving_in_fixed.nii.gz -t moving_to_fixed.mgz \
  moving_T1w.nii.gz fixed_T1w.nii.gz
```

第一个位置参数是 moving，第二个是 fixed；`-o` 保存 fixed 网格上的 moving，`-t` 保存 moving→fixed 变换。完整参数、双向输出、apply 命令、源码对应、当前 12 例 benchmark 和示意图见[功能说明](../../../docs/synthmorph/README.md)。

0.14 的 12 例真实 T1w 报告由 `pipeline.py` `e680d3…`、`spatial.py` `9c629a…` 生成。当前 `70e97c…`、`dab615…` 仅改变独立 apply 的 nearest 语义；registration 的 linear 路径按[源码等价证明](../../../validation/runtime_dependencies/synthmorph_linear_source_equivalence.public.json)继承，没有 fresh current-hash 完整重跑。GPU 下 moved Pearson 最低 `0.994337`，位移向量平均误差均值 `0.079139 mm`，FNIT/FreeSurfer 完整命令中位数为 `15.780/116.594 s`，FNIT 峰值 CUDA allocation 为 `13.426 GB`。CPU 下 moved Pearson 最低 `0.994810`，位移向量平均误差均值 `0.0000677 mm`，FNIT/FreeSurfer 完整命令中位数为 `140.185/164.549 s`。nearest 仅有独立 CPU/H100 语义测试，不能借用这些 registration 指标。输出合同、逐例结果、公开 OpenNeuro 示意图和限制见[功能说明](../../../docs/synthmorph/README.md)。
