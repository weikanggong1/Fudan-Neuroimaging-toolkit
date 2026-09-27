# SynthSeg 独立入口去除 Surfa：真实 T1 对照

这次修改只覆盖公开的 33 类 `SynthSeg`、`fnit synthseg` 和 recon-all 中复用的同一入口。模型、权重、预处理、后处理及软体积计算没有改变；替换的是标签图对象、最近邻回采样、色表读取和 MGH/NIfTI 写盘。仍依赖 Surfa 的其他 FNIT 函数不在本次验收范围。

## 输入与对照

- 节点：`gpucw1`，NVIDIA H100 PCIe；Conda Python，PyTorch 2.5.1；模型运行时关闭 cuDNN TF32，使用 4 个 CPU 线程。
- recon-all 已保存的真实 T1 `orig.mgz`，SHA-256：`d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`。输入、旧版和新版代码完全相同；仅换 I/O 实现。
- 同一被试原始 `sub-01_T1w.nii.gz`，SHA-256：`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`；用于 `keep_geometry=True`。
- 外部 SynthSeg 2.0 权重，SHA-256：`f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e`。三个官方标签、名称及拓扑 `.npy` 与权重同目录。
- 旧版 `synthseg.py` SHA-256 `07189dca3f825d12ec806af444896afc61335dab78acd02a0ffae7fb2e98db07`；两版均运行完整网络，结果未从旧文件直接复制。新版 `synthseg.py` SHA-256 `ef2b21fa9b285cc813940ef363adb7bcacde60b5d3717705d92eea72e24f8577`，`synthseg_io.py` SHA-256 `0aa8f43914d727ef7bbc639e37d23e3e0bc208a09be20725db4038c6cbd8134c`。

Python 的可复现调用参数如下；`image` 是三维 T1 路径，`keep_geometry` 决定输出网格，`color_lut` 是可选色表，`weights` 是外部权重目录。`result.segmentation` 含 `float32` 三维硬标签和 4×4 RAS 仿射；CSV 是各结构的软体积（mm³）。

```python
from fnit import SynthSeg

model = SynthSeg(
    weights="/path/to/weights",  # 输入：四份 SynthSeg 2.0 模型与标签文件目录
    device="cuda:0",            # 输入：推理设备
    threads=4,                  # 输入：PyTorch CPU 线程数
)
result = model(
    image="sub-01_T1w.nii.gz",  # 输入：单幅真实三维 T1
    keep_geometry=True,         # 输入：最近邻回采样到输入网格
    color_lut=None,             # 输入：可选 FreeSurferColorLUT.txt
)
result.segmentation.save("sub-01_synthseg.nii.gz")  # 输出：硬分割图
result.write_volumes_csv(
    source="sub-01_T1w.nii.gz",       # CSV 的被试文件名
    path="sub-01_synthseg.vol.csv",  # 输出：各结构软体积
)
```

对应的 FreeSurfer 8.2 单幅命令：

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --vol sub-01_synthseg.vol.csv --threads 4 --keepgeom
```

## 精度

| 两版同输入输出 | 硬标签差异 | 仿射最大绝对差 | CSV | 解压后的完整文件 |
|---|---:|---:|---|---|
| `orig.mgz`，输出 256³ MGZ | 0/16,777,216 | 0 | 逐字节相同 | 逐字节相同，含 MGH 头、数据和尾部标签 |
| 原始 NIfTI，`keep_geometry=True`，输出 256×156×256 | 0/10,223,616 | 0 | 逐字节相同 | 逐字节相同，含 NIfTI 头和扩展 |

对已保存的真实标签图，分别写出 MGZ 和 NIfTI，带或不带 1,811 项官方 LUT，共四种组合；所有新旧输出解压后均逐字节相同。回采样到原始 T1 网格的 10,223,616 个体素也全部相同。逐项输出见 [推理 JSON](inference_real_compare.json) 和 [I/O JSON](io_real_compare.json)。压缩文件的 SHA-256 可因 gzip 时间戳不同而变化，应比较解压内容。

与 **官方 FreeSurfer** 的对照沿用先前同输入报告：`orig.mgz` 硬标签 16,777,216/16,777,216 一致，MGH 头与体素载荷一致；软体积还有 6/33 列超出 0.005 mm³ 门槛，最大差 0.04 mm³。本次新旧实现的 CSV 逐字节相同，因此这部分差异既未增大也未解决。详见[同输入官方对照](../../recon_all/python_gpu_port/CONNECTED_SYNTHSEG_GPU_20260926.md)。

## 命令行和 Python 导入

在同一 Conda 环境中，测试进程先设定 `sys.modules["surfa"] = None`，再从 `fnit` 导入 `SynthSeg` 和命令行入口。公共 `fnit synthseg` 子命令在载入其他扩散或配准模块之前解析；取自上述真实 T1 的 96×96×96 子体积，以 `--device cpu --threads 4` 完成推理、NIfTI 和 CSV 写盘。命令行阶段耗时 8.776 秒，得到 96×125×97 的输出网格和 863,672 个非零标签体素。该子体积试验只证明无 Surfa 安装时命令行可运行，不作为全脑精度或速度 benchmark；全脑精度来自上面的两个完整 GPU Python API 对照。

两个小型 I/O 回归检查也在新 Conda 环境中、禁用 Surfa 导入后通过，覆盖 MGH 标签色表和半体素最近邻规则。全局 `fnit --help` 与其他子命令仍可能导入尚未迁移的包；本次只承诺 `fnit synthseg` 子命令与 `SynthSeg` Python API。

## 时间与资源

以下是单次顺序运行的阶段耗时（秒），不是重复试验的中位数。新旧网络相同，推理耗时受 CUDA 热身和共享节点负载影响，不能归因于 I/O 代码。

| 输入与阶段 | 旧版 Surfa | 新版 nibabel/NumPy |
|---|---:|---:|
| `orig.mgz`：模型载入 | 0.756 | 0.314 |
| `orig.mgz`：预处理、推理、后处理 | 5.328 | 3.973 |
| `orig.mgz`：MGZ 保存 | 1.030 | 1.952 |
| 原始 NIfTI：模型载入 | 0.269 | 0.346 |
| 原始 NIfTI：预处理、推理、后处理和回采样 | 3.513 | 3.325 |
| 原始 NIfTI：保存 | 0.163 | 0.228 |

峰值 CUDA 已分配显存：`orig.mgz` 17,010 MiB，原始 NIfTI 16,745 MiB。单独 I/O 对照中，MGZ 保存新实现约 0.92–1.02 秒、旧实现 0.31–0.35 秒；NIfTI 保存两者约 0.13–0.15 秒。以前报告的官方 `mri_synthseg` CPU 同输入约 154 秒；本次未重复运行官方命令，也未重复整套 recon-all。
