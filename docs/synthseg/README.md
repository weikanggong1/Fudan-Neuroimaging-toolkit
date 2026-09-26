# 独立 33 类 SynthSeg

[返回首页](../../README.md) · [源码目录](../../src/fnit/synthseg_parc/) · [权重](../WEIGHTS.md) · [验证报告](../../validation/synthseg/report.public.json)

`SynthSeg` 使用 FreeSurfer 8.2 的非 robust、非 parcellated SynthSeg 2.0 模型，从单幅 T1 生成 33 类结构标签和各结构软体积。它与 WMH-SynthSeg 是不同模型；本入口不输出 WMH 标签，也不生成皮层分区。推理使用 PyTorch，不需要安装 FreeSurfer、FSL 或 TensorFlow。

## 下载模型

在仓库根目录运行，或安装后用 `fnit-setup-weights` 代替 `python tools/setup_weights.py`。配置脚本下载固定的 `.h5` 和三个 `.npy` 文件，逐个核对大小与 SHA-256，然后记录权重目录。推理本身不联网。

```bash
python tools/setup_weights.py --model synthseg --dest /path/to/weights
```

这个组仅需四个文件，总计 53,087,016 字节，不需要下载 790 MB 的 WMH-SynthSeg 权重。已有模型时可用 `--verify-only` 检查。完整文件名、官方地址及哈希见[权重说明](../WEIGHTS.md)。

## Python API

```python
from fnit import SynthSeg

model = SynthSeg(device="cuda:0", threads=4)
result = model("sub-01_T1w.nii.gz")
result.segmentation.save("sub-01_synthseg.nii.gz")
result.write_volumes_csv("sub-01_T1w.nii.gz", "sub-01_synthseg.vol.csv")
print(result.total_intracranial_mm3, result.volumes_mm3)
```

`SynthSeg(weights=None, device="cpu", threads=None)` 在构造时加载一次模型；每次调用接收一幅图像。`weights` 可传包含四个文件的目录，或直接传 `synthseg_2.0.h5` 路径；三个 `.npy` 必须与这份 `.h5` 位于同一目录。省略 `weights` 时先查 `FNIT_WEIGHTS`、配置脚本记录的目录，再查默认缓存。输入是单幅 3D `.nii`、`.nii.gz` 或 `.mgz` T1 路径。

`result.segmentation` 是 `surfa.Volume`，默认位于 SynthSeg 预处理后的 RAS 方向、约 1 mm 网格。标签编号为整数值，按原版文件格式以 `float32` 保存。`model(image, keep_geometry=True)` 会将标签以最近邻法重采样到输入网格。`color_lut="/path/to/FreeSurferColorLUT.txt"` 可选地附加色表；默认不读取 FreeSurfer 文件。

`result.volumes_mm3` 是 `{前景标签编号: 软体积}`，`result.total_intracranial_mm3` 是所有前景软体积之和，后验概率先恢复到输入方向，再按原版 NumPy float32 顺序求和并保留三位小数；`result.label_names` 对应 32 个前景结构名。CSV 列顺序、总量和近似并列标签规则与本仓库 GPU recon-all 的 `mri_synthseg` 入口相同。`result.near_tie_voxels` 记录近似并列规则相对于普通 `argmax` 更改的体素数。

## 原版命令与参数对应

本模块对应 FreeSurfer 8.2 `mri_synthseg` 的单幅、SynthSeg 2.0、非 robust、非
parcellated 路径。原版单例命令为：

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --vol sub-01_synthseg.vol.csv --threads 4
```

本包执行同一输入输出角色的命令为：

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

| 本包参数 | 原版参数 | 输入或输出 |
|---|---|---|
| `--i` | `--i` | 一幅 3D T1；本包接受 `.nii`、`.nii.gz` 或 `.mgz` 文件 |
| `--o` | `--o` | 33 类硬分割图；默认位于预处理后的 RAS、约 1 mm 网格 |
| `--csv-vols` | `--vol` | 前景结构软体积和 total intracranial volume，单位 mm³ |
| `--weights` | `--model` | 官方 `synthseg_2.0.h5`；本包还从同目录读取三份标签、名称和拓扑 `.npy` |
| `--device cpu` | `--cpu` | 强制 CPU；本包也可用 `--device cuda:N` 选择具体 GPU |
| `--threads` | `--threads` | PyTorch CPU 线程数；本包 CLI 默认 4，原版默认 1 |
| `--keep-geometry` | `--keepgeom` | 以最近邻法把硬标签重采样回输入网格 |
| `--color-lut` | `--addctab` / `--noaddctab` | 本包只在显式提供 FreeSurfer LUT 时附加色表；原版默认附加色表 |

原版的目录输入、robust、皮层分区、QC、posterior、CT、Photo-SynthSeg 和其他
模型路径；本接口没有实现这些选项，也不会把它们近似为 33 类单幅 T1 路径。

## 命令行

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

可选参数为 `--weights /path/to/weights`、`--keep-geometry` 和 `--color-lut /path/to/FreeSurferColorLUT.txt`。命令行和 Python 每次均处理一幅图像。独立入口不依赖 recon-all 的原生运行包或个人 license。

## 与 FreeSurfer 8.2 的对照验证

三例仓库公开、去面容的 T1w 分别运行 FreeSurfer 8.2.0-1 原版
`mri_synthseg` CPU 命令和本包 H100 GPU 命令。两端读取同一份官方
`synthseg_2.0.h5`，均使用 4 个 CPU 线程；本包保持默认 TF32。每一臂都在新进程中
执行，计时包含进程和框架启动、权重与输入加载、推理、后处理、CSV 和 NIfTI 写盘。

| 输出一致性指标 | 三例结果 |
|---|---:|
| shape / 数值 affine / dtype | 3/3 一致 |
| 最低逐体素标签一致率 | 0.99998817 |
| 全部病例、全部前景标签的最低 Dice | 0.99902629 |
| 各病例标签 Dice 中位数的中位数 | 0.99998952 |
| CSV 共同数值列 | 33 |
| total intracranial volume 最大绝对误差 | 556.75 mm³（0.038%） |

软体积并非逐值相同；最大差异来自 total intracranial volume。硬分割只有少量边界
体素不同，不能把上表写成逐体素完全复现。完整逐例标签 Dice、输入哈希和软体积误差
见[公开 JSON](../../validation/synthseg/report.public.json)。

| 实现 | 设备 | 完整单例命令时间，中位数 [最小–最大] |
|---|---|---:|
| FreeSurfer 8.2.0-1 `mri_synthseg` | CPU | 156.758 [142.195–187.271] s |
| FNIT `synthseg` | H100 GPU | 5.867 [5.666–5.976] s |

两组命令按病例顺序串行运行于共享 gpucw1 节点；这是三例观察到的墙钟时间，不是隔离
节点后的硬件加速试验。

下图使用公开 `sub-02` T1w。两行分别为轴位和冠状位；中间两列在同一 SynthSeg
输出网格上叠加原版和本包标签，最后一列标出标签不同的体素。完整三维指标来自上表，
不从二维图片估计。

![公开 T1w、FreeSurfer SynthSeg 与 FNIT SynthSeg 的标签比较](figures/synthseg_comparison.png)

公开 CLI 与 Python API 调用同一套 33 类推理与软体积代码，相应接口测试见
[`tests/synthseg_parc/`](../../tests/synthseg_parc/)。该对照衡量本包对参考实现的
复现程度；三例没有人工结构分割真值，不能解释为临床分割准确率。验证范围只覆盖本页
声明的单幅 T1、SynthSeg 2.0、非 robust、非 parcellated 路径。

另有单例同输入 GPU 精度对照：关闭 SynthSeg 卷积 TF32 后，硬分割与原版 16,777,216 个体素全部一致；经单体素临界阈值修正及按官方 `float32` 格式重写已保存 CSV 后，33 列软体积最大差 0.04 mm³，仍有 6 列未达到现有 0.005 mm³ 统计表门槛。格式检查未重跑 GPU 推理；阈值修正仅在这一输入上验证。详见[同输入 GPU 对照](../../validation/recon_all/python_gpu_port/CONNECTED_SYNTHSEG_GPU_20260926.md)。
