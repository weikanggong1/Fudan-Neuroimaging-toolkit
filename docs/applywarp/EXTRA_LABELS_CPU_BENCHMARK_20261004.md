# ApplyWarp：真实解剖标签 CPU 补充对照

## 1. 功能

本例补齐真实离散解剖标签的完整 CPU 处理：FSL 分发的 Harvard–Oxford cortical `maxprob-thr0` 1 mm atlas 重采样到同一 MNI152 空间的 2 mm 网格。输入为已存在的 atlas，保留原 qform/sform；不生成模拟标签。此次是同空间标签网格转换，不将结果解释为非线性配准精度。

## 2. Python、输入与输出

```python
import nibabel as nib
import numpy as np
import torch
from fnit.applywarp import TorchApplyWarp

torch.set_num_threads(8)  # 同一 CPU 预算；1 线程观察则设置 1
source_label_path = "/path/to/FSL/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz"
reference_mni_path = "/path/to/FSL/data/standard/MNI152_T1_2mm.nii.gz"
same_space_fsl_matrix = np.eye(4)  # 本例现场核对 scaled-mm/header 后得到的 identity
label_warper = TorchApplyWarp(device="cpu")
output_label_path = label_warper.run(
    source_label_path, reference_mni_path, "/path/to/output/cortical_labels_2mm.nii",
    premat=same_space_fsl_matrix,
    interpolation="nearest",  # 保留离散标签
    output_dtype="short",  # int16 输出，与完整官方命令保持一致
)
output_label_image = nib.load(output_label_path)
```

- **输入：** 完整 `182×218×182` 真实 1 mm 分区 NIfTI，以及完整 `91×109×91` 2 mm MNI152 参考；仿射矩阵为 FSL scaled-mm 源→参考变换。此次根据实际 header 算得 identity。
- **输出：** 完整 2 mm 标签 NIfTI；保存 dtype、shape、affine、qform/sform、缩放 slope/intercept 与官方输出逐项核对。
- **参数：** `device="cpu"` 固定本轮 CPU；`nearest` 不混合标签；`output_dtype="short"` 将文件写成 int16；`premat` 指定已核验的同空间矩阵。其余参数保持功能页默认值。benchmark manifest 中的 `label_image=true` 声明真实离散标签，启用逐标签 Dice；它不是生产 API 参数。

## 3. 命令行

```bash
fnit applywarp --in /path/to/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz \
  --ref /path/to/MNI152_T1_2mm.nii.gz \
  --out /path/to/output/cortical_labels_2mm.nii \
  --premat /path/to/identity.mat --interp nearest --datatype short \
  --device cpu
```

独立补测使用项目统一 `tools/benchmark_multimodal_cpu.py` 和真实输入 manifest，显式 `--single-observation --threads 1,8`，与 World 对照共用 CPU 计时锁。完整 API 一次读取、计算、保存所有输出；完整进程时间另行记录。

## 4. 官方命令

```bash
FSLOUTPUTTYPE=NIFTI /path/to/FSL/bin/applywarp \
  --in=/path/to/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz \
  --ref=/path/to/MNI152_T1_2mm.nii.gz \
  --out=/path/to/output/cortical_labels_2mm.nii \
  --premat=/path/to/identity.mat --interp=nn --datatype=short
```

官方 nearest 完整操作链产生同一标签输出；benchmark manifest 的 `label_image=true` 用于结果检查，没有对应的 FSL 参数。固定参照为 FSL 6.0.7.4，官方仅在 benchmark 参照端调用。

## 5. 精度、耗时与图像范围

使用同一 nodecw10、相同 CPU 亲和性；1/8 是两方的线程预算上限，报告另记实际 CPU 时间。每预算一次完整观察，不据此推断稳定加速比。完整 `91×109×91` 输出的 902,629 值逐值一致，49 个实际出现的标签（含背景）Dice 全部为 1。两方均保存 int16、slope=1、intercept=0；shape、affine、2 mm voxel size、mm/sec、qform/sform=4 完全一致。

| CPU 预算 | 官方完整进程 | FNIT 完整进程 | FNIT 首次函数读写 | 官方实际平均 CPU 核数 | FNIT 实际平均 CPU 核数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2061 s | 2.1487 s | 0.3113 s | 0.938 | 0.939 |
| 8 | 0.2000 s | 2.1532 s | 0.3833 s | 0.943 | 1.484 |

完整进程包含解释器、框架导入和全部输入/输出；首次函数时钟排除进程及 Torch/适配器初始化，计入该调用触发的功能导入。函数与官方 CLI 的计时范围不同，不混用它们计算速度比。8 的预算未使官方成为 8 核并行程序；实际 CPU 核数为子进程 user+system CPU 时间除以该次 wall time。

**本标签小任务精度通过，完整进程速度目标未达到。** 数值与逐标签 Dice、每份 dtype/scaling/空间合同、源程序及公共输入 SHA-256 见[聚合报告](extra_labels_cpu_benchmark_20261004.public.json)。这两次 single observation 补齐真实标签输入，独立于[普通入口的 50 组配对](CPU_BENCHMARK_20261004.md)。

本报告只发布聚合指标与已分发 atlas 的 SHA-256；不重新分发 atlas 原文件，不绘制新增个体影像。公开 T1 示例脑图见 [普通入口 CPU 报告](CPU_BENCHMARK_20261004.md)。

## 6. 更新记录

| 日期 | 范围 |
|---|---|
| 2026-10-04 | 补充真实 Harvard–Oxford 离散标签的完整 1 mm→2 mm CPU 对照；未修改生产采样代码。 |

## 7. 参考和原实现

- [FSL FNIRT User Guide：ApplyWarp](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#applywarp)。
- [FSL atlases](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html)，Harvard–Oxford atlas 的来源与引用。
- [FSL 6.0 许可](../../licenses/FSL-6.0.txt)；atlas 留在原分发目录，本文不授予重新分发权利。
- [FNIT ApplyWarp 功能页](README.md) 和 [普通入口 CPU 主报告](CPU_BENCHMARK_20261004.md)。
