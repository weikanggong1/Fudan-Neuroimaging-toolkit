# 三例公开 FLAIR：运行 WMH-SynthSeg

[返回主页](../README.md) · [功能说明](../docs/wmh_synthseg/README.md) · [数据来源与 SHA-256](wmh_data/SOURCES.json)

`wmh_data/` 中的 `sub-02`、`sub-03`、`sub-04` 源于 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) 的 CC0 FLAIR。发布文件保留输入网格，清除了原版 SynthStrip 脑掩膜外扩 6 mm 以外的强度以及 NIfTI 头中的标识文字。仓库不包含原始影像或人工病灶标注。制作脚本、原图地址及发布文件哈希见 [`tools/prepare_wmh_examples.py`](../tools/prepare_wmh_examples.py) 和 [`SOURCES.json`](wmh_data/SOURCES.json)。

从仓库根目录执行：

```bash
python -m pip install .
python examples/check_wmh_data.py
python tools/setup_weights.py --model wmh-synthseg
```

依次安装本包、校验三份 FLAIR 的大小与 SHA-256 及 NIfTI 头，再从 FreeSurfer 官方下载并校验约 791 MB 的 WMH checkpoint。脚本会记录权重目录，推理时无需安装 FreeSurfer。GPU 推理需要与驱动兼容的 CUDA 版 PyTorch；用 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'` 检查。

先运行一例。下面的参数对应原版 `mri_WMHsynthseg --i ... --o ... --crop --save_lesion_probabilities --csv_vols ...`：

```bash
fnit wmh-synthseg \
  --i examples/wmh_data/sub-04_FLAIR.nii.gz \
  --o examples/results/wmh_single/sub-04_seg.nii.gz \
  --device cuda:0 --threads 4 --crop \
  --save_lesion_probabilities \
  --csv_vols examples/results/wmh_single/sub-04_volumes.csv
```

`--i` 读取 3D FLAIR；`--o` 写 33 类整数值标签图，文件存储类型与原版一样为 float32。`--crop` 先定位脑再限制推理区域；`--save_lesion_probabilities` 另写 `sub-04_seg.lesion_probs.nii.gz`；`--csv_vols` 写各类后验概率的软体积，单位 mm³。`--device` 与 `--threads` 设置设备和 PyTorch CPU 线程。输出位于处理后的 RAS/1 mm 网格，可能不同于输入网格。没有 GPU 时改用 `--device cpu`。

Python 调用返回影像对象和体积字典，不自动写文件：

```python
from pathlib import Path
from fnit import WMHSynthSeg

Path("examples/results/wmh_single").mkdir(parents=True, exist_ok=True)
model = WMHSynthSeg(device="cuda:0", threads=4)
result = model(
    image="examples/wmh_data/sub-04_FLAIR.nii.gz",  # 输入：单幅 3D FLAIR
    crop=True,  # 预处理：根据脑区域裁剪后推理
    save_lesion_probabilities=True,  # 输出：同时返回 WMH 概率图
)
result.segmentation.save(
    path="examples/results/wmh_single/sub-04_python_seg.nii.gz",  # 输出：33 类标签图
)
result.lesion_probability.save(
    path="examples/results/wmh_single/sub-04_python_lesion_probs.nii.gz",  # 输出：WMH 概率图
)
print(result.volumes_mm3[77])  # WMH 软体积；不是硬标签 77 的体素数。
```

CLI 会自动创建输出父目录；Python 示例每次调用模型处理一幅影像。

两版程序使用同一份发布输入的比较结果及并排图制作步骤见 [WMH 图示](../docs/figures/README.md) 和 [3 例对照](../validation/wmh/README.md)。这些公开图像没有人工 WMH 真值；这里的检查用于比较接口与输出，不能评价临床检测精度。
