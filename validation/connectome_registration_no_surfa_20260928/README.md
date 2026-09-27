# Connectome 自动配准移除 Surfa：同输入核对

## 改动和函数合同

`fnit.connectome.pipeline._registration` 保留原有 6 DOF / normmi TorchFLIRT 搜索，只将旧 `sf.ImageGeometry`、`sf.Volume` 和 `sf.load_volume` 分别替换为仓库内 `Volume` 与 FLIRT 的 NiBabel 路径读取。输入为三维 `b0: torch.Tensor`、DWI 体素到 RAS 世界坐标的 `dwi_affine: torch.Tensor[4,4]`、配对 T1 文件路径 `t1_path`、运行设备 `device: torch.device`。输出为 `torch.float32[4,4]`，映射 **DWI world-RAS 到 T1 world-RAS**。完整 connectome 的输入、四张矩阵和输出目录说明见[功能文档](../../docs/connectome/README.md)。

官方软件对应命令如下。`-in` 是校正 DWI 的平均 b0，`-ref` 是配对 T1，`-cost normmi` 是归一化互信息，`-dof 6` 是刚性配准，`-omat` 保存 FSL scaled-mm 矩阵；比较前需将官方矩阵转成 world-RAS。

```python
import nibabel as nib
import numpy as np
import torch
from fnit.connectome.pipeline import _registration

b0_image = nib.load("mean_b0.nii.gz")
transform = _registration(
    b0=torch.as_tensor(np.asarray(b0_image.get_fdata(dtype=np.float32))),  # 三维 b0 强度
    dwi_affine=torch.as_tensor(b0_image.affine, dtype=torch.float32),    # DWI 体素到 RAS 世界仿射
    t1_path="T1w.nii.gz",                                               # 同次 T1w 文件
    device=torch.device("cpu"),                                         # 配准运行设备
)
# transform：DWI world-RAS 到 T1 world-RAS 的 4×4 float32 张量。
```

```bash
flirt -in mean_b0.nii.gz -ref T1w.nii.gz \
  -cost normmi -dof 6 -omat official_flirt.mat
```

本包公开接口可通过 `UKBConnectome(device="cpu", synthseg_weights=None)(dwi=..., bvals=..., bvecs=..., t1=..., atlas_dwi=..., t1_segmentation=..., segmentation_source="synthseg", dwi_to_t1_world=None, n_seeds=100, seed=0)` 触发自动配准。`dwi_to_t1_world=None` 是关键；其他参数分别提供已校正 DWI、梯度、T1、脑区图、分割、权重和追踪播种。此报告**只隔离运行 `_registration`**，没有再次运行该完整调用。

## 真实数据和结果

输入为公开 ds004666 同次 T1w 和 TOPUP/EDDY 校正 AP-DWI 的五个 b0 平均图。b0 是校正 DWI 第 0、21、42、63、84 个体积的实测强度均值，保留原 DWI 仿射，形状 104×104×72，SHA256 `0c1dee4472694a9c43d8355136ae2467d3a90d9f76729371ef95623fc7b1413d`。T1 形状 208×274×245，SHA256 `80ace482fbc8a5f18423e202118e58d8c9e9ac9bbd1fea9f243de3fff2572017`。输入文件不随仓库分发；[校正输入来源和原命令](../connectome/ds004666/corrected_input_provenance.public.json)可核对。

同一 Conda 环境、headcw CPU、8 个 PyTorch 线程，对旧、新 `_registration` 各运行一次。更换前后的 b0/T1 载入体素、仿射、体素大小逐值一致。

| 项目 | 旧版 Surfa 路径 | 新版 NiBabel/Volume 路径 |
|---|---:|---:|
| `_registration` 墙钟时间 | 15.754 s | 14.207 s |
| 输出矩阵 | 4×4、float32、CPU | 与旧版 **16/16 元素逐值相同** |
| `.npy` 文件 SHA256 | `32a318fc7813b281838bd964cb0d3965012c72cbbdbe48db4e2f4d3d3a7f5221` | 相同 |
| Surfa 模块 | 已载入 | 禁止导入，未载入 |

单次共享节点墙钟时间不能推断稳定提速。官方 FSL 6.0.7.4 在**同一张完整头部 b0 和 T1**上耗时 13.43 s。将 FSL `.mat` 换算为 world-RAS 后，在 b0 的 13×13×13 网格点上与 FNIT 矩阵比较，位移差 RMS **1.261 mm**、最大 **2.224 mm**。此前[脑内 b0/T1 的三次验证](../connectome/ds004666/ANATOMY_STAGE_20260927.md)输入经过脑提取，矩阵位移差范围更小；两项实验输入不同，不能互换结论。新旧 FNIT 矩阵完全相同，当前 FSL 差异不是本次移除 Surfa 造成的。

禁用 Surfa 导入后，`fnit connectome --help` 正常返回；针对性 CPU pipeline 测试为 **2 passed、1 deselected**，另有 CLI 测试 **4 passed**；两组都禁用了 Surfa 导入。小样本 pipeline 测试显式传入配准矩阵，检查其余流程接线，不构成自动配准或真实数据全流程基准。未重新运行真实数据 FOD、纤维追踪、SIFT2 和四张连接矩阵，也未进行 GPU 配对计时。既有完整 connectome 数值对照仍见[ds004666 报告](../connectome/ds004666/README.md)。机器可读数值见[`report.json`](report.json)。
