# TorchFAST

本目录实现已去脑单帧 T1 的三组织 HMRF-EM 分割、乘性偏置场校正和分数体积估计。计算使用 PyTorch；文件读写使用仓库的 NiBabel 影像类，不在运行时导入 Surfa、FSL 或 FreeSurfer。

```python
from fnit.fast import TorchFAST

model = TorchFAST(device="cuda:0")  # 选择 PyTorch 设备；CPU 可改为 "cpu"
result = model(image="T1_brain.nii.gz", mask=None)  # 输入已去脑 T1；None 表示使用 T1 > 0
result.pve_gm.save("T1_brain_pve_1.nii.gz")  # 输出 GM 分数体积
```

返回值还包含 CSF/WM 分数体积、硬分类、PVE 分类、混合组织类型、偏置场、校正图及组织均值/方差。参数、每张输出和 FSL 原生命令见[完整中文说明](../../../docs/fast/README.md)，真实数据验收见[验证记录](../../../validation/fast/no_surfa_20260928/README.md)。

FAST4 2111.3 的未改动源码放在 `upstream_fast4/`，仅用于许可与溯源，不编译或导入。该部分及本改写受 FSL 6.0 非商业许可约束。
