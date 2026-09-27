# TorchEDDY 源码目录

`TorchEDDY` 对单被试 4D DWI 做基于 volume 的运动和 eddy-current 校正，并读取 TorchTOPUP 输出。计算使用 PyTorch CPU/CUDA，运行时不调用 FSL。

```python
from fnit import TorchEDDY

model = TorchEDDY(
    device="cuda:0",  # 计算设备；可改为 "cpu"
)
result = model.run(
    imain="AP.nii.gz",  # 输入：单被试 4D DWI
    mask="nodif_brain_mask.nii.gz",  # 输入：DWI 网格上的 3D 脑 mask
    topup="fieldmap_out",  # 输入：TOPUP coefficient 和 movement 的根名
    acqp="acqparams.txt",  # 输入：相位编码方向和总读出时间
    index="eddy_index.txt",  # 输入：每个 DWI volume 对应的 acqp 行号
    bvecs="AP.bvec",  # 输入：FSL 3×N 梯度方向
    bvals="AP.bval",  # 输入：与 volume 顺序一致的 b-value
    out="eddy/data",  # 输出：FSL 风格 basename
    ref_scan_no=0,  # 参考 volume 的从 0 开始索引
    overwrite=False,  # 是否覆盖已有输出
)
```

对应的原软件命令：

```bash
eddy_openmp --imain=AP.nii.gz --mask=nodif_brain_mask.nii.gz \
  --topup=fieldmap_out --acqp=acqparams.txt --index=eddy_index.txt \
  --bvecs=AP.bvec --bvals=AP.bval --out=eddy/data --ref_scan_no=0
```

输出包括 `data.nii.gz`、旋转后的 `data.eddy_rotated_bvecs`、参数、运动 RMS、残差和 QC。完整结构见[功能说明](../../../docs/eddy/README.md)。真实数据报告由 FNIT 0.14.0 生成；八项共享核心输出与 FSL 对照所用候选逐文件 SHA-256 相同，报告所列数值源码与当前 0.16.0 一致。精度、耗时、显存和示意图见[功能说明](../../../docs/eddy/README.md)。
