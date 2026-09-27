# TorchDTIFIT 源码目录

`TorchDTIFIT` 在 mask 内对 b0 与单个扩散 shell 拟合 diffusion tensor，并输出 FA、MD、L1、L2、L3、MO、V1/V2/V3 和 S0。计算使用 PyTorch CPU/CUDA，运行时不调用 FSL。

```python
from fnit import TorchDTIFIT

model = TorchDTIFIT(
    device="cuda:0",  # 计算设备；可改为 "cpu"
)
result = model.run(
    data="data_b1000.nii.gz",  # 输入：b0 与目标 shell 的 4D DWI
    mask="nodif_brain_mask.nii.gz",  # 输入：同一网格的 3D 脑 mask
    bvecs="data_b1000.bvec",  # 输入：FSL 3×N 梯度方向
    bvals="data_b1000.bval",  # 输入：与 volume 顺序一致的 b-value
    output_prefix="dti",  # 输出：所有参数图的 basename
    save_tensor=False,  # 是否另写六通道 FSL tensor
    overwrite=False,  # 是否覆盖已有输出
)
```

对应的原软件命令：

```bash
dtifit -k data_b1000.nii.gz -m nodif_brain_mask.nii.gz \
  -r data_b1000.bvec -b data_b1000.bval -o dti
```

`result.maps` 以参数名索引 nibabel 图像；写盘文件名和 tensor 六通道顺序见[完整说明](../../../docs/dtifit/README.md)。FNIT 0.14.0 报告记录候选文件 SHA-256；其数值文件与当前 0.16.0 发布逐字节相同，且未记录 CUDA peak allocation。
