# TorchTOPUP 源码目录

`TorchTOPUP` 用 PyTorch 复现 FSL 6.0.7.4 `b02b0.cnf` 的 AP/PA 两帧 b0 路径。它写出 FSL 角色对应的 coefficient、movement、Hz field、校正 b0 和 Jacobian 文件；运行时不调用 FSL。

```python
from fnit import TorchTOPUP

model = TorchTOPUP(
    device="cuda:0",  # 计算设备；可改为 "cpu"
)
result = model.run(
    imain="B0_AP_PA.nii.gz",  # 输入：两帧相反相位编码 b0
    datain="acqparams.txt",  # 输入：每帧的相位编码方向和总读出时间
    out="fieldmap_out",  # 输出根名：coefficient 与 movement 文件
    fout="fieldmap_fout",  # 输出根名：Hz 场图
    iout="fieldmap_iout",  # 输出根名：两帧校正图
    jacout="fieldmap_jacout",  # 输出根名：两帧 Jacobian
    overwrite=False,  # 是否覆盖已有输出
)
```

对应的原软件命令：

```bash
topup --imain=B0_AP_PA.nii.gz --datain=acqparams.txt \
  --config=b02b0.cnf --out=fieldmap_out \
  --fout=fieldmap_fout --iout=fieldmap_iout --jacout=fieldmap_jacout
```

`imain`、`datain` 与六类输出的 shape、坐标和文件结构见[完整说明](../../../docs/topup/README.md)。FNIT 0.14.0 报告记录候选文件 SHA-256；其数值文件与当前 0.16.0 发布逐字节相同，病例数为一。
