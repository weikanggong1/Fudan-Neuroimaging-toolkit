# FNIT PyTorch FLIRT 模块

模块公开两种配准配置：12 自由度仿射 `corratio`，以及原 UKB 连接组流程使用的 6 自由度刚性 `normmi`。8 mm 角度搜索使用默认相关比；6DOF 的 4/2/1 mm 精化使用 normmi。实现仅在 PyTorch 上运行，不在运行时调用 FSL。

- `core.py`：代价函数、优化、重采样和 `TorchFLIRT`。
- `coordinates.py`：FSL scaled-mm 与 RAS 世界坐标矩阵转换。
- `types.py`：输入验证与 `FLIRTResult` 输出结构。
- `standalone.py`：`run_flirt` 路径 API 和原子写入。
- `cli.py`、`__main__.py`：`fnit-flirt` 命令行入口。

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="b0_brain.nii.gz",     # 移动图像：去脑 3D b0
    reference="T1_brain.nii.gz", # 参考图像：去脑 3D T1，定义输出网格
    output="b0_in_T1.nii.gz",    # 输出：T1 网格上的重采样 b0
    omat="b0_to_T1_fsl.mat",     # 输出：输入→参考的 FSL scaled-mm 4×4 矩阵
    init=None,                    # 可选初始矩阵
    dof=6,                        # 6DOF 刚性变换
    cost="normmi",                # 精化阶段归一化互信息代价
    device="cuda:0",             # PyTorch GPU
    overwrite=False,             # 不覆盖现有文件
)
```

`result.moved` 是参考网格的 `surfa.Volume`；`result.matrix`/`fsl_matrix` 是 FSL scaled-mm 矩阵；`result.moving_to_fixed_world` 和 `fixed_to_moving_world` 是 RAS 世界坐标正反矩阵；`result.qc` 记录配置和验证范围。等价软件命令为 `flirt -in b0_brain.nii.gz -ref T1_brain.nii.gz -cost normmi -dof 6 -out b0_in_T1.nii.gz -omat b0_to_T1_fsl.mat`。当前真实配对 UKB 同输入对照的矩阵平均位移 `0.186655 mm`、最大 `0.367350 mm`；FNIT H100 GPU `19.155 s`/`0.746 GiB`，FSL CPU `10.02 s`。尚未逐值一致。

每个参数及输出结构、12DOF 测试和图像对照见[完整使用说明](../../../docs/flirt/README.md)；原 UKB 配准命令、输入来源、精度及时间见[阶段报告](../../../validation/connectome/ORIGINAL_UKB_FLIRT_STAGE_20260927.md)。源码及修改受[非商业 FSL 许可证](../../../licenses/FSL-6.0.txt)约束。
