# PyTorch FLIRT 模块

本目录实现 12 DOF / corratio 和 6 DOF / normmi 线性配准。`core.py` 负责 PyTorch 配准，`coordinates.py` 处理 FSL scaled-mm 与 world-RAS 坐标，`types.py` 管理图像输入和结果，`standalone.py` 提供文件接口，`cli.py` 提供命令行。路径输入经 NiBabel 读取；包内不导入 Surfa，也不调用 FSL。

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="input_T1.nii.gz",             # 待移动的单帧三维图像
    reference="reference_T1.nii.gz",   # 固定图像及输出网格
    output="registered_T1.nii.gz",     # 输出重采样图像
    omat="input_to_reference.mat",     # 输出 FSL scaled-mm 坐标矩阵
    init=None,                          # 初始 FSL 矩阵；None 为单位矩阵
    inweight=None,                      # 输入图像权重
    refweight=None,                     # 参考图像权重
    dof=12,                             # 12 自由度仿射配准
    cost="corratio",                    # 相关比代价函数
    device="cuda:0",                    # 运行设备；可改为 cpu
    overwrite=False,                    # 不覆盖已有文件
)
# result.moved：参考网格图像；result.matrix：输入到参考的 FSL 4×4 矩阵；result.qc：运行记录。
```

对应命令 `fnit-flirt -in input_T1.nii.gz -ref reference_T1.nii.gz -out registered_T1.nii.gz -omat input_to_reference.mat -dof 12 -cost corratio --device cuda:0`；官方对照命令是 `flirt -in input_T1.nii.gz -ref reference_T1.nii.gz -out registered_T1.nii.gz -omat input_to_reference.mat -dof 12 -cost corratio`。这两条命令的输入、参考、图像输出和矩阵输出含义相同。

函数参数、内存图像接口、结果字段、坐标说明、真实数据的精度与时间对照见[完整中文文档](../../../docs/flirt/README.md)。2026-09-28 移除 Surfa 的两种配置同输入验证与限制见[迁移报告](../../../validation/flirt_no_surfa_20260928/README.md)。旧 10 例 FSL 对照属于迁移前历史记录，尚未用新路径逐例重测。

6 DOF / normmi 模式的 8 mm 搜索使用默认 corratio，4/2/1 mm 精化使用 normmi。原 UKB 配对 b0→T1 的迁移前同输入对照：平均矩阵位移 `0.186655 mm`，H100 求解及重采样 `19.155 s` / `0.746 GiB`，FSL CPU 完整命令 `10.02 s`；阶段诊断与迁移后待复测事项见[原 UKB 报告](../../../validation/connectome/ORIGINAL_UKB_FLIRT_STAGE_20260927.md)。

本移植及随附上游源码适用 FSL Software Licence Release 6.0（非商业用途）。
