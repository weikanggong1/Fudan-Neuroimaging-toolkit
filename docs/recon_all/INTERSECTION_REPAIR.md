# 表面自相交修复

[返回阶段索引](CONDA_CPP_STAGES.md)

`remove_intersection_surface(...)` 检查两个不共享顶点的三角面是否相交。零相交时保留原文件；非零时调用当前 Conda 环境从固定源码编译的 `mris_remove_intersection` 迭代修复，并重新检查结果，仍有相交则报错。标准流程在拓扑修复和 remesh 后，对每侧 `orig` 执行此步骤。

`input_path` 是 FreeSurfer 三角表面路径，坐标为被试 surface RAS；`output_path` 为修复后的表面路径，可与输入相同；`binary` 是 FNIT 自编译程序路径，未给时从当前 Python 环境的 `bin/` 查找；`assets_dir` 指 FNIT 数据资产目录，设置原生子进程的 `FREESURFER_HOME`。返回 `(相交面数, 被标记顶点数)` 两个整数。输出保留三角面结构和坐标约定；若迭代修复仍有相交，抛出 `RuntimeError`。

```python
from pathlib import Path
from fnit.recon_all.mris_remove_intersection_python import remove_intersection_surface

intersecting_faces, marked_vertices = remove_intersection_surface(
    input_path=Path("/data/subjects/sub01/surf/lh.orig"),  # 拓扑修复后的左侧原始表面
    output_path=Path("/data/subjects/sub01/surf/lh.orig"),  # 原位写回修复结果
    binary=Path("/opt/conda/envs/fnit/bin/mris_remove_intersection"),  # FNIT 自编译程序
    assets_dir=Path("/data/fnit-assets"),  # 已校验的 FNIT 资产目录
)
# 两个返回值分别为输入相交三角面数和受影响顶点数。
```

对应官方命令为 `mris_remove_intersection surf/lh.orig surf/lh.orig`。FNIT 运行时只调用自身 Conda 中的编译产物。

当前真实 T1 连续运行的最终 `orig` 为零相交。2026-09-29 FNIT 的检测与保留耗时为左侧 1.21 秒、右侧 1.41 秒；2026-09-24 官方日志中的 `mris_remove_intersection` 分别为 4.89 秒和 4.61 秒。两次使用各自产生的 `orig`，不能计算配对速度比。

同一真实 T1 自产的修复后、remesh 前 `lh.orig.premesh` 自然出现 7 个相交面。隔离调用此函数后，独立复检为 **0 个相交面**；100,676 个顶点和 201,348 个有序面保持不变，19 个顶点坐标改变，最大位移 2.175 mm。[输入、程序和输出哈希](../../validation/recon_all/python_gpu_port/intersection_positive_real_20260929.json)可复查。这验证了非零分支确实执行并复检，但当前没有官方对同一非零输入的配对输出或耗时，不能据此声称局部修复位移与官方一致。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
