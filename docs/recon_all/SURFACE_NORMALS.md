# 表面法向与面关联索引

[重建入口](README.md) · [当前性能记录](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)

标准球面每轮需要重新计算法向。真实左半球的函数剖析显示，法向调用累计占约 62% 的阶段时间，其中大部分耗时是 Python 循环反复构造面关联索引。本轮以稳定排序替换该循环，继续复用已有 Numba 单精度法向内核。每个顶点的面、角点及累计顺序保持原样，没有改变法向定义或精度，也不新增依赖。

## 输入、输出及限制

`initial_vertex_normals(vertices, triangles) -> numpy.ndarray` 是内部几何步骤，没有独立官方 CLI；它用于 `mris_sphere`、`mris_place_surface` 对应流程。两个参数均必填：

| 参数或返回值 | 结构与含义 |
| --- | --- |
| `vertices` | `(N, 3)` 顶点坐标，转换为 float32；重建使用 surface RAS mm，单位缩放不影响单位法向。 |
| `triangles` | `(F, 3)` 有序三角面顶点索引，转换为 int32；须为合法索引，行顺序参与法向累计。 |
| 返回数组 | `(N, 3)` float32 单位法向，无关联面或零法向的顶点为零向量；不写文件或修改输入。 |

非法形状或索引沿用 NumPy/Numba 的错误行为；函数不能用于四边形索引。排序仅构造关联索引，原始面不重排。数值内核留在 CPU，索引和法向随后供已有球面优化器使用，没有新增 CPU/GPU 传输。

```python
import numpy as np
from nibabel.freesurfer.io import read_geometry
from fnit.recon_all.place_surface_normals import initial_vertex_normals

vertices, triangles = read_geometry("/data/subjects/sub01/surf/lh.inflated")
normals = initial_vertex_normals(
    vertices=np.asarray(vertices, dtype=np.float32),  # surface RAS mm 的顶点坐标
    triangles=np.asarray(triangles, dtype=np.int32),  # 保留文件中的有序三角面
)
# normals 为与输入顶点同序的 (N, 3) float32 单位法向。
```

## 真实同输入回归

2026-09-30 在同一 gpucw1、PyTorch 四线程/Numba 128 线程下，对两例真实 T1 的左右半球 inflated 和 sphere 共八张冻结网格比较旧/新函数。八项法向元素全部一致。首个调用包含 JIT，其余时间单独保留在[逐项报告](../../validation/recon_all/python_gpu_port/performance_20260930/normals_stage_pair_report.json)；报告绑定旧/新源码和每张输入 SHA-256。完整球面和原始 T1 整例须继续验证，不能把这项索引优化的局部加速当作整例提速。

[固定路径复现脚本](../../validation/recon_all/python_gpu_port/performance_20260930/benchmark_normals_pair.py)读取 `full_sub01_279e09f` / `full_sub02_279e09f` 的八张网格，调用旧函数及当前函数，写出差异数、最大差、耗时和哈希；任何元素不同会失败。它仅用于这套隔离实验，不将官方表面作为生产输入。

## 原实现与参考文献

- [固定版本 mris_sphere](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_sphere/mris_sphere.cpp)，法向属于其内部几何计算。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
