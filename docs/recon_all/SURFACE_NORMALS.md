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

2026-09-30 在同一 gpucw1、PyTorch 四线程/Numba 128 线程下，对两例真实 T1 的左右半球 inflated 和 sphere 共八张冻结网格比较旧/新函数。八项法向元素全部一致。首个调用包含 JIT，其余时间单独保留在[逐项报告](../../validation/recon_all/python_gpu_port/performance_20260930/normals_stage_pair_report.json)；报告绑定旧/新源码和每张输入 SHA-256。七项后续调用从 1.327–1.530 s 降至 0.213–0.249 s；不以首次 JIT 调用计算稳定速度比。

同一 sub-01 左侧 inflated/smoothwm 输入的完整 standard sphere 已进一步通过回归：`279e09f` 为 541.307 s，`e036f57` 为 288.842 s，观察耗时减少 46.64%，224 次法向更新后的最终坐标和有序面全部一致。两轮均含 cProfile、JIT、读写和共享主机负载；这是冻结输入阶段比较，不能当作整例提速。对应[旧剖析](../../validation/recon_all/python_gpu_port/performance_20260930/sub01/sphere_profile_279/profile.txt)、[新剖析](../../validation/recon_all/python_gpu_port/performance_20260930/sub01/sphere_profile_e036f57/profile.txt)和[新结果与哈希](../../validation/recon_all/python_gpu_port/performance_20260930/sphere_profile_e036f57_report.json)保留完整证据。法向累计耗时从 337.03 s 降到 65.93 s，距离目标函数目前占 98.12 s；没有据此改写其算法。

sub-01 右侧也完成[完整冻结球面回归](../../validation/recon_all/python_gpu_port/performance_20260930/sub01/sphere_profile_e036f57_rh/report.json)：134 次更新、154.357 s，最终坐标和有序面与 `279e09f` 整例中的同输入球面完全一致。右侧没有同口径的旧 cProfile 时间，因此只报告新耗时与精度，不计算右侧阶段速度比。两张球面报告均记录输入、函数源码、脚本哈希及实际计算提交；[右侧脚本](../../validation/recon_all/python_gpu_port/performance_20260930/sphere_profile_e036f57_rh.py)沿用左侧显式参数和失败条件。

sub-02 在同一 nodecw10、相同线程策略下从原始 T1 和空目录完成 66 阶段、138 项输出及双侧网格检查。[本次整例](../../validation/recon_all/python_gpu_port/performance_20260930/sub02/full_e036f57_run.json)为 5756.04 s，直接基线为 6099.80 s，本次观察缩短 5.64%；外层命令为 5764.93 s。左右 surface 阶段从 1013.69/944.25 s 降至 828.81/796.28 s。[完整比较](../../validation/recon_all/python_gpu_port/performance_20260930/sub02/full_e036f57_pair/summary.json)对基线 138/138 通过，七张分割图各标签 Dice=1、white/pial 同索引距离为 0、68/45/70 区统计差为 0。对官方仍为 2/138，历史数值差异没有被修正或隐藏；总体等效仍未判定。共享硬件单次整例不是稳定吞吐证明。

sub-01 的[已初始化 CUDA API 整例](../../validation/recon_all/python_gpu_port/performance_20260930/sub01/full_e036f57_run.json)同样完成全部输出与网格检查，[对直接基线 138/138](../../validation/recon_all/python_gpu_port/performance_20260930/sub01/full_e036f57_pair/summary.json)，分区、表面和脑区统计没有改变。双侧 surface 从 1117.56/935.27 s 降到 832.94/735.07 s，合计减少 484.83 s；但完整函数为 7422.34 s，相对直接基线慢 17.74%。WM 编辑、wm_pretess 与 MNI 等前段的新增耗时仍计入总时间；这些阶段不执行本次修改的面关联索引，延迟原因尚未确认。详见[整例实测和诊断](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)，未以局部提速宣称 GPU 整例加速。

[固定路径复现脚本](../../validation/recon_all/python_gpu_port/performance_20260930/benchmark_normals_pair.py)读取 `full_sub01_279e09f` / `full_sub02_279e09f` 的八张网格，调用旧函数及当前函数，写出差异数、最大差、耗时和哈希；任何元素不同会失败。它仅用于这套隔离实验，不将官方表面作为生产输入。

## 原实现与参考文献

- [固定版本 mris_sphere](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_sphere/mris_sphere.cpp)，法向属于其内部几何计算。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
