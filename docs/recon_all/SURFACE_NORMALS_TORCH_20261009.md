# 有序顶点法向的 PyTorch 后端

## 功能简介

`TorchFaceNormalTopology` 为固定有序三角网格缓存角点索引，用 PyTorch
计算 `mris_sphere` 与 `mris_place_surface` 使用的顶点法向。每个 CSR 位置
并行计算各顶点，同一顶点的面贡献仍按原面号和角点顺序相加。没有原子归约，
没有全顶点两两距离，也没有 epsilon 钳制；零向量返回零。

当前仅是法向子阶段，完整拓扑 GA、remesh、inflate、自相交修复及 white/pial
顶点更新仍使用已有后端。`run_standard_sphere(normals_device=...)` 提供显式
候选入口，默认 `None` 保留已验证的 Numba 法向。尚未用此候选完成球面整步
或原始 T1 整例回归，不自动改变生产后端。所有依赖已在主页 Conda 环境中声明，无新增依赖。

## Python 调用、输入和输出

```python
import numpy as np
import torch
from fnit.recon_all.place_surface_normals import TorchFaceNormalTopology

normal_topology = TorchFaceNormalTopology(
    triangles=surface_faces,             # (F,3) 有序整数三角面，每项是顶点索引
    nvertices=len(surface_vertices_mm),   # N，包含孤立顶点
    device="cuda:0",                     # 显式目标 GPU；"cpu" 用于诊断
)
vertex_coordinates = torch.as_tensor(
    surface_vertices_mm,                 # (N,3)，surface RAS 坐标，单位 mm
    dtype=torch.float32,                  # 完整 FP32；不使用 FP16/BF16
    device="cuda:0",                     # 必须与拓扑缓存的实际设备相同
)
vertex_normals = normal_topology.evaluate_tensor(
    vertices=vertex_coordinates,         # (N,3) FP32；只读，不改变坐标
)
# vertex_normals 是同设备 (N,3) FP32 张量；单位法向，无长度单位。
normal_array = normal_topology.evaluate(
    vertices=np.asarray(surface_vertices_mm, dtype=np.float32),  # NumPy兼容接口
)
# normal_array 是 (N,3) float32 NumPy；此调用包含 H2D 和 D2H。
```

构造参数 `triangles`、`nvertices` 必填；`device` 默认 `"cuda"`，推荐显式
指定索引。关联面索引缓存占用 `N×最大关联面数` 级别空间，常规皮层网格约
几十 MB；极高价数网格可能需要更多内存。CUDA 不可用或分配失败直接抛错，
不回退 CPU。非法面形状、非整数面、越界索引、错误顶点数抛 `ValueError`。
`evaluate_tensor` 要求 FP32、同设备及同形状；类型错误抛 `TypeError`，
其余契约错误抛 `ValueError`。空网格返回 `(0,3)`；孤立点法向为零。

缓存只保存拓扑。坐标每次重算，不跨 white.preaparc、final white、pial
共享坐标或法向。换有序面必须新建缓存，`validate(triangles=..., nvertices=...)`
可检查一致性。现有 `initial_vertex_normals(..., topology=normal_topology)`
及 `CoordinateNormalCache(topology=normal_topology)` 接受此子类。
本实现不改变 TF32、默认 dtype 或 autocast；点乘/除法/平方根以 FP32 计算。

## 命令行调用

法向是内部算子，没有独立功能 CLI。显式球面候选入口为：

```bash
python -m fnit.recon_all.sphere_standard_run \
  /data/subject/surf/lh.inflated \
  /data/subject/surf/lh.smoothwm \
  /data/diagnostic/lh.sphere.torch-normals \
  --normals-device cuda:0 \
  --averaging-device cpu \
  --finish-device cpu \
  --report /data/diagnostic/lh.sphere.torch-normals.json
```

三个位置参数依次为输入 inflated、输入 smoothwm 和独立诊断输出；二者须
有相同有序面和顶点编号。`--normals-device` 默认不设置，沿 Numba；显式
cpu/cuda 只替换法向。平均和最终清理仍各按已有设备参数运行。
报告增加 `normals_backend` 和 `normals_device`，不改变收敛规则。
该候选整步尚未验收，输出需自行完成同输入及网格质量对照。

## 对应原软件调用

本函数属于 `mris_sphere`/`mris_place_surface` 内部法向计算，没有独立等价命令。
完整 sphere 对照为 `mris_sphere INPUT_INFLATED OUTPUT_SPHERE`，完整表面放置
参数见既有 [white](WHITE_PREAPARC_CONDA_CHAIN.md) 与
[pial](PYTHON_PIAL_PLACEMENT.md) 页面。生产内核不调用 FreeSurfer 命令。

## 本版真实数据精度和时间

2026-10-09 在同一 H100 节点测量公开 ds000114 sub-06/sub-07 的双侧
`orig`、`white.preaparc`、`sphere`：12 张网格。比较基线是本项目已验证的
Numba 有序法向；本轮没有独立执行官方内部法向算子，因此不把该结果称为
对官方或端到端重建的通过。

Numba 与 Torch 均 4 线程，热缓存，3 轮 ABBA/BAAB，共每后端 6 次配对采样；
包含 NumPy API 边界，读入和拓扑构造另列在 JSON。表中单位为 ms。
预先规定最大法向分量误差 `5×10⁻⁷`，12 张均通过，最大实测 `1.79×10⁻⁷`。

| 冻结网格 | 顶点数 | Numba CPU | Torch CPU API | 最大绝对误差 |
|---|---:|---:|---:|---:|
| `ds000114_sub-06/lh.orig` | 130,346 | 186.90 | 34.06 | 1.19e-07 |
| `ds000114_sub-06/lh.white.preaparc` | 130,346 | 152.45 | 31.38 | 1.19e-07 |
| `ds000114_sub-06/lh.sphere` | 130,346 | 138.48 | 29.78 | 1.19e-07 |
| `ds000114_sub-06/rh.orig` | 132,837 | 142.07 | 37.11 | 1.79e-07 |
| `ds000114_sub-06/rh.white.preaparc` | 132,837 | 140.31 | 36.83 | 1.79e-07 |
| `ds000114_sub-06/rh.sphere` | 132,837 | 139.30 | 36.63 | 1.79e-07 |
| `ds000114_sub-07/lh.orig` | 114,342 | 120.77 | 32.11 | 1.19e-07 |
| `ds000114_sub-07/lh.white.preaparc` | 114,342 | 119.95 | 32.80 | 1.79e-07 |
| `ds000114_sub-07/lh.sphere` | 114,342 | 117.64 | 32.18 | 1.79e-07 |
| `ds000114_sub-07/rh.orig` | 114,824 | 122.16 | 26.76 | 1.19e-07 |
| `ds000114_sub-07/rh.white.preaparc` | 114,824 | 120.78 | 26.68 | 1.19e-07 |
| `ds000114_sub-07/rh.sphere` | 114,824 | 121.69 | 26.45 | 1.79e-07 |

CPU 热 API 观测为 26.45–37.11 ms，原 Numba 为 117.64–186.90 ms；这是
法向局部结果，不代表 sphere 或整例加速。结果有浮点尾差，严格逐位复现
未通过；固定法向子算子容差通过。脑区指标及整体指标等效未评估。

CUDA 两个新进程均在首个 120 字节索引分配时报 OOM；同时 GPU0
`nvidia-smi` 显示空闲 77,305 MiB。第三个最小探针在 `cudaMemGetInfo`
阶段即失败。根因未确定；GPU 精度、速度、实际进程显存和干净环境隔离
验收均未完成。未把缓存关闭或 unavailable 记为 0 显存。完整收据和复现脚本：
[本轮验证目录](../../validation/recon_all/optimizations/20261009_ordered_normals/README.md)。

## 最近版本与 benchmark

- 当前候选：替换未接入的原子累加草稿，保留有序累计和极小向量规则；新增
  12 张真实网格配对报告及 CUDA 初始化失败记录。
- 基线：`937263e04eb1be1a5053952ff2b673213cdf966f` 加设备绑定后的候选工作树；实际测试
  源码 SHA-256 绑定在 JSON 中，不能把其标为当前 main 的整例结果。
- 既有法向及缓存证据保留于 [有序法向](SURFACE_NORMALS.md) 和
  [目标函数缓存](SURFACE_NORMAL_OBJECTIVE_CACHE_20261007.md)。

## 参考文献与原实现

- FreeSurfer 固定源码：
  [mrisurf metric properties](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)。
- Dale, Fischl & Sereno. Cortical surface-based analysis. I. Segmentation and
  surface reconstruction. NeuroImage 9, 179–194 (1999).
- Fischl, Sereno & Dale. Cortical surface-based analysis. II: Inflation, flattening,
  and a surface-based coordinate system. NeuroImage 9, 195–207 (1999).
