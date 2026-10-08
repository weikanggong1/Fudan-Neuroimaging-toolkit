# 表面几何的 PyTorch GPU 内核

## 功能

`fnit.recon_all.torch_surface_kernel` 提供不改变网格拓扑的三角面几何计算：

* `face_geometry`：面面积、单位法向量和面中心；
* `vertex_area_normals`：按三角面累加顶点面积和面积加权法向量；
* `edge_lengths`：每个三角面的三条边长。

这些运算是 white、pial、sphere 和 remesh 阶段中可以独立缓存的几何基础量。输入顶点坐标在同一个 surface RAS 空间，单位为 mm；面索引是 `(F, 3)` 的整数张量。函数支持 `(V, 3)`/`(F, 3)` 和批量 `(B, V, 3)`/共享 `(F, 3)` 的输入，并在 `device="cuda"` 时使用 PyTorch GPU 张量。

## Python 调用

```python
import torch
from fnit.recon_all.torch_surface_kernel import face_geometry, vertex_area_normals

vertices = torch.as_tensor(surface_vertices_mm, dtype=torch.float32, device="cuda")  # (V, 3)，surface RAS，mm
faces = torch.as_tensor(surface_faces, dtype=torch.int64, device="cuda")  # (F, 3)，有序三角面索引
areas_mm2, normals, centers_mm = face_geometry(vertices, faces, device=vertices.device)  # 面基础量
vertex_area_mm2, vertex_normals = vertex_area_normals(vertices, faces, device=vertices.device)  # 顶点基础量
```

返回值只含几何基础量，不写文件，不改变输入网格，也不执行拓扑修复、重网格、white/pial 顶点更新或相交消除。零面积面返回零法向量，不产生 NaN。需要写 FreeSurfer surface、curvature 或 annotation 的调用方仍应使用现有 nibabel/格式 I/O。

## 为什么暂不替换完整 C++ 阶段

`mris_fix_topology`、`mris_remesh`、`mris_place_surface` 和 `mris_inflate` 包含顺序更新、拓扑改变、碰撞处理或收敛判定；仅把其中的向量运算改成 GPU，不能证明算法等价。因此本内核当前是可验证的 GPU 几何基础层，不改变 recon-all 默认后端。完成同输入阶段回归、跨半球和跨被试表面距离验证后，才可以把独立的指标阶段切换到它。

## 对应原软件和验证

原 FreeSurfer 阶段：`mris_anatomical_stats`、`mris_curvature` 及 white/pial 内部的三角面几何计算。该模块不调用 FreeSurfer 可执行文件。CPU 与 CUDA 的单元回归比较面积、法向量、中心和边长；完整 recon-all 的严格复现、表面距离和脑区指标仍需在真实 T1 上单独报告。

