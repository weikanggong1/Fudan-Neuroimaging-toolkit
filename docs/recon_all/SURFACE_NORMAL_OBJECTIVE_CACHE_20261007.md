# white/pial 法向与目标函数缓存（2026-10-07）

## 功能

`place_pial_t1()` 和 `place_white_preaparc_prefix()` 在同一优化循环中会先用
当前坐标计算梯度，再用相同坐标计算目标函数。两处都需要同一有序三角网格的
顶点法向。`CoordinateNormalCache` 按 **NumPy 坐标数组对象身份**保留最近一次
float32 法向，使目标函数和梯度共享一次 `FaceNormalTopology.evaluate()`。
缓存不跨半球、white.preaparc、final white 或 pial 网格共享。

## 输入与输出

- 输入：`FaceNormalTopology`（固定的 `(F, 3)` 有序整数面）和 `(N, 3)` 的
  `float32` surface RAS/mm 坐标数组。
- 输出：`(N, 3)` `float32` 单位法向；零度顶点按原实现返回零向量。
- `place_pial_t1()` 的目标函数缓存同时保留 `(sse, rms)`，只对同一坐标对象
  生效；每次 pial pass 更新目标强度、平滑尺度和原始面积后显式失效。
- 坐标若被原地修改，调用 `cache.clear()` 后再取法向。优化器内部将坐标视为
  只读并在接受新试步时创建新数组，因此不会改变原有更新顺序。

## Python 调用

```python
from fnit.recon_all.place_surface_normals import (
    CoordinateNormalCache, FaceNormalTopology,
)

topology = FaceNormalTopology(faces, len(vertices))
normal_cache = CoordinateNormalCache(topology)
normals_for_gradient = normal_cache.evaluate(vertices)  # 同对象后续调用复用
normals_for_objective = normal_cache.evaluate(vertices)
assert normals_for_gradient is normals_for_objective
```

## 命令行与原软件对应

该缓存属于 `mris_place_surface --white`/`--pial` 的内部计算，没有独立命令。
完整流程仍通过 `fnit-recon-all` 调度；FreeSurfer 对应阶段为
`mris_place_surface --white`、`mris_place_surface --pial`。缓存不改变采样、碰撞、
步长接受、法向累加顺序或输出坐标。

## 验证

`tests/recon_all/test_sphere_serial_optimization.py` 验证同一坐标对象只返回同一
缓存结果、换坐标对象重新计算、显式 `clear()` 后原地修改结果与原始
`initial_vertex_normals(..., topology=...)` 逐元素一致。由于本次仅复用同一
float32 结果，没有放宽严格数值比较，也未启用半精度。
