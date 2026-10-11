# 球面图谱到顶点图：自有 Python 接口

## 1．功能

`run_paint_atlas()` 在被试的 `sphere.reg` 上采样固定曲率图谱，按输入网格的一阶邻接执行五轮有序平均，写出 `avg_curv`。标准配方使用从零开始的第 6 帧；`-a 5` 表示五轮平均，不能把它解释为第 5 帧或五个最近邻。CPU 路径采用 Numba 浮点运算，平均复用 FNIT 已有实现；CUDA 路径复用现有 PyTorch 采样和平均函数。运行时不调用原软件程序。

本接口覆盖标准 `avg_curv` 配方，不包含原命令的 `normalize`、`variance` 或 `coords` 分支。

## 2．Python 调用、输入与输出

```python
from pathlib import Path
from fnit.recon_all.mrisp_paint_python import run_paint_atlas

paint_report = run_paint_atlas(
    atlas_path=Path("/data/assets/average/lh.folding.atlas.tif"),  # 已校验的左侧固定图谱
    sphere_path=Path("/data/subjects/sub01/surf/lh.sphere.reg"),  # 被试注册球面，surface RAS，毫米
    output_path=Path("/data/subjects/sub01/surf/lh.avg_curv"),  # 相同顶点顺序的 float32 morph 图
    frame=6,  # TIFF 帧号，从零开始，默认 6
    averages=5,  # 完整一阶邻接平均轮数，非负整数，默认 5
    device="cpu",  # 当前四侧精确回归设备；CUDA 须写明如 cuda:0
    report_path=Path("/data/subjects/sub01/scripts/lh.paint.json"),  # 可选 JSON 报告，默认 None
)
```

输入 TIFF 的每帧为 `(azimuth, polar)` 二维图；整型 32 位像素按 float32 位型解释，浮点图直接转换为 float32。球面文件包含 `(N,3)` 有限坐标和 `(F,3)` 有序整数三角面。坐标单位为 surface RAS 毫米；临时居中遵循包围盒中心距离 0.1 mm 的原规则，不修改源网格。

输出为 `(N,)` float32 FreeSurfer morph 格式，与输入顶点逐个对应；写出面数 `F`，无标签重新编号或体积空间变换。函数返回字典，包括三个路径、`frame_zero_based`、`averages`、设备与后端、顶点/面数、dtype、输入/输出及源码 SHA-256、读取/计算/写入/总秒数和 `native_program_called=False`。不设置报告路径时仍返回此字典。

缺失文件、帧号越界、非有限值、非法形状/索引、非整数轮数及 CUDA 错误会抛异常；不复制参考结果或静默切换设备。库调用保持调用方的 TF32 设置，不启用 FP16/BF16。

## 3．命令行调用

```bash
python -m fnit.recon_all.mrisp_paint_python \
  --atlas-path /data/assets/average/lh.folding.atlas.tif \
  --sphere-path /data/subjects/sub01/surf/lh.sphere.reg \
  --output-path /data/subjects/sub01/surf/lh.avg_curv \
  --frame 6 --averages 5 --device cpu \
  --report-path /data/subjects/sub01/scripts/lh.paint.json
```

参数含义与 Python 接口相同；调用者应先设置固定的线程预算。

## 4．原软件参考调用

```bash
mrisp_paint -a 5 lh.folding.atlas.tif#6 lh.sphere.reg lh.avg_curv
```

原程序只在独立 benchmark 中生成参考。此次固定源码为 FreeSurfer 提交 `d932c45b7941662ea380a05efef580568b98d41a`，参考程序 SHA 和实际命令完整保留在机器可读收据中。

## 5．当前真实数据精度与耗时

2026-10-10 使用公开 ds000114 两例的 FNIT 自产 `sphere.reg`，在同一 A100 节点、CPU 8–11、四线程上运行完整文件 API。候选来源为 `54f681dc` 加收据绑定的模块；不是原始 T1 整例测试。原程序每个输入重复两次，四侧重复结果均逐元素一致。

| 输入 | 顶点数 | 候选不同元素 / 最大误差 / P99 | morph 字节 | FNIT 含 I/O 秒 | 原生首次 / 重复秒 |
|---|---:|---|---|---:|---:|
| sub06 左 | 130,679 | 0 / 0 / 0 | 相同 | 5.865 | 0.612 / 0.568 |
| sub06 右 | 132,459 | 0 / 0 / 0 | 相同 | 5.660 | 0.629 / 0.561 |
| sub07 左 | 114,247 | 0 / 0 / 0 | 相同 | 5.089 | 0.640 / 0.511 |
| sub07 右 | 114,951 | 0 / 0 / 0 | 相同 | 4.937 | 0.544 / 0.500 |

此表是实际共享负载观察，包含文件读写和首次 Numba 编译；不代表独占吞吐，也没有整例提速结论。CPU 测试不占用 CUDA；CUDA 四侧与原生的本次完整比较尚未完成，不能将 CPU 结论移用到 CUDA。

完整报告见 [公开收据](../../validation/recon_all/optimizations/20261010_paint_intersection/python_paint/report.json)。严格复现通过的是固定 `avg_curv` 阶段；整体指标等效仍未判定。

## 6．更新与 benchmark 记录

最初复用通用球面采样时，四侧最大误差约 `2.68e-6`–`4.29e-6`。定位到 C++ float 参数的 `sqrt`/`atan2` 重载：先计算 double 再写回 float32 与 float 运算结果不同。CPU Numba 改用 float32 ufunc，保留原半径、插值及平均顺序后，四侧误差全部为零。该修复未改 atlas 帧、平均次数、输入网格或验收门。

首轮 benchmark 的参考输出未使用半球前缀，原程序自动补前缀导致比较器找不到文件；仅修正诊断文件命名，失败记录保留。旧报告作为定位证据保留在本次目录，不能当作新代码结果。

## 7．参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [固定 mrisp_paint 命令](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mrisp_paint/mrisp_paint.cpp)。
- [固定参数化采样代码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisp.cpp)；[有序曲率平均](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_vals.cpp)。
