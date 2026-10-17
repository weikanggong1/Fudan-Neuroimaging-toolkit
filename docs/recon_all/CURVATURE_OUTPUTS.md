# 图谱平均曲率与曲率统计

[返回阶段索引](CONDA_CPP_STAGES.md) · [单 T1 输出结构](README.md)

标准流程先在每侧已配准球面上映射 folding atlas 的第 6 幅图，再对 `smoothwm` 计算曲率附图及统计。曲率统计已提供完整自有Python/Numba/PyTorch入口；显式原生表面配方使用固定源码在 Conda 内独立构建的程序，不调用系统安装的软件。

## 早期主曲率图：`_write_principal_curvature_maps(...)`

输入 `surf` 是自产表面目录，`hemi` 为 `lh/rh`，`device` 为 PyTorch
设备。函数读取同侧 `white.preaparc` 和 `inflated` 三角表面，坐标采用
surface RAS/mm；写出 `H.white.preaparc.{H,K}` 和 `H.inflated.{H,K}`。
这些文件为沿原顶点顺序的 float32 `(N,)` morph，均值曲率 H 的单位为
mm⁻¹，高斯曲率 K 的单位为 mm⁻²。几何和有序面保持不变。

`include_smoothwm=True` 默认保留兼容行为：另读取 `H.smoothwm`，生成
`H.smoothwm.{H,K,K1,K2}.crv`，K1/K2 单位为 mm⁻¹。设置 False 时不读、
不计算、不写出该表面的四图，也不删除已有文件。无返回值；输入、
设备、计算或写出失败直接抛异常。这是 pipeline 内部函数，无独立 CLI。

```python
from pathlib import Path
from fnit.recon_all.native_free import _write_principal_curvature_maps

_write_principal_curvature_maps(
    surf=Path("/data/subjects/sub01/surf"),  # 自产 preaparc/inflated 表面目录
    hemi="lh",                             # 同侧有序顶点；右侧使用 rh
    device="cuda:0",                       # 明确目标 GPU，沿用 float32/TF32
    include_smoothwm=False,                 # 原生配方将四图留给最终完整统计
)
```

生产调度在 `surface_placement_backend="native"` 时采用 False；Python
放置路径保留 True。球面配准、GCSA 注释及脑区统计均从当前几何计算自身
特征，不消费早期 smoothwm 四图。完整 138 项输出仍在最终曲率统计后检查。
早期连续二次面拟合与最终离散曲率定义不同，不能缓存前者替代后者。
white.preaparc/inflated 的 H/K 继续由原有 PyTorch 函数生成。

半球报告 `early_curvature_maps` 保存实际早期生产函数、设备、已生成的
相对文件列表、`include_smoothwm`、含读写秒数，以及最终 smoothwm
曲率图对应的统计阶段和生产者。后两个字段声明后续生产路径，实际完成
状态由对应 `stats_H_curv` 阶段及完整输出门确认。早期秒数已经包含在
surface 半球墙钟内，不能再加到总耗时；生产路径不额外插入 CUDA 同步。

2026-10-17 已在冻结提交 `71cc91899bdf9713690cddbe32054386d8f58d63`
完成两例四侧同输入 A/B/B/A。A 保留三表面拟合，B 延迟 smoothwm
四图；下表为含早期计算、相同自编译完整统计及读写的 API 中位数。
两模式共用进程，首次初始化成本和缓存边界见[阶段报告](../../validation/recon_all/optimizations/20261017_curvature_schedule/README.md)。

| 公开 T1 | 半球 | A / 秒 | B / 秒 |
| --- | --- | ---: | ---: |
| ds000114 sub-06 | LH | 3.3505 | 2.6613 |
| ds000114 sub-06 | RH | 3.1178 | 2.7380 |
| ds000114 sub-07 | LH | 2.7465 | 2.4061 |
| ds000114 sub-07 | RH | 2.6944 | 2.3592 |

最终 smoothwm 八图和完整 stats 均精确一致；保留的 preaparc/inflated
H/K 严格比较失败，原报告保持 `comparison_failed` / EXIT 2。同一 A
模式重复也不同。固定 sub-06 LH 几何后，首个差异位于原有 CUDA 法线
`index_add_` 累加（最大 1.7881×10⁻⁷）；固定同一法线后，两表面的
主曲率拟合各四次精确一致。没有放宽门或为此改变 GPU 实现。

静态生产调用及所选原生命令源码审查未发现这些早期 H/K 的下游重建
消费者；注册、GCSA、脑区统计按几何计算自身特征，最终八图独立生成。
这是阶段重复性定位和静态依赖审查，不是动态文件访问或整例误差证明。
小型机器摘要见[重复性记录](../../validation/recon_all/optimizations/20261017_curvature_schedule/repeatability.public.json)。
本轮两例原始T1的四次配对与四项评估已完成，见[完整结果](../../validation/recon_all/optimizations/20261017_sphere_normals/WHOLE_RESULTS.md)：
CLI观察缩短4.10%/3.02%，同期已报脑区量无变化；各44张局部图仍有20张
零容差失败，完整自有同时显存仍未知。没有将阶段尾差改成严格通过。
独立契约测试见
[`test_principal_curvature_schedule.py`](../../tests/recon_all/test_principal_curvature_schedule.py)。

## `avg_curv`：`_run_avg_curv(...)`

输入 `binary` 为当前 Conda 环境中的 `mrisp_paint`；`subject` 为被试目录；`hemi` 是 `lh` 或 `rh`；`atlas` 为同侧 `average/H.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif`；`assets` 为 FNIT 模板目录，设置子进程的 `FREESURFER_HOME`。`surf/H.sphere.reg` 是被试的有序球面网格，坐标为球面坐标，顶点顺序对应同侧 `white`。函数无返回值；生成 `surf/H.avg_curv`，为与球面顶点顺序一致的 float32 曲率标量。

```python
from pathlib import Path
from fnit.recon_all.native_free import _run_avg_curv

_run_avg_curv(
    binary=Path("/opt/conda/envs/fnit/bin/mrisp_paint"),  # FNIT 自编译程序
    subject=Path("/data/subjects/sub01"),  # 已生成 H.sphere.reg 的被试目录
    hemi="lh",  # 左半球；右侧使用 "rh"
    atlas=Path("/data/fnit-assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"),  # 同侧图谱
    assets=Path("/data/fnit-assets"),  # 已校验的 FNIT 资产目录
)
```

对应官方命令：

```bash
mrisp_paint -a 5 average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 \
  surf/lh.sphere.reg surf/lh.avg_curv
```

## 曲率附图与完整统计：`write_curvature_stats(...)`

[完整七节说明与本版benchmark](FULL_CURVATURE_STATS_PYTHON_20261010.md)。
输入为同侧自产 `surf/H.smoothwm` 三角网格及 `surf/H.curv`、`surf/H.sulc`
FP32顶点图，全部沿原顶点顺序；网格坐标为surface RAS/mm。输出完整
`stats/H.curv.stats` 和 `surf/H.smoothwm.{K,H,K1,K2,S,C,BE,FI}.crv`。
函数不调用原版程序，不需要模板、权重或许可证。

```python
from fnit.recon_all.curvature_stats_python import write_curvature_stats

curvature_report = write_curvature_stats(
    subject_dir="/data/subjects/sub01",  # 自产smoothwm/curv/sulc被试目录
    hemisphere="lh",                    # 左侧；右侧为rh
    device="cuda:0",                    # 明确目标GPU；cpu也是完整自有路径
    angle_backend="source_libm",        # 固定源acosf数学步骤及CPU sqrtf
    averages=0,                         # 标准流程不额外平均原始纹理
    overwrite=True,                     # 晚阶段按流程语义重写早期曲率输出
)
```

返回路径清单、网格规模、全部统计及含检查/传输/读写的总墙钟。
两例四侧完整同输入回归中，八图与完整统计文件均逐字节一致。
全参数、单位、限制与失败行为见完整说明。显式原生表面配方由
`_run_curvature_stats(binary=..., subject=..., hemi=..., assets=...,
device="cpu", python_only=False)` 调用同一固定命令；binary 必须是已校验的
独立 Conda 构建程序，assets 为声明资产目录。该路径返回 None，要求
`stats/H.curv.stats` 及八张 smoothwm 图全部存在且非空，否则抛
FileNotFoundError；失败不会由早期近似四图补齐。`python_only=True`
则使用上面的完整自有入口并返回其报告，不执行程序。

### 固定参考命令

```bash
mris_curvature_stats -m --writeCurvatureFiles -G \
  -o ../stats/lh.curv.stats -F smoothwm sub01 lh curv sulc
```

## 历史同输入记录（2026-09-29）

使用 2026-09-24 官方 `a_official` 的冻结左侧球面、`smoothwm` 和顶点图作输入，FNIT 使用 2026-09-29 新主页 Conda 环境编译的程序。对照的是已有官方结果文件；FNIT 标准运行路径不调用预装程序。

| 输出 | 同索引相关系数 | 最大绝对差 | FNIT 实测耗时 | 官方耗时来源 |
| --- | ---: | ---: | ---: | --- |
| `lh.avg_curv`，106,622 顶点 | 0.99999999999977 | 2.44×10⁻⁶ | 1.31 s | 旧运行日志未给单命令计时 |
| `lh.smoothwm.BE.crv` | 1.0 | 2.44×10⁻⁴ | 见下行 | 见下行 |
| `lh.smoothwm.C.crv` | 0.9999999999999992 | 1.91×10⁻⁶ | 见下行 | 见下行 |
| `lh.smoothwm.FI.crv` | 0.9999999999999998 | 0.0078125 | 见下行 | 见下行 |
| `lh.smoothwm.S.crv` | 0.9999999999999979 | 0.015625 | 曲率统计合计 1.28 s | 旧官方日志 3.42 s |

这些是共享服务器上不同时间的单次观察值，不能作为严格速度比。官方 `avg_curv` 单命令耗时缺失；原始运行日志仅保存了命令和结果。冻结官方上游的吻合不等于 FNIT 自产球面后的整例吻合。文档数值对应 2026-09-29 的本地配对运行。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
