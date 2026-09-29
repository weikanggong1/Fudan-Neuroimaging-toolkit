# 图谱平均曲率与曲率统计

[返回阶段索引](CONDA_CPP_STAGES.md) · [单 T1 输出结构](README.md)

标准流程先在每侧已配准球面上映射 folding atlas 的第 6 幅图，再对 `smoothwm` 计算曲率附图及统计。两个原生程序都从固定源码在 FNIT Conda 环境中编译；下面的官方命令只用于说明参数对应关系。

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

## 曲率附图：`_run_curvature_stats(...)`

输入 `binary` 为 FNIT Conda 中的 `mris_curvature_stats`；`subject` 为被试目录；`hemi` 为 `lh` 或 `rh`；`assets` 为同一资产目录。函数无返回值；读取 `surf/H.smoothwm` 三角面及 `surf/H.curv`、`surf/H.sulc` 顶点图，生成 `stats/H.curv.stats` 文本和 `surf/H.smoothwm.{BE,C,FI,S}.crv` float32 顶点图。附图的顶点序号与 `smoothwm` 一致，曲率单位依各个指标定义；不能作为体素图读取。

```python
from pathlib import Path
from fnit.recon_all.native_free import _run_curvature_stats

_run_curvature_stats(
    binary=Path("/opt/conda/envs/fnit/bin/mris_curvature_stats"),  # FNIT 自编译程序
    subject=Path("/data/subjects/sub01"),  # 已有 smoothwm、curv、sulc 的目录
    hemi="lh",  # 左半球；右侧使用 "rh"
    assets=Path("/data/fnit-assets"),  # 已校验的 FNIT 资产目录
)
```

对应官方命令，在被试的 `scripts/` 目录执行：

```bash
mris_curvature_stats -m --writeCurvatureFiles -G \
  -o ../stats/lh.curv.stats -F smoothwm sub01 lh curv sulc
```

## 真实 T1 的同输入核对

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
