# 多图谱统计复用

同一半球的最终 white 用于五套皮层统计，pial 用于 `aparc.pial.stats`。原调用在每套统计中重新读表面、计算顶点面积、white/pial 面积厚度体积和主曲率；每个脑区的 `bool()`、`float()` 以及 CUDA bool 索引还会触发主机等待。

`SurfaceStatsCache` 在一段统计调用期间复用表面基础量。每套图谱仍使用自己的注释及 cortex 选择；输出十列、标签语义、坐标和全局表头的逐顶点 float32 累加顺序保持不变。这是统计阶段的优化，不替换 white/pial 放置、厚度算法或官方参考。

## 缓存边界与数值定义

缓存键包含文件完整路径、大小、纳秒修改时间及状态时间。同一有序网格的 `white.preaparc`、最终 `white` 和 `pial` 是不同的表面版本，不能复用坐标、法线、面积或主曲率。覆盖同一路径后也须重新计算。统计期间禁止其他任务并发改写输入；缓存仅存在于当前上下文，不跨被试或设备使用。

缓存的项目包括网格 NumPy/设备张量、单位法线、两跳邻接、顶点面积、三角面面积、主曲率、曲率统计基础量、厚度、注释、皮层标签以及 `-no-th3` 体积基础量。全局表头只有在有效顶点选择完全相同时复用面积和平均厚度结果；eTIV 按实际 xfm 版本复用。

`-no-th3` 的每个三角面基础量为

```text
mean(thickness[face]) × (white_face_area + pial_face_area) / 6
```

此基础量按原有角顺序累加到三个顶点，再按脑区求和。`lh.volume`/`rh.volume` 使用的 TH3 三四面体分解仍由 `vertex_th3_volume()` 独立计算，不能作为 `-no-th3` 脑区体积输入。

`.area` 顶点图保留原 `norm(cross) * (0.5 / 3.0)` 的单次 float32 乘法。脑区 `SurfArea` 使用独立的 `roi_vertex_area()`：面面积 float32 除以3后转为 float64，按面角累计；不读取已在 float32 中累计舍入的顶点面积图。主曲率保留原两跳邻域、2048 顶点分块、SVD 截断阈值及病态回退。CUDA 矩阵乘法默认 TF32，未使用 FP16/BF16。

CPU 注释先确定脑区顶点和静态切片，保留每个脑区的原顶点顺序。GPU 对这些切片执行原 float64 sum/mean/std，并将整套面积、厚度和体积摘要一次回传；不再为每区读取 GPU bool/float。曲率列沿用原 NumPy 汇总公式和求和顺序。

## 输入、输出和失败行为

| 接口 | 全部输入及默认值 | 输出 |
|---|---|---|
| `SurfaceStatsCache` | `device="cuda:0"`，显式目标 GPU 或 `cpu` | 可用 `with` 管理的缓存，`counters` 是读取/计算次数字典 |
| `roi_vertex_area`（缓存内部方法） | `geometry`为同一缓存`geometry(path=...)`返回的网格对象，含`xyz (N,3)`和`triangles (F,3)`；无参数默认值 | 同设备`(N,) float64`张量，单位mm²；面面积分摊后累计，按该表面版本缓存。输入设备/索引不合法时由PyTorch抛异常。 |
| `write_anatomical_stats` | `subject` 被试目录；`hemi` 为 lh/rh；`atlas` 注释名；`surface` 为 white/pial；`brainvol_stats` 为体积字典；`output` 路径；`device="cuda:0"`；`cache=None` | 创建父目录并写完整 .stats，返回 `Path`；None 使用临时缓存 |
| `anatomical_stats_rows` | `white`、`pial`、`surface` 三个表面；`area_map`；`thickness`；`annotation`；`cortex_label` 或 None；`device="cuda:0"`；`cache=None` | `list[str]`，十列与原格式一致 |
| `roi_area_thickness` | `surface`、`annotation`、`thickness`；`device="cuda:0"`；`cache=None` | `{脑区名: (NumVert, SurfArea, ThickAvg, ThickStd)}`；std 为总体标准差 |
| `roi_gray_volume` | `white`、`pial`、`thickness`、`annotation`；`device="cuda:0"`；`cache=None` | `{脑区名: GrayVol}`，定义为 no-th3 |
| `curvature_columns` | `surface`、`area_map`、`annotation`、`cortex_label` 或 None；`device="cuda:0"`；`cache=None` | `{脑区名: (MeanCurv, GausCurv, FoldInd, CurvInd)}` |
| `principal_curvatures` | `vertices` 为 `(N,3)` 坐标；`faces` 为 `(F,3)` 顶点索引；`device="cuda:0"` | 两个 `(N,)` float32 NumPy 数组，按绝对值较大者为 k1 |
| `anatomical_stats_global_lines` | `area_map`、`thickness`、`annotation`、`cortex_label` 或 None；`brainvol_stats` 字典/路径；`talairach_xfm`；`surface="white"`；`voxel_volume=1.0`；`cache=None` | `list[str]`，完整 `# Measure` 行 |

表面是 surface RAS 坐标（mm）；面积 mm²、厚度 mm、体积 mm³，主曲率 mm⁻¹。morph 和 .annot 顶点图始终对应输入网格的有序顶点，不是原始 T1 体素网格。`area_map` 必须对应正在汇总曲率的表面。white/pial 顶点数和有序三角面必须相同。

缺失文件、无效表面/半球、顶点数或有序面不一致、脑区名不一致、设备不一致均抛异常。排除 `corpuscallosum`、`unknown`、`Unknown`、`Medial_wall` 和空脑区的规则保持原实现。缓存不主动关闭分配器缓存、不调用 `empty_cache()`，也不增加生产流程的显式 CUDA 同步；退出 `with` 后释放自身引用。

### 带具名参数的调用

```python
from pathlib import Path
from fnit.recon_all.anatomical_stats_file import write_anatomical_stats
from fnit.recon_all.anatomical_stats_global import read_brain_volume_stats
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache

subject_directory = Path("/data/fnit/sub-01")  # FNIT 生成的被试目录
brain_volume_measures = read_brain_volume_stats(
    path=subject_directory / "stats" / "brainvol.stats",  # 已生成的全脑 mm³ 体积
)
target_device = "cuda:0"  # 当前进程内明确的目标 GPU
with SurfaceStatsCache(device=target_device) as statistics_cache:
    surface_geometry = statistics_cache.geometry(
        path=subject_directory / "surf" / "lh.white",  # 当前最终white的有序网格；覆盖文件会使缓存失效
    )
    regional_area_base = statistics_cache.roi_vertex_area(
        geometry=surface_geometry,  # 同一缓存的几何对象；返回每顶点float64/mm²脑区面积基础量
    )
    for atlas_name in ("aparc", "aparc.a2009s", "aparc.DKTatlas"):
        write_anatomical_stats(
            subject=subject_directory,  # 包含 surf/label/mri/stats 的目录
            hemi="lh",  # 当前半球；另一半球使用另一段缓存上下文
            atlas=atlas_name,  # 实际 .annot 的图谱名
            surface="white",  # 选择最终 white，不能代用 white.preaparc
            brainvol_stats=brain_volume_measures,  # 全脑体积字典
            output=subject_directory / "stats" / f"lh.{atlas_name}.stats",  # 输出表
            device=target_device,  # 必须与缓存设备一致
            cache=statistics_cache,  # 同一表面版本跨图谱复用
        )
```

## 原实现及命令

这是 [`mris_anatomical_stats -no-th3`](https://surfer.nmr.mgh.harvard.edu/fswiki/mris_anatomical_stats) 的统计内部步骤；法线、邻接和主曲率缓存没有独立的官方命令。冻结参考使用 [FreeSurfer 固定源码 d932c45](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_anatomical_stats/mris_anatomical_stats.cpp)。生产统计只读取 FNIT 自产表面、顶点图、注释及变换，不运行该参考命令。

独立参考路径的典型调用为：

```bash
mris_anatomical_stats -no-th3 -a lh.aparc.annot -cortex lh.cortex.label \
  -f lh.aparc.stats sub-01 lh white
```

FreeSurfer 皮层表面方法背景：Dale, Fischl & Sereno, *NeuroImage* 9, 179–194 (1999), [doi:10.1006/nimg.1998.0395](https://doi.org/10.1006/nimg.1998.0395)。算法回归以固定版本实际源码和同输入结果为准。

## 当前验证及复现

`tests/recon_all/test_surface_stats_cache.py` 检查缓存身份、覆盖失效、no-th3/TH3 区别、原有顺序归约、曲率复用和设备检查。人工小网格仅用于单元测试，不是性能 benchmark。

真实数据配对脚本为 [`benchmark_surface_stats_cache.py`](../../validation/recon_all/python_gpu_port/benchmark_surface_stats_cache.py)。它从冻结原有源目录加载旧实现，以完全相同的已生成真实 white/pial/thickness/annotation 输入生成双侧各六份统计；重复时交换旧/新执行顺序。计时包含网格/图谱读取、加载到目标设备、计算、回传和完整输出写入，并在计时边界同步明确的目标 GPU。

回归门槛预先固定为全部 .stats 文本相同。逐列数值差异同时保留，不能用它们替代此门槛。可选官方对照只报告现有差异，整体指标等效状态仍为未判定。报告绑定候选/基线代码版本、实际模块哈希和输入哈希，并记录硬件、线程、精度、计算复用次数和分配器峰值。分配缓存关闭时 allocated/reserved 写 null；父子进程同时占用须由外层 NVML 监测。

新增实现只使用主页环境已声明的 NumPy、PyTorch、SciPy 与 nibabel，没有新增依赖。

### 2026-10-01 两例真实表面的本次实测

本次使用此前从原始 T1 连续生成的 `sub-01`、`sub-02` FNIT 表面作为冻结同输入。统计输入在旧实现与缓存实现之间完全相同；不重跑表面放置，也不把手动统计补跑称为整例。每轮比较双侧各六份 `.stats`，包含三套主要图谱的 white、aparc 的 pial、BA_exvivo 和 BA_exvivo.thresh 的 white。

计时均在 `gpucw1`、同一物理 GPU UUID `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba` 上进行，逻辑设备 `cuda:0`，PyTorch 2.5.1、Python 3.11.16、PyTorch 线程预算 4；矩阵乘法和 cuDNN TF32 均开启，没有半精度。分配缓存策略保持 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`。

| 真实输入 | 轮次及执行顺序 | 旧实现（s） | 缓存实现（s） | 本统计阶段速度比 | 完整文本回归 |
|---|---|---:|---:|---:|---|
| sub-01 | 1：旧 → 新 | 95.864 | 16.934 | 5.661× | 12/12 相同 |
| sub-01 | 2：新 → 旧 | 73.893 | 12.444 | 5.938× | 12/12 相同 |
| sub-02 | 1：旧 → 新 | 127.219 | 45.661 | 2.786× | 12/12 相同 |
| sub-02 | 2：新 → 旧 | 162.184 | 44.968 | 3.607× | 12/12 相同 |

两例共 48 次文件对照全部逐字节相同，包括全局表头、脑区名称与十列数值；保存文本的逐列最大绝对差均为 0。汇总时再次读取全部输出核对，未降低容差或以平均相关性替代回归。每半球的实际计数均为：读取网格 2 次、顶点面积 2 次、主曲率 2 次（white/pial 各一次）、no-th3 体积 1 次、每套图谱摘要回传 1 次。最终 white 的五套统计复用了同一份主曲率，pial 的结果单独缓存。

| 监测范围：整个旧/新配对命令 | sub-01 | sub-02 |
|---|---:|---:|
| 命令墙钟，包含启动、输入哈希、所有统计和报告写入（s） | 203.600 | 384.544 |
| 父子进程同时 GPU 占用的采样最大值（bytes） | 2,027,945,984 | 2,036,334,592 |
| 同一占用换算为十进制 GB | 2.028 | 2.036 |
| 请求采样间隔（s） | 2.0 | 2.0 |
| 实际最大采样间隔（s） | 2.279 | 2.286 |
| 单次 NVML 查询超时（s） | 5.0 | 5.0 |
| 失败的进程显存查询次数 | 0 | 0 |

显存数值覆盖整个旧/新配对进程，不能分配为单独旧实现或新实现的峰值，也未证明捕获连续峰值。分配缓存关闭时 PyTorch allocated/reserved 为 null，不表示 GPU 占用为零。两轮的绝对耗时有波动，表中只给各自同输入配对速度比；没有从此阶段结果推算整例提速。本次没有新增官方统计对照，整体指标等效仍未判定。

#### 实际测试版本

基线为完整提交 `e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68`。候选是基础提交 `3d9856c9dc659a50b685dbc8c0b9b6c695461fe7` 上的冻结工作区快照，报告标签为 `3d9856c9dc659a50b685dbc8c0b9b6c695461fe7+stage1tar0444db72`；此标签不代表一个干净 Git 提交。候选冻结归档 SHA-256 为：

```text
0444db72c248fac01bf9385c615da4f0e9fbade94ebbe2ccd2a70942f0c2180f
```

远端固定来源是 `volume_parity_20260930/fixes_20261001/stage1code`。该轮实际执行的统计模块 SHA-256 如下，两例相同；它们标识2026-10-01快照，不代表后续ROI面积修复的源码。

| 候选统计模块 | SHA-256 |
|---|---|
| anatomical_stats_file.py | `3aef07cbf0788276341125774717f1e0555a1197b2faeba5c9b75610a468df53` |
| anatomical_stats_rows.py | `0832f2f33006435c330305a9dee224c33441df8a1b7fb420561413a7668fe5f9` |
| anatomical_stats_global.py | `1f33e2e32bbaa3c2676d68a092497a22e3a824be382cc05abe54557300087235` |
| surface_roi_gpu.py | `3c285f800aa5f81b08426a6f104c7b06bdf03137a2f9a51c45ef2e0b018011c8` |
| surface_roi_curvature_gpu.py | `7cb53358fe2004a88cc2a1d89f603564e5824e2c86bfec7b9588efecf74fb126` |
| surface_stats_cache.py | `fd92ec7a92bd0d00bbac72c8d3a16461f8c3df5c8688fd3998c350eab70ce9d8` |

配对脚本 SHA-256 为 `e2438c0ef5050174a682a134e0c5d8055496c76e0bd0bd5fb1affd34b0404008`。每例报告还保留 26 个输入文件的完整 SHA-256、旧实现的模块哈希及实际参数；这些绑定用于识别测试输入和代码，不以文档后续提交替换实际测试版本。统计阶段未加载神经网络权重，也未执行外部 C++ 可执行程序，因此不存在本阶段新增的权重或候选可执行程序版本。

完整机器可读材料：

- [两例汇总及再次文件核对](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/stats_summary.json)
- [sub-01 原始配对报告](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/stats/report.json)、[原始 GPU 监测](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/stats_monitor/monitor.json)
- [sub-02 原始配对报告](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_stats/report.json)、[原始 GPU 监测](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_stats_monitor/monitor.json)

```python
import subprocess

subprocess.run(
    args=[
        "/opt/conda/envs/fnit/bin/python",  # 实际 FNIT 安装环境
        "validation/recon_all/python_gpu_port/benchmark_surface_stats_cache.py",  # 配对脚本
        "--subject", "/data/fnit/sub-01",  # 已完成真实 FNIT 被试，不读取参考中间结果
        "--baseline-source", "/data/fnit/frozen_baseline/src",  # 固定旧源码
        "--baseline-commit", "<baseline-commit>",  # 实际旧提交
        "--candidate-commit", "<candidate-commit>",  # 本次实际测试提交
        "--output", "/data/fnit/diagnostic/stats_cache_sub01",  # 必须不存在的隔离输出目录
        "--device", "cuda:0",  # 明确目标设备
        "--threads", "4",  # 与旧实现相同的线程预算
        "--repetitions", "2",  # 交换执行顺序
        "--official-subject", "/data/reference/sub-01",  # 只用于独立统计诊断，可省略
    ],
    check=True,  # 输入/回归失败直接抛异常
)
```

本页不以单阶段缓存命中数宣布整例加速；真实阶段与整例结果必须分别附机器可读报告。

## 2026-10-03 脑区面积候选修复

特别说明成熟子函数 `SurfaceStatsCache.roi_base()` 的精度问题：原函数复用了 `.area` 的 float32 顶点累加量，而上游 `mris_anatomical_stats.cpp` 逐面以 `f->area / VERTICES_PER_FACE` 累计到 double 脑区面积。两者数学定义相同，但舍入位置不同。候选新增 `roi_vertex_area(geometry)`，保留原 CUDA 计算和一次摘要回传，不改变公开面积图、厚度、no-th3 体积或主曲率。其输入 `geometry` 是由当前缓存 `geometry(path)` 读取的 `_Geometry`（surface RAS/mm，同路径文件版本）；输出是同顶点顺序 `(N,)` float64 设备张量，单位 mm²。缓存按该几何版本持有，退出上下文释放；无效输入仍由 `geometry()` 抛异常。该内部方法没有独立 CLI，Python、FNIT命令和官方命令沿用上文完整统计接口。

```python
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache
with SurfaceStatsCache(device="cuda:0") as statistics_cache:  # 显式GPU与缓存生命周期
    surface_geometry = statistics_cache.geometry(path="/data/sub01/surf/lh.white")  # 最终white，mm
    roi_vertex_areas = statistics_cache.roi_vertex_area(geometry=surface_geometry)  # (N,) double/mm²，供脑区汇总
```

本轮代码基线为 `816e5610417a4c587caf321049438a9554139016`。已完成源码公式核对、编译语法与差异检查；6项单元测试在服务器通过（2.61 s），覆盖面分摊及公开面积图兼容性、no-th3定义和版本失效。fnit_main_env缺少pytest，复用服务器已有同Python 3.11的pytest纯Python测试组件，关闭插件自动加载；计算依赖仍来自fnit_main_env。用户明确授权最小范围补丁和验证脚本上传后，候选已部署到独立验证副本。同输入GPU/官方验证已通过共享锁排队，不能把历史表格改标为本次候选结果。诊断脚本与报告保存在 `validation/recon_all/accuracy_20261003/task_05/`；未获真实验证前候选尚未验收。最终10例整例由协调者运行，整体指标等效为 `not_assessed`。
