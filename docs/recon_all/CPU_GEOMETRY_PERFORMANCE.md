# remesh 与 quick sphere 的顺序计算优化

## 功能与实现范围

本轮保留 FreeSurfer 8.2 固定源码 `d932c45b7941662ea380a05efef580568b98d41a`
的算法规则，优化 FNIT 的 Python 执行开销。remesh 重新划分拓扑修复后的
网格，供 white 放置使用；quick sphere 从未修复的 inflated 网格生成
qsphere.nofix，供拓扑修复的球面前处理使用。两者是固定上游算法的
Python/Numba CPU 移植，保留完整步骤；生产不执行原软件命令。

本页 remesh/quick 实测绑定 **`c24852054f3321c1142b1ae88fa3d2bf68329bb3`**；
标准球面接口与新实测绑定 **`74ae022edd932e9f3c57c6e253678f13853f8842`**。
真实配对实际运行的是下面记录的源码归档，当前 remesh/quick 模块与实测
归档 SHA 相同。两例从原始T1开始的新整例已完成：GPU命令墙钟减少14.676%，CPU减少0.388%，不能将局部阶段提速等同于整例提速；[完整配对](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/WHOLE_RESULTS.md)保留未修改原生阶段变慢的实测。
Numba 已在主页 [environment.yml](../../environment.yml) 和
[pyproject.toml](../../pyproject.toml) 中声明，无新增依赖或半精度。

## 实际瓶颈与计时边界

以下是冻结候选 **1b8c36d** 的两例原始 T1 连续整例记录，不是本轮已提交
优化的结果。秒数来自 [sub01](../../validation/recon_all/python_gpu_port/performance_20261001/whole/sub01/candidate_run.json)
和 [sub02](../../validation/recon_all/python_gpu_port/performance_20261001/whole/sub02/candidate_run.json)。

| 被试/半球 | remesh 含读写 | quick sphere 含读写 | standard sphere 含读写 | 其中 metric 初始化 | 其中 finish |
| --- | ---: | ---: | ---: | ---: | ---: |
| sub01 LH | 130.714 s | 76.305 s | 238.426 s | 25.815 s | 1.766 s |
| sub01 RH | 126.888 s | 77.869 s | 176.046 s | 23.135 s | 1.454 s |
| sub02 LH | 124.655 s | 81.841 s | 156.990 s | 26.214 s | 0.017 s |
| sub02 RH | 120.311 s | 83.079 s | 141.241 s | 24.225 s | 1.925 s |

`surfaces.H.topology_remesh_seconds` 只包围 `remesh_surface(...)`；
`topology_python_seconds` 是 `smoothwm.nofix` 平滑、remesh 和相交检查之和。
拓扑 GA 单独计入 `topology_native_seconds`，quick sphere 另计
`sphere_seconds.qsphere_nofix_python`。`sphere_seconds.sphere` 等于
`standard_sphere_report.total_seconds_including_io`；metric/finish 是其子段，
不能再加到总段。上述 1b 记录没有 remesh/quick 的本轮 cProfile，因此源码
热点仍需冻结同输入剖析确认，不能从整段秒数推断各内核占比。

## 本轮修改

### remesh

[`mris_remesh_python.py`](../../src/fnit/recon_all/mris_remesh_python.py) 保留动态
拆边/缩边的顺序和规则。初始边队列由逐项 `heappush` 改成 `heapify`：构建
从 O(E log E) 变为 O(E)，长度相同仍由原边编号决定次序；动态更新仍用原来
的重新入堆规则。

平滑复用项目已有 Numba，将面积/法线累加和逐顶点更新编译成顺序 CPU
内核。每次平滑只构建一次升序邻接 CSR，并合并重复的面叉积/长度计算。
每轮仍重算几何量，保持 float64、面序/角点序累加和顶点升序原位更新，
不启用 `fastmath` 或并行归约。这里不能直接改为所有顶点同时更新的 GPU
算法：后一顶点会读取前一顶点本轮已经更新的坐标。

### quick sphere

[`sphere_quick_python.py`](../../src/fnit/recon_all/sphere_quick_python.py) 的面积
梯度与能量每次只 gather 一份面坐标、计算一次叉积；同次线搜索复用
坐标/梯度的 float64 试探基础数组，候选更新仍按原表达式量化为 float32。
角点梯度汇总复用 `inflate_python._accumulate_corner_normals`，依照原面序、
角点序累加 float32，避免另写相同归约内核。四轮权重、平滑轮数、继承动量、
步长试探、二次拟合、停止和接受规则保持原值。

### standard sphere：有序平均与静态拓扑复用

后续串行优化已修改 standard sphere，实测版本为 74ae022。
复用已有 RegistrationGradientAverager，在主设备为 CUDA 时执行完整有序
Jacobi 梯度平均；目标函数的独立顶点行用 Numba 四线程计算，再按原顶点序
归约。只缓存固定面 CSR 和原 smoothwm 面积，变化坐标及法向每轮重算。
不跳过固定平均轮数、线搜索或相交清理，不启用 fastmath。
最新四半球冻结输入结果与完整接口见[串行优化](SERIAL_OPTIMIZATION.md)。
本页下面 remesh/quick 的历史实测仍绑定其实际归档，不能改标为本次新结果。

## Python 接口：输入、输出及失败行为

| 函数 | 完整输入与默认值 | 输出及空间 |
| --- | --- | --- |
| `remesh_geometry(vertices, faces, iterations=3)` | 有限 (N,3) mm 坐标；有序 (F,3) 整数三角面；非负迭代数 | 新编号的 float32(Nnew,3) 坐标、int32(Fnew,3) 面；surface RAS |
| `remesh_surface(input_path, output_path, iterations=3)` | FreeSurfer 三角表面路径、输出路径、非负迭代数 | 写坐标/面并保留输入原始 volume-info 等尾部；返回 None |
| `smooth(mesh, repeats=2)` | 内部 Mesh 的点、面、边邻接及边界标志；平滑轮数 | 原位更新 `Mesh.points` 的 float64 坐标三元组列表；返回 None |
| `quick_sphere_from_inflated(vertices, faces, niterations=25)` | (N,3) mm inflated 坐标、有序三角面；每个线搜索阶段最大迭代数 | float32(N,3) radius=100 mm 球面，以及 `(k,averages,mode,dt)` 有序 trace；面及顶点编号不变 |
| `quick_sphere_from_projected(vertices, faces, original_face_area, original_total_area, niterations=25, initial_momentum=None)` | 已按上游规则投影的 float32(N,3) mm 球面、有序(F,3)面、初始化(F,)mm²面积及总mm²面积；线搜索上限25；可选float32(N,3)继承动量，None表示零 | 同上；此诊断接口要求调用者提供正确上游状态，不能用任意投影替代标准前处理；动量相位仍固定10轮 |
| `write_quick_sphere(input_path, output_path)` | `inflated.nofix` 三角表面路径和输出路径；固定默认 25 | 写 `qsphere.nofix`，保留原面/几何尾部；返回 None |
| `run_standard_sphere(inflated, smoothwm, output, finish_device="cpu", averaging_device="cpu")` | 同顶点/有序面的 inflated/smoothwm 路径、输出路径；finish_device默认CPU，只控制末尾清理 | 写float32(N,3)半径100mm sphere，并返回下面列出的dict；目标函数仍CPU；averaging_device控制有序平均，面/顶点编号保留 |

负 remesh 迭代数抛 `ValueError`；无效索引、非流形拓扑和退化法线沿用
异常/断言，文件读写失败抛异常。quick sphere 的非三角格式、截断文件或
非法数组抛异常。quick 的 `niterations` 仅影响线搜索相位，没有专门的负值
校验；0不能表示跳过全部优化，应使用标准默认25，不以它减少正常步骤。
标准 sphere 的不同有序面抛 ValueError，3000次更新内不收敛抛 RuntimeError。
失败输出不能作为成功文件继续使用。remesh 不包含最终
相交修复，该门控仍由 runner 单独执行。内部 `smooth` 和面积/线搜索内核
属于上游命令内部步骤，没有独立官方 CLI。

标准 sphere 返回 dict：`inflated/smoothwm/output` 路径、`finish_device`、
`initial_negative_area_pct` 百分数、`projection_seconds`、
`metric_seconds_including_jit`、`finish_seconds`、`total_seconds_including_io`，
以及 `updates` 和 `negative_counts`。每个 updates 项记录 index、stage、
weight、averages、dt、seconds；negative_counts 是末尾清理的负面积面计数。
子时间已计入总时间。quick 数组 API 的 trace 是四元素元组列表，不含耗时；
文件 API 返回 None，完整耗时与 trace 由 benchmark 包装另外保存。

本轮改动涉及的内部算子也保留原定义：

| 函数 | 全部输入 | 返回结构 |
| --- | --- | --- |
| `_face_geometry_from_points(points)` | float32(F,3,3)三角面坐标，surface RAS/mm | 两个边向量(F,3)、叉积(F,3)、绝对面积(F,)mm²；原float32顺序 |
| `nonlinear_area_gradient(vertices, faces, original_face_area, original_total_area, k)` | float32(N,3)mm坐标、有序(F,3)面、初始化(F,)mm²面积、总mm²面积、原轮权重k=10/40/160/640，均必填 | float32(N,3)梯度、float目标能量、int负面积面数；不改输入 |
| `nonlinear_area_sse(vertices, faces, original_total_area, k)` | 同序坐标/面、初始化总mm²面积、原轮权重，均必填 | float目标能量；不写文件、不改输入 |
| `_line_minimize(vertices, gradient, faces, original_total_area, k)` | float32(N,3)mm坐标及同shape梯度、有序面、总mm²面积、原轮权重，均必填 | 原试探/二次拟合顺序选中的float dt；零平均梯度返回0；坐标与梯度的double临时数组仅在本次搜索复用 |

这些内部目标数值使用上游的面积缩放定义，不是新增厚度/面积统计指标；
非法数组/索引沿用 NumPy 异常，不在内部修补网格。有限性、拓扑及完整输出
由上层阶段检查。下面示例使用全部具名参数：

```python
from fnit.recon_all.mris_remesh_python import remesh_surface
from fnit.recon_all.sphere_quick_python import write_quick_sphere
from fnit.recon_all.sphere_standard_run import run_standard_sphere

remesh_surface(
    input_path="/data/fnit_subject/surf/lh.orig.premesh",  # 自产拓扑修复后的三角网格，surface RAS/mm
    output_path="/data/validation/remesh_new/lh.orig",   # 新诊断目录的重划分结果；后续另执行相交检查
    iterations=3,                                        # 标准流程固定的三轮拆边/缩边/平滑
)
write_quick_sphere(
    input_path="/data/fnit_subject/surf/lh.inflated.nofix",  # 自产未修复拓扑的 inflated 网格
    output_path="/data/validation/quick_new/lh.qsphere.nofix",  # 固定四轮 quick-sphere 的有序输出
)
standard_report = run_standard_sphere(
    inflated="/data/fnit_subject/surf/lh.inflated",     # 真实有序inflated；surface RAS/mm
    smoothwm="/data/fnit_subject/surf/lh.smoothwm",     # 同顶点及面编号，提供metric
    output="/data/validation/sphere_new/lh.sphere",     # 新诊断目录的标准sphere
    finish_device="cpu",                              # 仅末尾清理
    averaging_device="cuda:0",                         # 复用已有GPU有序梯度平均器
)
```

上述输出父目录须事先创建；这三个调用展示不同阶段的接口，不是一条从
原始T1开始的整例命令，也不会跳过拓扑修复或 white 放置的前置步骤。

## CLI 调用

quick sphere 自带模块 CLI，只有两个必填位置参数，无 device、N 或迭代
选项；迭代规则固定为上面的标准默认值。输出父目录须存在；推荐在新目录
写出，文件接口可能覆盖同路径的现有结果。

```bash
# 两个参数依次是自产 inflated.nofix 和诊断输出 qsphere.nofix。
python -m fnit.recon_all.sphere_quick_python \
  /data/fnit_subject/surf/lh.inflated.nofix \
  /data/validation/quick_new/lh.qsphere.nofix
```

remesh 模块没有独立 main/CLI 或已安装的 remesh console command；通过上面
具名 Python API 使用，计时和真实输入回归使用下节现有 benchmark CLI。
标准 sphere 的 CLI 为：

```bash
# 同序 inflated/smoothwm、输出 sphere；finish-device 只选择末尾清理。
python -m fnit.recon_all.sphere_standard_run \
  /data/fnit_subject/surf/lh.inflated \
  /data/fnit_subject/surf/lh.smoothwm \
  /data/validation/sphere_new/lh.sphere \
  --finish-device cpu \
  --averaging-device cuda:0 \
  --report /data/validation/sphere_new/api_report.json
```

`--finish-device` 默认cpu，`--report` 可省略，省略时JSON仍写标准输出；averaging-device默认cpu，选择cuda:0仅迁移有序梯度平均。模块 CLI 不管理线程预算，正式配对脚本记录并固定各线程设置。

## 官方调用、原代码与文献

官方 benchmark 对应以下命令；位置参数依次为相同冻结输入与独立参考输出，
`--iters 3` 对应 remesh 的默认3轮，`-q` 选择 quick-sphere：

```bash
mris_remesh --remesh --iters 3 \
  /data/fnit_subject/surf/lh.orig.premesh /data/reference/lh.orig
mris_sphere -q \
  /data/fnit_subject/surf/lh.inflated.nofix /data/reference/lh.qsphere.nofix
```

标准 sphere 则为 `mris_sphere INPUT OUTPUT`，还需要同被试 smoothwm 的
metric 输入；实际固定流程及参数见
[标准球面验证资料](../../validation/recon_all/python_gpu_port/SPHERE_STANDARD_STATUS.md)。
这些命令仅用于独立 benchmark，生产函数不执行它们。上游实现：
[mris_remesh](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_remesh/mris_remesh.cpp)、
[mris_sphere](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_sphere/mris_sphere.cpp)、
[表面内部计算](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_deform.cpp)。
参考：Fischl B, Sereno MI, Dale AM. *Cortical surface-based analysis. II:
Inflation, flattening, and a surface-based coordinate system.* NeuroImage 9
(1999), 195–207。[原论文索引](https://pubmed.ncbi.nlm.nih.gov/9931269/)，
DOI：[10.1006/nimg.1998.0396](https://doi.org/10.1006/nimg.1998.0396)。

## 最新真实数据与验证范围

本地小网格测试 9 项通过：双精度平滑指纹、包含缩边的完整 remesh 指纹、
相同边长的队列顺序及 float32 角点累加均保持冻结旧规则。另以旧 764607c
运行完整 quick sphere，包含翻折面的 116 次更新 trace 与最终坐标完全相同；
两种 CLI 的独立子进程、报告写出及严格比较也完成模拟网格检查。测试环境
为 Python 3.11.16、NumPy 1.26.4、Numba 0.67.0、nibabel 5.4.2、Torch 2.5.1。
这些是小网格/接口验证，不能替代真实 T1；pytest 的 4.87 s 不是阶段耗时。
版本、实际计时键和边界见 [机器可读审计](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_geometry_optimization_audit.json)。

真实四半球配对已完成；实际基线为
`1b8c36d25a68e253a1e59b6d02114890afa467de`，候选为 `764607c` 加
归档 `7a79fed40ab97a0675ba9fef2112eb58e7b393090ce4e4388604e7cccbf1a8f0`。
使用两例 FNIT 自产的冻结前置网格，在 gpucw1 同机依次执行独立 CPU
进程，Torch/Numba/OMP/BLAS 为4，float64 remesh 和原float32 quick规则。
每侧一次 AB 配对、未启用 cProfile，阶段墙钟包含文件读写/JIT，不含
Python 导入；进程墙钟及 RSS 另外记录。当前两模块的 SHA 与实测候选相同。

| 真实输入 | remesh 基线 | remesh 候选 | quick 基线 | quick 候选 | 严格阶段回归 |
| --- | ---: | ---: | ---: | ---: | --- |
| sub01 LH | 119.283782 s | 105.333419 s | 79.449374 s | 70.325066 s | 全部通过 |
| sub01 RH | 115.113825 s | 104.008209 s | 82.657348 s | 74.927106 s | 全部通过 |
| sub02 LH | 140.459920 s | 127.147096 s | 92.658763 s | 85.156390 s | 全部通过 |
| sub02 RH | 140.617393 s | 119.204590 s | 95.265691 s | 85.288362 s | 全部通过 |

remesh 的阶段墙钟减少 9.48%–15.23%，quick sphere 减少 8.10%–11.48%。
全部坐标最大/P99误差及不同分量数为0，有序面/几何尾部相同；remesh
每次拆边/缩边接受数量、平滑调用尺寸及 quick 全部 trace 相同。整文件
头部可能由 writer 重建，严格门槛是上述有序几何/尾部，不声称所有输出文件
逐字节相同。这些结果不能直接推算整例速度，亦未在本轮重跑官方程序。

| 原始报告 | SHA-256 |
| --- | --- |
| [remesh 前三侧](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/remesh_other_three_pair.json) | `0d409050a578230495ef92106fe674b39be18e540db0868fa8c0fbc726ea7b87` |
| [remesh sub02 RH](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/remesh_sub02_rh_pair.json) | `bd20abb9d1d0396124f73a445e0c43a14f1c3ff26ee37438f97cc854f19e47a9` |
| [quick 四侧](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/quick_sphere_four_pair.json) | `f4d18dc247b26e43fe18f22e0af846e7ca874316125697da62b8d5960d8715b5` |

完整上下游边界见 [当前阶段报告](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/README.md)。冻结旧实现的历史真实
[remesh 同输入记录](../../validation/recon_all/python_gpu_port/REMESH_VALIDATION.md)
仍是参考证据，不能用于给本轮优化背书。

## 真实输入回归 CLI

[`benchmark_cpu_geometry_pair.py`](../../validation/recon_all/python_gpu_port/benchmark_cpu_geometry_pair.py)
用独立 CPU 进程对同一冻结网格比较旧/新代码，固定线程预算；第二轮颠倒
顺序。分别记录包含导入的进程墙钟和包含读写/JIT 的阶段墙钟、RSS、输入/
源码/输出 SHA；各实现 Numba 缓存隔离，首轮包括编译，后轮可复用缓存。
使用 `--profile` 时保存完整 pstats 与热点文本，额外剖析开销必须单独报告。
在性能配对之外可传入 `--official-output` 做独立参考诊断。

```bash
python validation/recon_all/python_gpu_port/benchmark_cpu_geometry_pair.py \
  --stage remesh \
  --input /data/fnit_subject/surf/lh.orig.premesh /data/fnit_subject/surf/rh.orig.premesh \
  --baseline-source /data/frozen_1b8c36d/src \
  --baseline-version 1b8c36d25a68e253a1e59b6d02114890afa467de \
  --candidate-source /data/frozen_candidate/src \
  --candidate-version 764607c+archiveSHA:7a79fed40ab97a0675ba9fef2112eb58e7b393090ce4e4388604e7cccbf1a8f0 \
  --output /data/validation/remesh_pair_new_directory \
  --threads 4 \
  --repetitions 2
```

参数逐项说明：`--stage` 必填，选择 remesh 或 quick-sphere；`--input` 必填，
一次列出全部真实 `orig.premesh`（quick 则为 `inflated.nofix`），不要重复
写同一个 `--input`；两个 source 是冻结 FNIT `src`
目录，version 为各自版本标识（还会自动记录实际源码 SHA）；output 必须为
新的诊断目录；`threads` 默认4、接受正整数，本次实测固定4；`repetitions`
默认2、接受正整数，本例2提供反序两轮，以上已完成实测为1轮。`--profile`
默认关闭，开启得到pstats/热点文本，不把剖析时间当无剖析时间。
`--official-output` 默认不使用，可一次列出与 inputs 等长的已生成参考
表面作诊断；不会执行官方程序，也不会把参考作为计算输入。

回归门槛预先固定：坐标、有序面和几何尾部完全相同；remesh 每次拆边/
缩边接受数及平滑尺寸、quick 全部迭代 trace 也须相同。不同网格不做同索引
距离结论。报告将执行完成、严格回归和官方诊断分别保存；整体指标等效
保持 `not_assessed`，阶段配对不能推断整例提速。

## 版本记录与待验证项

- 旧整例及本次性能基线：`1b8c36d25a68e253a1e59b6d02114890afa467de`。
- 真实阶段候选：[CPU 源码归档](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/candidate_source_snapshot.json)，
  SHA `7a79fed40ab97a0675ba9fef2112eb58e7b393090ce4e4388604e7cccbf1a8f0`。
- 当前提交：`c24852054f3321c1142b1ae88fa3d2bf68329bb3`；remesh 模块 SHA
  `fc4cb44cb9517b17a164e87f68125148cba96005f6911367d297ac9ff308f9d1`，
  quick 模块 SHA `e511ce901f8115cd2cc4d4f4bb8d405dd8891fbb075b8e29dc66bf8d6d4e210b`，
  与实测归档相同，不能把旧归档运行记录重新标成当前提交实测。
- 当前完整整例及所有内部阶段时间见[GPU配对](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/whole/sub01/timing_pair_summary.json)与[CPU配对](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/whole/sub02/timing_pair_summary.json)。
  本页两个函数只用CPU；完整GPU流程同一时刻父子进程采样峰值14,508,097,536字节，
  低于20,000,000,000字节预算，但未证明连续峰值。不能从CPU子阶段推断显存。

分别保留执行完成、输出完整性、网格质量、严格阶段复现和整例指标状态。
目前整体指标等效标准尚未正式确认，阶段完全相同不构成整体等效批准；
原有官方精度与局部质量问题继续记录，不通过降低阈值或省略步骤换取速度。
