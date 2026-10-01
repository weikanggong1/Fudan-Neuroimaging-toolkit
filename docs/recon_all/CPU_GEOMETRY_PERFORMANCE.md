# remesh 与 quick sphere 的顺序计算优化

本轮保留 FreeSurfer 8.2 固定源码 `d932c45b7941662ea380a05efef580568b98d41a`
的算法规则，优化 FNIT 的 Python 执行开销。standard sphere 只做源码审计；
本轮没有修改其算法、注册共享内核或调度默认值，也没有增加运行依赖。
Numba 已在主页 `environment.yml` 和 `pyproject.toml` 中声明。

## 实际瓶颈与计时边界

以下是冻结候选 **1b8c36d** 的两例原始 T1 连续整例记录，不是本轮未提交
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

### standard sphere：只读审计

当前主要优化仍是完整 NumPy/Numba CPU 实现；已有 PyTorch finish 内核只
覆盖相交清理，冻结 1b 的 finish 四侧均低于 2 s。先检查完整阶段中的以下
位置，不应根据不同目标的 registration GPU 内核直接替换它：

| 位置 | 可检查的开销 | 当前证据 |
| --- | --- | --- |
| `sphere_standard_line_search.first_epoch_line_search` | 每次更新的 Python 顺序范数求和、每个 trial 重复转换 float64 基础数组 | 本轮源码审计；尚无新单项剖析 |
| `_distance_sse` | 每个 trial 遍历抽样邻接并计算球面弧距离，O(M) | 旧 e036 cProfile 供定位；不能作为本轮比例 |
| `sphere_standard_average` | 多轮顺序邻接梯度平均，O(rounds×(N+E)) | 固定轮数有算法含义，不能跳过 |
| `sphere_standard_metric._reciprocal_average` | 对每项在线性邻接中查找反向项，O(Σ dᵥdₙ) | 可研究保持遍历/重复平均语义的索引缓存；未修改 |

## 输入、输出及失败行为

| 函数 | 完整输入与默认值 | 输出及空间 |
| --- | --- | --- |
| `remesh_geometry(vertices, faces, iterations=3)` | 有限 (N,3) mm 坐标；有序 (F,3) 整数三角面；非负迭代数 | 新编号的 float32(Nnew,3) 坐标、int32(Fnew,3) 面；surface RAS |
| `remesh_surface(input_path, output_path, iterations=3)` | FreeSurfer 三角表面路径、输出路径、非负迭代数 | 写坐标/面并保留输入原始 volume-info 等尾部；返回 None |
| `smooth(mesh, repeats=2)` | 内部 Mesh 的点、面、边邻接及边界标志；平滑轮数 | 原位更新 `Mesh.points` 的 float64 坐标三元组列表；返回 None |
| `quick_sphere_from_inflated(vertices, faces, niterations=25)` | (N,3) mm inflated 坐标、有序三角面；每个线搜索阶段最大迭代数 | float32(N,3) radius=100 mm 球面，以及 `(k,averages,mode,dt)` 有序 trace；面及顶点编号不变 |
| `write_quick_sphere(input_path, output_path)` | `inflated.nofix` 三角表面路径和输出路径；固定默认 25 | 写 `qsphere.nofix`，保留原面/几何尾部；返回 None |

负 remesh 迭代数抛 `ValueError`；无效索引、非流形拓扑和退化法线沿用
异常/断言，文件读写失败抛异常。quick sphere 的非三角格式、截断文件或
非法数组抛异常。失败输出不能作为成功文件继续使用。remesh 不包含最终
相交修复，该门控仍由 runner 单独执行。内部 `smooth` 和面积/线搜索内核
属于上游命令内部步骤，没有独立官方 CLI。

```python
from fnit.recon_all.mris_remesh_python import remesh_surface
from fnit.recon_all.sphere_quick_python import write_quick_sphere

remesh_surface(
    input_path="/data/fnit_subject/surf/lh.orig.premesh",  # 自产拓扑修复后的三角网格，surface RAS/mm
    output_path="/data/fnit_subject/surf/lh.orig",        # 重划分结果；后续另执行相交检查
    iterations=3,                                        # 标准流程固定的三轮拆边/缩边/平滑
)
write_quick_sphere(
    input_path="/data/fnit_subject/surf/lh.inflated.nofix",  # 自产未修复拓扑的 inflated 网格
    output_path="/data/fnit_subject/surf/lh.qsphere.nofix",  # 固定四轮 quick-sphere 的有序输出
)
```

官方 benchmark 对应 `mris_remesh --remesh --iters 3 INPUT OUTPUT` 和
`mris_sphere -q INPUT OUTPUT`；生产函数不执行这些程序。上游实现：
[mris_remesh](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_remesh/mris_remesh.cpp)、
[mris_sphere](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_sphere/mris_sphere.cpp)、
[表面内部计算](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_deform.cpp)。
参考：Fischl et al., *Cortical surface-based analysis II*, NeuroImage 9 (1999),
195–207，DOI: [10.1006/nimg.1998.0396](https://doi.org/10.1006/nimg.1998.0396)。

## 验证范围与真实回归命令

本地小网格测试 9 项通过：双精度平滑指纹、包含缩边的完整 remesh 指纹、
相同边长的队列顺序及 float32 角点累加均保持冻结旧规则。另以旧 764607c
运行完整 quick sphere，包含翻折面的 116 次更新 trace 与最终坐标完全相同；
两种 CLI 的独立子进程、报告写出及严格比较也完成模拟网格检查。测试环境
为 Python 3.11.16、NumPy 1.26.4、Numba 0.67.0、nibabel 5.4.2、Torch 2.5.1。
这些是小网格/接口验证，不能替代真实 T1；pytest 的 4.87 s 不是阶段耗时。
版本、实际计时键和边界见 [机器可读审计](../../validation/recon_all/python_gpu_port/performance_20261001/cpu_geometry_optimization_audit.json)。

真实四半球配对已完成：remesh 减少 9.48%–15.23%，quick sphere 减少 8.10%–11.48%。坐标、有序面、尾部、拆缩边接受数量及 quick trace 完全相同。计时含输入输出及首次编译/缓存加载；每侧一次、gpucw1、4 线程，共享主机观察，不能直接推算整例。完整结果见 [当前阶段报告](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/README.md)。冻结旧实现的历史真实
[remesh 同输入记录](../../validation/recon_all/python_gpu_port/REMESH_VALIDATION.md)
仍是参考证据，不能用于给本轮优化背书。

[`benchmark_cpu_geometry_pair.py`](../../validation/recon_all/python_gpu_port/benchmark_cpu_geometry_pair.py)
用独立 CPU 进程对同一冻结网格比较旧/新代码，固定线程预算；第二轮颠倒
顺序。分别记录包含导入的进程墙钟和包含读写/JIT 的阶段墙钟、RSS、输入/
源码/输出 SHA；各实现 Numba 缓存隔离，首轮包括编译，后轮可复用缓存。
使用 `--profile` 时保存完整 pstats 与热点文本，额外剖析开销必须单独报告。
在性能配对之外可传入 `--official-output` 做独立参考诊断。

```bash
python benchmark_cpu_geometry_pair.py \
  --stage remesh \
  --input /data/fnit_subject/surf/lh.orig.premesh /data/fnit_subject/surf/rh.orig.premesh \
  --baseline-source /data/frozen_764607c/src \
  --baseline-version 764607c2e34c04bdc418fa64540380ff9d22c112 \
  --candidate-source /data/frozen_candidate/src \
  --candidate-version working-tree-with-recorded-source-sha256 \
  --output /data/validation/remesh_pair_new_directory \
  --threads 4 \
  --repetitions 2
```

参数逐项说明：`--stage` 选择 remesh 或 quick-sphere；`--input` 使用真实
`orig.premesh`（quick 则为 `inflated.nofix`）；两个 source 是冻结 FNIT `src`
目录，version 为各自版本标识（还会自动记录实际源码 SHA）；output 必须为
新的诊断目录；threads 同时固定 Torch/Numba/OMP/BLAS 为 4；repetitions=2
提供反序两轮。追加 `--profile` 得到热点，不把剖析时间当无剖析时间。

回归门槛预先固定：坐标、有序面和几何尾部完全相同；remesh 每次拆边/
缩边接受数及平滑尺寸、quick 全部迭代 trace 也须相同。不同网格不做同索引
距离结论。报告将执行完成、严格回归和官方诊断分别保存；整体指标等效
保持 `not_assessed`，阶段配对不能推断整例提速。
