# recon-all 十分钟目标：超过 100 秒阶段与优化任务

本页绑定当前工作树的真实阶段记录，不把历史报告当作新整例结果。记录来自冻结 `3a0c9aba6321b4981fd8174b4b191515459aa38b` 的九例 FNIT 运行；每例使用同一台 gpucw1、四线程和 `cuda:0`。当前九例 FNIT 中位数为 2669.520 秒，尚未达到 600 秒目标。

## 实测热点

| 阶段 | 九例范围（秒） | 当前实现 | 主要限制 |
|---|---:|---|---|
| `surface_hemisphere_group` | 791.35–1041.12 | 双侧独立进程；拓扑为 Conda C++，平滑/重网格/球面为 Python/Numba，白质放置为 Conda C++ | 单侧表面链仍包含动态碰撞、拓扑 GA、remesh 和标准球面；这些步骤有输出依赖 |
| `finish_surface_hemisphere_group` | 369.38–661.83 | 双侧 Conda C++ final white/pial，指标使用 FNIT PyTorch CUDA | white→pial 是同半球串行；指标读写和原生进程仍有边界 |
| `register_hemisphere_group` | 233.33–367.87 | FNIT Python/Numba，梯度平均使用有序 CUDA，重叠清理在 CPU | 目标函数、步长和部分重叠检查仍在 CPU |
| `annotation_hemisphere_group` | 119.42–189.63 | FNIT Python 标签投射/图谱写出，双侧已并行 | 每套图谱依赖同一注册球面；跨图谱共享缓存尚未统一 |
| `mri_em_register` | 119.83–213.92 | 固定 FreeSurfer 源码的 Conda C++，`cpu_cached` 后端 | EM 迭代和 GCA 搜索仍是 CPU；不能只凭相关性替换算法 |
| `n4` | 124.61–132.16 | Conda 独立编译 ITK N4，单重建线程 | B-spline/收敛迭代依赖强，改变线程或 GPU 算子需体素级回归 |

`input_talairach`、T1/brain normalize、mri segment/fill、MNI nonlinear 和其余统计在当前记录中均低于 100 秒；它们仍需在优化后检查是否出现新的关键路径。阶段总和不能简单相加：半球组内的 worker 秒数与组墙钟是嵌套计时。

## 五个串行控制任务

1. **white/pial 碰撞内核（已提交候选，待同输入回归）**
   - 文件：`src/fnit/recon_all/place_surface_self_repulsion.py`。
   - 方法：用 Numba 标记数组执行一/二环邻域 membership，保留候选顺序、累加顺序和 float32 运算；避免每个空间桶候选再次线性扫描邻域。
   - 依据：冻结真实白质 placement 记录中 collision 约 473.8 秒，明显高于 gradient 约 63.4 秒和 objective 约 51.2 秒。
   - 当前验证：gpucw1 合成网格逐元素回归通过（新旧 force 差异 0，energy 有限）；真实 white/pial 未重跑，不能宣称几何等效或达到 600 秒。合成回归机器报告见 [kernel_regression_20261007.json](../../validation/recon_all/accuracy_20261003/runtime/kernel_regression_20261007.json)。

2. **拓扑/remesh/standard sphere**
   - 先保持动态 collapse、实时邻接和面接受顺序；不能机械并行或改成半精度。
   - 优先针对 `sphere_standard_line_search` 的 `_distance_sse`、metric/邻接缓存和 JIT 初始化做同输入逐顶点/trace 回归。现有实测 standard sphere 完整约 156.9 秒，line-search 约 74.7 秒；这是下一处可量化热点。
   - remesh 的动态失效规则需要保留；已有 CSR/heap 优化不重复实现。

3. **N4/EM 原生阶段**
   - N4 保留 ITK/FP32 语义；先扫 `reconstruction_threads=1/2/4/8`，记录每级耗时、峰值内存、体素最大误差和收敛轨迹，再决定默认线程。
   - EM 保留 Conda C++ `cpu_cached`；先扫固定线程和 GCA/template 进程内缓存。没有同输入变换、LTA、后续归一化回归前，不切换成新 GPU EM。

4. **跨阶段 DAG 与资源令牌**
   - 左右半球四个阶段组已经并行；`surface → register → annotation → finish_surface` 的输出依赖禁止跨组重叠。
   - 可测试的重叠候选是 `orig.mgz` 可用后的 SynthSeg 与 N4，或已完成表面输入后的独立缺陷投射；实现前要固定 CPU/GPU/IO 令牌，防止显存叠加和共享文件竞争。
   - 每个候选须有依赖顺序、单 GPU token、共享路径互斥和重复读计数合同测试；只报告关键路径上限，不把理论上限当作实测提速。

5. **GPU 化白质/球面主算子和指标缓存**
   - 逐步把可验证的 C++/CPU 算子迁移到已有 PyTorch/Numba GPU 内核：优先碰撞候选、球面梯度/距离 SSE、表面指标缓存；保留官方步长接受与 FP32 例外。
   - 每半球缓存同一表面版本的网格、邻接、法线、面积和主曲率；white.preaparc、final white、pial 必须分开缓存。脑区体积继续使用 `-no-th3` 定义。
   - 端到端计时必须包含模型加载、传输、进程启动和写出；同时记录父子进程同期显存。任何替换先做冻结输入回归，再做自产链和原始 T1 整例。

## 十分钟目标的验收边界

600 秒是新的性能目标，不是已建立的精度门槛。当前阶段关键路径显示，仅靠半球并行不能达到该目标；需要 surface/finish/register 主内核获得数量级加速。优化期间继续分别报告：严格逐文件复现、优化是否引入退化、分区 Dice/双向表面距离/厚度面积体积偏差，以及端到端墙钟和同期显存。整体指标等效仍为 `not_assessed`，不得通过放宽阈值或删除失败输出换取通过。
