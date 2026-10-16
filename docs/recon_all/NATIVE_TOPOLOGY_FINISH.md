# 显式 Conda 表面加速配方

## 1. 功能与流程

本配方针对表面步骤的完整墙钟：拓扑遗传搜索继续由固定源码独立编译的 `mris_fix_topology_fnit` 执行，先以同一 Conda 安装的 `mris_remove_intersection` 清理 GA 的私有 premesh 副本，再以 `mris_remesh` 执行三轮 remesh，最后再次执行相交清理。原始 `orig.premesh` 始终保留。`surface_placement_backend="native"` 调用自编译程序放置 `white.preaparc`、最终 `white` 和 pial，并用 `mrisp_paint` 绘制 `avg_curv`、`mris_curvature_stats` 生成完整曲率统计。此次新增的 `topology_finish_backend` 只选择 preclean、remesh 与 postclean。CUDA 上的厚度、面积、平均曲率和脑区统计继续使用自有 PyTorch 函数。默认 `topology_finish_backend="python"` 保留原流程；完整独立 Python 配方也保持可用。

```mermaid
flowchart LR
  A[自产 filled 与 norm] --> B[orig.nofix 与 qsphere.nofix]
  B --> C[Conda 自编译拓扑 GA]
  C --> D[orig.premesh]
  D --> P[Conda 自编译 preclean 私有候选]
  P --> E[Conda 自编译三轮 remesh 私有候选]
  E --> F[Conda 自编译 postclean 私有候选]
  F --> G{双谓词零相交及闭合流形 genus-0 门}
  G -->|通过| H[原子发布 orig 并继续 white/pial]
  G -->|失败| I[保留原 orig、候选及收据，停止]
```

## 2. Python 调用、输入和输出

以下最小示例只展示原生表面选项；其余选项使用接口默认值，**不等于本轮整例测速配方**。本轮完整配方和复现命令见第 3 节。

```python
from pathlib import Path
from fnit.recon_all.native_free import run_recon_all_python

reconstruction_report = run_recon_all_python(
    t1=Path("/data/public/sub-06_T1w.nii.gz"),  # 原始单幅 3D T1w NIfTI，scanner 空间
    subject_dir=Path("/data/runs/sub-06"),  # 全新空目录；生成 138 项标准输出及 JSON
    weights_dir=Path("/data/fnit/weights"),  # 已校验的 FNIT 权重目录
    assets_dir=Path("/data/fnit/assets"),  # 已校验的模板、图谱及数据资产
    native_bin_dir=Path("/data/fnit-conda/bin"),  # 固定源码在 Conda 内独立编译的程序
    backend="native",  # 允许显式的混合 FNIT/Conda 配方
    topology_backend="native",  # 自编译完整拓扑 GA；不改 Python 默认配方
    topology_finish_backend="conda-cpp",  # preclean→三轮 remesh→postclean→严格门→原子发布
    surface_placement_backend="native",  # 已有自编译 white/pial 和 atlas paint 路径
    device="cuda:0",  # Synth 与自有 GPU 阶段的目标设备；C++ 表面程序使用 CPU
    threads=4,  # 全被试 CPU 线程预算；双侧 worker 时平分
    hemisphere_workers=2,  # 使用私有目录运行双侧，并在屏障后发布
)
```

`orig.premesh`、`orig` 均为 FreeSurfer 三角面文件，顶点 `(N, 3)` 使用 surface RAS、毫米，面 `(F, 3)` 为有序顶点索引。`mri/filled.mgz` 与 `mri/norm.mgz` 属于 conformed 体素网格。`mris_remesh` 可以改变顶点数和拓扑；清理阶段必须保留 remesh 后的有序面、顶点数与 MRI 几何头。成功时 `surf/H.orig` 是清理后的输出，`scripts/H.native-preclean.json`、`scripts/H.native-remesh.json` 与 `scripts/H.native-intersection.json` 记录输入、候选、程序和输出 SHA-256、阶段秒数及质量检查；`scripts/H.native-topology-finish.json` 绑定完整链、最终候选 SHA 与原子发布，原始 premesh 的 SHA 也会复核。末次候选须通过闭合、单连通、genus-0、非流形顶点、面方向、零面积与两种独立相交谓词的检查。候选使用 `surf/H.orig.native-preclean.candidate`、`H.orig.native-remesh.candidate`、`H.orig.native-intersection.candidate`；原子写出暂用 `H.orig.native-publish.candidate`，替换 `H.orig` 后该暂存文件消失。文件名从 `lh.`/`rh.` 开始，按现有半球所有权规则经过双侧成功屏障发布；任一侧质量检查失败时，两侧均不向公共被试目录发布候选或改写 `orig`。顶层返回值和 `fnit-native-free-run.json` 还记录程序 SHA、138 项路径和完整墙钟。上述 `scripts/H.native-*.json` 是单半球函数内部路径；本轮双侧 `groups` 模式按通用发布规则保留为 `scripts/surface.H.H.native-*.json`，实际公共文件以半球组报告的 `published` 列表为准。

`topology_finish_backend` 仅接受 `python`（默认）和 `conda-cpp`。后者只允许与 `backend="native"`、`topology_backend="native"` 搭配；程序缺失、命令失败、无效网格、MRI 几何头改变或任一相交谓词仍为非零均抛异常。原生程序退出码 0 不代表相交门通过。三步均先写入 `surf/H.orig.native-*.candidate`。直接调用单半球阶段函数时，成功和失败均保留三份候选及收据；双侧组模式下，仅在两侧全部成功后发布候选，失败则按既有组清理规则删除私有副本并保留 worker 日志和组报告。只有最终候选与完整质量收据的 SHA 一致时，才通过单次原子替换发布 `orig`；失败时原有 `orig` 保持原字节，或继续不存在。`backend="python"` 的纯自有计算配方继续使用 Python remesh 和清理，不会检查或调用这些原生程序。

安装前提是主页 Conda 路径确实独立构建并列出 `mris_remesh`、`mris_remove_intersection` 和 `mris_fix_topology_fnit` 的哈希。若当前安装清单缺少 `mris_remesh`，入口会在创建被试输出前报缺失程序；不能从系统 FreeSurfer 复制二进制。个人许可证只留在私有运行环境，不进入仓库。

## 3. 命令行调用

与上方对应的最小 CLI 示例：

```bash
fnit-recon-all /data/public/sub-06_T1w.nii.gz /data/runs/sub-06 \
  --weights-dir /data/fnit/weights \
  --assets-dir /data/fnit/assets \
  --native-bin-dir /data/fnit-conda/bin \
  --backend native \
  --topology-backend native \
  --topology-finish-backend conda-cpp \
  --surface-placement-backend native \
  --device cuda:0 \
  --threads 4 \
  --hemisphere-workers 2
```

命令行位置参数依次是原始 T1 路径、全新空输出目录；各具名参数含义与上方 Python 调用逐项相同。整例计时包含资源检查、加载、数据传输、计算和写出；阶段秒数嵌套在半球组中，不能相加成整例墙钟。

本轮测速冻结科学源码为 `93affde8d369ee6ade22f2c105701ed6577e89a4`。完整调度使用[两例整例复现脚本](../../tools/run_recon_native_end_to_end_cases.py)，从两个原始 T1 的全新空目录顺序启动，逐例保存外层墙钟、退出码及 benchmark。`cases-json` 是列表，每项包含唯一的 `case` 目录名、原始影像的绝对路径 `t1` 和文件 `sha256`；所有输入哈希通过后才创建输出。权重、资产和程序目录需已经安装、校验；`FS_LICENSE` 在私有环境指向个人许可证。调用前应将目标 GPU UUID 映射到 `CUDA_VISIBLE_DEVICES`，脚本中的 `cuda:0` 是该可见设备的逻辑编号。

```bash
python tools/run_recon_native_end_to_end_cases.py \
  --cases-json /data/private/two_cases.json \
  --output-root /data/runs/native_finish_93affde8_new \
  --benchmark-script tools/benchmark_recon_torch_end_to_end.py \
  --weights-dir /data/fnit/weights \
  --assets-dir /data/fnit/assets \
  --native-bin-dir /data/fnit-conda/bin \
  --code-version 93affde8d369ee6ade22f2c105701ed6577e89a4 \
  --n4-backend simpleitk \
  --n4-execution in-process \
  -- \
  --input-execution serial \
  --hemisphere-execution groups \
  --hemisphere-workers 2 \
  --native-optimizations torch \
  --normalization-controls-backend torch \
  --normalization-initial-bias-backend torch \
  --wm-backend torch-optimized \
  --wm-edit-backend python \
  --wm-execution isolated \
  --gca-inverse-backend torch \
  --gca-candidate-chunk 4096 \
  --gca-reuse-atlas-context \
  --gca-execution isolated \
  --fill-backend numba \
  --defects-backend torch \
  --inflate-backend torch \
  --sphere-finish-backend torch \
  --annotation-gibbs-backend numba \
  --mni-execution parallel-late
```

脚本固定 `backend=native`、`topology_backend=native`、`topology_finish_backend=conda-cpp`、`surface_placement_backend=native`、`device=cuda:0`、总线程 `4`；双侧 worker 各分配 `2` 线程。输出根必须不存在。`code-version` 是报告标签，运行者仍需核对源码提交及 SHA；仅填写此参数不证明实际使用了对应源码。

| 本轮选项 | 实际计算与调度 |
| --- | --- |
| `n4_backend=simpleitk`、`n4_execution=in-process`、`input_execution=serial` | N4 使用通用 ITK 的 Python API/CPU；N4 与 SynthSeg 串行，不调用 FreeSurfer N4 阶段程序 |
| `native_optimizations=torch`、GCA 三个参数和 `gca_reuse_atlas_context=True` | 自有 Torch 候选评分/GPU 求逆、完整 Python EM；每批 4096 候选，复用不可变 atlas context，在独立 exec 计算 |
| 两项 `normalization_*_backend=torch` | 复用已有 GPU 邻域与初始偏置；控制点等有序 CPU 计算保留 |
| `wm_backend=torch-optimized`、`wm_execution=isolated`、`wm_edit_backend=python` | 优化后的自有 Torch/CPU 混合 WM 分割在独立 exec 运行，aseg 核心编辑保持完整 Python/Numba |
| `fill_backend=numba`、`defects_backend=torch` | filled 使用自有 Numba；缺陷体积使用完整 PyTorch 投射，父进程按 LH→RH 累积共享文件 |
| `inflate_backend=torch`、`sphere_finish_backend=torch` | **标准** inflated/sulc 和球面 dense 收尾使用自有 GPU；native placement 的 **nofix inflation** 仍调用 Conda C++ `mris_inflate`。quick sphere、球面优化和配准继续使用自有 Python/Numba 与 GPU averaging；法向沿用默认 Numba |
| `surface_placement_backend=native` | preaparc/final white/pial 使用 Conda C++；white 可选择通过能力校验的 white_fast，pial 使用标准 `mris_place_surface`；avg_curv 和 curv.stats 也使用 Conda C++ |
| CUDA 顶点指标及脑区统计 | thickness、area/area.pial、curv/curv.pial、area.mid 和顶点体积继续使用自有 PyTorch；六套图谱统计共享表面版本缓存，脑区体积保持 `-no-th3` 定义 |
| `annotation_gibbs_backend=numba`、`hemisphere_execution=groups` | 注释使用 Torch 几何与有序 Numba Gibbs；双侧按 surface→register→annotation→finish_surface 屏障推进，单侧最终 white 后接 pial；最终 GPU 指标串行 |
| `mni_execution=parallel-late` | 完整自有 MNI GPU 非线性阶段延后，与 CPU 最终网格检查并行；join 后检查 138 项输出 |

原生曲率统计最终生成 smoothwm 的 K/H/K1/K2/S/C/BE/FI 八图。表面前段仍会用自有 GPU 计算部分主曲率图，因此这份配方尚有重复计算；不能仅因运行报告列出 `mris_place_surface` 路径或 SHA，就将 CUDA 顶点指标记为原生计算。

`orig` 发布前执行完整结构与双谓词零相交门。整例的**生产基本网格门**检查同序面、有限坐标、Euler/闭合以及最终 white/pial 自相交；native placement 下，sphere/sphere.reg 局部翻折和 white↔pial 穿越不属于这一门的通过条件，必须在事后评估中另报。生产门通过不能代替局部质量、严格复现或整体指标等效验收。


## 4. 对应原软件调用

```bash
mris_fix_topology_fnit -ga -seed 1234 -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh SUBJECT lh
mris_remove_intersection SUBJECT/surf/lh.orig.premesh SUBJECT/surf/lh.orig.native-preclean.candidate
mris_remesh -i SUBJECT/surf/lh.orig.native-preclean.candidate -o SUBJECT/surf/lh.orig.native-remesh.candidate --remesh --iters 3
mris_remove_intersection SUBJECT/surf/lh.orig.native-remesh.candidate SUBJECT/surf/lh.orig.native-intersection.candidate
mris_place_surface --pial --lh --i SUBJECT/surf/lh.white --o SUBJECT/surf/lh.pial.T1
mrisp_paint -a 5 ATLAS#6 SUBJECT/surf/lh.sphere.reg SUBJECT/surf/lh.avg_curv
```

上述三步结束后，FNIT 独立验证最终私有候选，再原子发布 `orig`。这些是命令结构示意；实际 white/pial 命令还包含分割、标签、强度阈值及固定参数，详见[Conda 阶段](CONDA_CPP_STAGES.md)和[表面放置](PYTHON_PIAL_PLACEMENT.md)。FNIT 只执行已声明的 Conda 独立构建程序，不调用系统预装的 FreeSurfer。

## 5. 真实数据精度、耗时与脑图

截至本页创建时，**新增的 preclean + 原生 remesh + postclean 配方尚未完成原始 T1 整例**，因此 2,000 秒目标、两例精度、显存和整体指标等效均为未评估。现有同环境上游诊断显示：纯 Python GA 的 sub-06 LH／sub-07 RH 为 17,145.582／9,671.259 秒；相同 FNIT 自产五项输入交给独立 Conda C++ 到 `orig.premesh` 为 146.488／318.226 秒。阶段范围不同，不能把比值当整例加速。仅将 GA 改为 C++、继续使用 Python remesh/清理的上一个混合配方，sub-06 完整墙钟 5,349.477 秒、138/138 项和网格门通过；sub-07 清理残留 11／11 面后停止。原生相交清理在该失败输入上同样残留 11／11 面，因此新的完整配方必须在真实数据重新验收。数值和误差原因见[版本绑定的运行诊断](../../validation/recon_all/optimizations/20261016_topology_ga_runtime_debug/README.md)。

既有完整 T1 表面叠加[图示](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/whole/sub01/figures/t1_surface_overlay.png)属于旧版；不能当作本次原生收尾配方的结果。本轮验收后应另生成当前版本的表面、误差和局部异常图，并将新报告链接到此处。

## 6. 更新和 benchmark 记录

- 2026-10-16：增加显式 `topology_finish_backend="conda-cpp"` 接线；preclean、remesh3、postclean 均使用私有候选，结构与双谓词零相交门通过后原子发布。新配方的真实同输入、整例、GPU 归属和主分支发布仍须按新冻结源码报告。
- 2026-10-16 旧冻结配方：仅切换 C++ GA 的两例记录及 sub-07 失败重放见[拓扑运行诊断](../../validation/recon_all/optimizations/20261016_topology_ga_runtime_debug/README.md)。不能把失败前的 976.856 秒称为完整运行时间。
- 更早 remesh、相交的同输入逐位和耗时记录见[remesh 说明](REMESH_SCALAR_STORAGE_20261010.md)与[相交清理说明](INTERSECTION_REPAIR.md)。

## 7. 源码和参考

FNIT 调度与质量门：[native_free.py](../../src/fnit/recon_all/native_free.py)；自有可回退实现：[mris_remesh_python.py](../../src/fnit/recon_all/mris_remesh_python.py)、[mris_remove_intersection_python.py](../../src/fnit/recon_all/mris_remove_intersection_python.py)。原软件固定提交 `d932c45b7941662ea380a05efef580568b98d41a`：[FreeSurfer mris_remesh](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_remesh/mris_remesh.cpp)、[FreeSurfer mris_remove_intersection](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_remove_intersection/mris_remove_intersection.cpp)。原始方法见 Fischl et al., *Cerebral Cortex* 2004（自动拓扑修复）及 Dale et al., *NeuroImage* 1999（皮层表面重建）。
