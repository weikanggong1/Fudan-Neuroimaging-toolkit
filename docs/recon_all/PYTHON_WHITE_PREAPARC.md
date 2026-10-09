# Python white.preaparc 四轮放置

## 1. 功能与流程

`place_white_preaparc()` 把已有白质首轮实现接成完整四轮实验接口。它从同一
组 MRI、灰白阈值和 `orig` 开始，执行初始平滑与相交修复、逐轮冻结顶点、
白质目标强度搜索、四轮顺序优化及最终相交修复。MRI、rip、边界搜索、
强度梯度、自斥力、目标函数和碰撞规则均复用已有模块，没有复制另一套
pial 优化器。

本接口显式选择实验路径，生产 recon-all 默认仍调用独立 Conda 构建组件。
它对应 `white.preaparc`；带 aparc、`rip-label` 和独立 `rip-surf` 的最终
`white` 是另一个调用分支，尚未由本接口完成。PyTorch 可处理 signed
averaging、normal/tangent spring 和两跳二次曲率；边界搜索、自斥力、
目标函数、有序 Gauss–Seidel 碰撞接受及相交清理仍在 CPU。它尚不是完整
纯 GPU 表面放置。

```mermaid
flowchart TD
    A[orig 与三张同网格 MRI、灰白阈值] --> B[平滑5次、初始相交修复]
    B --> C[首次 rip]
    C --> D[本轮 rip 与 white 目标强度重估]
    D --> E[目标强度平均5次、重置面积/SSE/RMS/步长]
    E --> F[强度、自斥力和固定网格正则梯度]
    F --> G[原有序碰撞接受和目标函数]
    G --> H{缩步终止或100次迭代}
    H -->|继续本轮| F
    H -->|下一轮| D
    H -->|完成第4轮| I[最终相交修复与实际写出]
```

四轮梯度平均次数是 **4/2/1/0**，sigma 是 **2/1/0.5/0.25 mm**，每轮至多
100 次迭代。拒绝试步达到缩步上限时恢复本步起点并正常结束该轮；后续轮
重新计算冻结集合、目标强度和初始 SSE/RMS。固定网格邻接只创建一次；
rip 改变时重建 PyTorch 正则上下文。初始参考坐标始终保持在起始清理之后
的状态，不随轮间更新变化。最后清理复用已实现的完整 soap-bubble 修复，
残余相交会抛异常。

依赖均已在主页 Conda 环境中声明：PyTorch、NumPy、Numba、SciPy 和
nibabel。函数自身不执行 FreeSurfer 或其他外部程序，不自动启用 FP16/BF16，
也不改变调用方的全局 TF32 策略。

## 2. Python 调用、输入与输出

```python
from pathlib import Path
from fnit.recon_all.place_white_preaparc_python import place_white_preaparc

white_report = place_white_preaparc(
    subject_dir=Path("/data/fnit/sub-07"),  # 下表五项前置文件的目录
    hemi="lh",  # 左侧；右侧指定 rh
    output=Path("/data/diagnostic/sub-07/lh.white.preaparc"),  # 独立实验表面
    max_steps=400,  # 四轮总迭代保护上限；每轮仍至多100步
    output_volume=Path("/data/diagnostic/sub-07/mrisps.wpa.lh.mgz"),  # 可选预处理MRI
    regularization_backend="torch",  # 显式接入已有 PyTorch 固定网格正则梯度
    candidate_backend="tree",  # 原完整动态候选与有序接受；默认保留
    device="cuda:0",  # 当前进程内明确的目标 GPU
    trace_callback=None,  # 可选每步诊断回调；坐标和记录是独立副本
)
```

### 五项前置文件

| 文件 | 数据结构、空间和含义 |
|---|---|
| `surf/H.orig` | FreeSurfer 三角表面：`(N,3)` 坐标、`(F,3)` 有序面和体积几何尾部；坐标为 surface RAS，单位 mm；要求已修复拓扑 |
| `surf/autodet.gw.stats.H.dat` | 文本键值；至少包含 `MID_GRAY` 与 `white_inside_hi/border_hi/border_low/outside_low/outside_hi`，用于白质强度准备和边界搜索 |
| `mri/brain.finalsurfs.mgz` | 三维放置强度图；conformed MRI 网格，强度经既有规则转为 uint8 |
| `mri/wm.mgz` | 同一网格的 WM 编辑图；用于白质强度准备，保留 255 恢复语义 |
| `mri/aseg.presurf.mgz` | 同一网格的离散解剖标签；用于白质 midline/BG rip 与边界限制 |

三张 MRI 的 shape 和 affine 必须相同；表面尾部提供从 surface RAS 到 MRI
体素坐标的变换。`orig.premesh` 属于上游灰白阈值生成输入，本函数读取
冻结阈值，不再重复生成阈值。生产数据链必须使用 FNIT 自产前置文件；
官方上游交叉输入只用于独立诊断目录。

### 全部参数

| 参数 | 默认值与限制 |
|---|---|
| `subject_dir` | 必填，包含上述五项前置文件的目录 |
| `hemi` | 必填，`"lh"` 或 `"rh"`；决定半球标签规则 |
| `output` | 必填，独立 FreeSurfer 三角表面路径；禁止覆盖任一输入 |
| `max_steps` | 400，正整数；达到上限但未完成四轮时抛异常 |
| `output_volume` | `None`；指定时写 uint8 MGZ，保留 `brain.finalsurfs` 的 shape、affine、体素大小；不能覆盖输入或表面输出 |
| `regularization_backend` | `"cpu"`；`"torch"` 使用已实现的固定网格正则上下文，并要求明确 `device` |
| `candidate_backend` | `"tree"`；`"snapshot"` 和 `"torch_snapshot"` 是保守预候选实验，仍保留实时顺序窄相接受；后者要求明确 `device` |
| `device` | `None`；PyTorch 后端须显式指定如 `"cuda:0"` 或 `"cpu"`，不静默回退 CPU |
| `trace_callback` | `None`；接收 `(step, pass_index, vertices_copy, record_copy)`；记录包含实际试步、SSE/RMS、接受/拒绝和步长；回调耗时计入墙钟 |

表面输出保留输入顶点数、有序面与几何尾部，只替换坐标。返回 `dict`：

| 字段 | 结构与单位 |
|---|---|
| `output/output_volume` | 实际输出路径；未指定体积时为 `None` |
| `hemisphere/regularization_backend/candidate_backend/device` | 实际选择 |
| `complete_four_passes` | 成功返回时为 `True`；仅表示四轮计算完成，不表示官方数值验收通过 |
| `vertices/faces/ripped_vertices/held_vertices/steps` | 网格大小、最终冻结和最近一次试步受阻顶点数、总迭代数 |
| `pass_ends` | 四个全局迭代终点 |
| `passes` | 四条轮级记录：平均次数、sigma、迭代数、冻结顶点数、初末 SSE/RMS 和停止原因 |
| `per_step` | 每步记录；`trial_trace` 保存每次试步的实际 dt、SSE/RMS、缩步、拒绝和终止决定 |
| `initial_sse/initial_rms/step_sse/step_rms` | 初始及最后一次接受坐标对应的目标值；SSE 为既有加权目标，RMS 为影像强度误差 |
| `initial_cleanup/cleanup` | 起始/最终相交修复记录：相交面数、平滑周期和实际修复信息 |
| `seconds/stage_seconds` | 函数墙钟及分项秒；包含校验、读取、准备、传输、梯度、碰撞、目标函数、轮间重估、清理、写出与回调；各项和为墙钟 |

失败抛异常，不返回伪造完成状态。缺文件、无效参数、MRI 网格不一致、
未完成四轮、残余相交、CUDA 不可用或 OOM 均保留原异常；算法失败前不写
表面。实际 I/O 失败可能留下部分输出，调用者应保留异常及运行日志。

旧 `place_white_preaparc_prefix()`/`first_white_preaparc_step()` 继续作为
首轮 1–17 步诊断，不执行完整清理，也不改为完整 white 输出。

## 3. 命令行调用

```bash
python -m fnit.recon_all.place_white_preaparc_python \
  /data/fnit/sub-07 lh /data/diagnostic/sub-07/lh.white.preaparc \
  --complete \
  --max-steps 400 \
  --output-volume /data/diagnostic/sub-07/mrisps.wpa.lh.mgz \
  --regularization-backend torch \
  --candidate-backend tree \
  --device cuda:0
```

三个位置参数依次是前置目录、半球和独立输出；其余参数对应 Python 参数表。
不加 `--complete` 保持原首轮诊断调用，使用 `--steps 1`–`17` 和可选
`--diagnostics` NPZ。首轮选项与完整四轮选项不能混用。

## 4. 原软件调用

固定 FreeSurfer 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 的
对应 `white.preaparc` 调用如下；只在隔离 benchmark 环境执行参考程序。

```bash
mris_place_surface \
  --adgws-in /data/frozen/sub-07/surf/autodet.gw.stats.lh.dat \
  --wm /data/frozen/sub-07/mri/wm.mgz --threads 4 \
  --invol /data/frozen/sub-07/mri/brain.finalsurfs.mgz --lh \
  --i /data/frozen/sub-07/surf/lh.orig \
  --o /data/reference/sub-07/lh.white.preaparc \
  --white --seg /data/frozen/sub-07/mri/aseg.presurf.mgz \
  --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof \
  --outvol /data/reference/sub-07/mrisps.wpa.lh.mgz
```

`--rip-bg-no-annot` 的 preaparc 分支不依赖 aparc，`--rip-bg-lof` 仅在
RequireAnnot 分支影响区域选择。最终 white 使用的 annotation、label 和
独立 rip surface 不能被此命令或本接口省略。

本轮现场读取并核对的源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `mris_make_surfaces/mris_place_surface.cpp` | `b57f6e18f2e3215fc99fb1a8720aa345459585e4618ed2b8fa2779c2ee0f9dcf` |
| `utils/mrisurf_mri.cpp` | `16740fce04268f8d4e7841592acf3ff89e394bf211e6c0ab0a7b541527a7c092` |

## 5. 当前精度、耗时与可视化

**完整四轮真实数据 benchmark 尚未完成，不能报告完整 white 提速或官方
等效。** 本轮本地 CPU 控制契约 7/7 通过，包括四轮平均次数/sigma、目标
与 SSE 重估、rip 掩膜更新、终止拒绝恢复、100 次每轮上限、未完成不写出、
保留有序面/几何以及输出体积空间。模拟契约不替代真实影像验收。

此前同输入 `sub-07` 左侧**首步**在 H100 上测得 CPU 68.221 s、PyTorch
62.148 s，坐标/有序面/接受决定无差异。记录绑定当次 `4939d41c` 加模块
SHA，不是本轮完整四轮结果，也不能推断本轮完整耗时。
[首步原报告](../../validation/recon_all/optimizations/20261009_placement_torch/white_prefix_sub07_lh_final_v2_gpu0.json)
和[已实现梯度的分项结果](PYTORCH_PLACEMENT_REGULARIZATION.md)继续作为
组件证据。

复现脚本
[benchmark_placement_full_white.py](../../validation/recon_all/python_gpu_port/benchmark_placement_full_white.py)
固定五项输入 SHA，绑定实际导入源码与脚本 SHA、线程、主机、TF32 和显式
GPU，逐步保存坐标摘要及决策。CPU/PyTorch 完整几何与轨迹分开比较；可选
官方与 Conda 程序在相同五项输入重跑，默认参考重复两次。参考输出不会被
候选读取。

```bash
# 四线程预算必须在启动 NumPy/Numba 之前设置；输出目录必须尚不存在。
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python validation/recon_all/python_gpu_port/benchmark_placement_full_white.py \
  --subject /data/frozen/sub-07 \
  --candidate-directory /data/frozen-code/src/fnit/recon_all \
  --output-directory /data/runs/white-sub07-lh-v1 \
  --code-base-commit ACTUAL_TESTED_COMMIT \
  --hemisphere lh --backends cpu torch --device cuda:0 --threads 4 \
  --max-steps 400 --candidate-backend tree \
  --official-binary /data/reference-bin/mris_place_surface \
  --conda-binary /data/conda-bin/mris_place_surface \
  --assets-directory /data/declared-assets --reference-repeat 2
```

脚本包含坐标哈希和增量报告写入的实际墙钟；GPU 测量前后同步明确目标
设备。allocated/reserved 仅是 PyTorch 分配器峰值，整个进程及并发进程
同时显存仍需外部采样，不能据分配器值宣布满足 20,000,000,000 字节整例
预算。完整 brain overlay/异常区域图待真实四轮结果生成；此时不附模拟
脑图冒充当前 benchmark。不同硬件的历史记录分开报告。

## 6. 更新记录

| 日期/版本 | 修改与证据 |
|---|---|
| 2026-10-09，本轮实验接线 | 接通 preaparc 四轮、初始/最终清理、轮间冻结与目标重估、完整试步记录、uint8 诊断体积；7 项控制契约通过；真实四轮待测 |
| 2026-10-09，`4939d41c` 及模块 SHA | 首步 PyTorch 正则同输入无新差异；仅首步阶段证据 |
| 既有白质首轮诊断 | 1–17 步对照接口保留，承担定位参考作用；没有删除仍使用的诊断算子 |

完整四轮同输入回归通过之后，才考虑生产接线；最终 white 的独立 rip 与
annotation 分支仍需实现、同输入验证及两例双侧复核。严格复现、优化新
退化和整例指标等效分别报告，不降低既有门槛。

## 7. 参考文献与源码

- [固定版本 mris_place_surface.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_make_surfaces/mris_place_surface.cpp)：外层白质调度、rip 与命令参数。
- [固定版本 mrisurf_mri.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_mri.cpp)：边界搜索和放置内部步骤；这些函数没有独立 CLI。
- [固定版本 mrisurf_deform.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_deform.cpp)：MRISpositionSurface 与相交清理。
- Dale, Fischl and Sereno. Cortical surface-based analysis. I. Segmentation and surface reconstruction. *NeuroImage* 9, 179–194 (1999). [DOI](https://doi.org/10.1006/nimg.1998.0395)。
- Fischl and Dale. Measuring the thickness of the human cerebral cortex from magnetic resonance images. *PNAS* 97, 11050–11055 (2000). [DOI](https://doi.org/10.1073/pnas.200033797)。
