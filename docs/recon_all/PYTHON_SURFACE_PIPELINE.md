# 不执行原版阶段程序的完整 Python 候选

## 1. 功能简介

`surface_placement_python` 将已有完整四轮优化器接入 recon-all，生成 `white.preaparc`、最终 `white` 和 `pial.T1`，不运行 `mris_place_surface`。允许 Python、NumPy、SciPy、Numba 和 PyTorch；有序 CPU 计算继续复用，CUDA 计算完整候选和相交标记。`backend="python"` 另提供不执行原版阶段程序的完整候选配方；完整数字拓扑与原 GA 不同，整例输出、形状和指标仍须单独验证。

```mermaid
flowchart LR
  R[原始单 T1] --> V[conform / SimpleITK N4 / Torch GCA]
  V --> W[归一化 / WM / Numba 编辑 / filled]
  W --> A[完整数字拓扑 / remesh / orig]
  A --> B[阈值表 / 完整 white.preaparc]
  B --> C[平滑 / inflation / sphere.reg / 注释]
  C --> D[完整最终 white]
  D --> E[完整 pial.T1 / 清理]
  E --> F[厚度 / 面积 / 曲率 / 脑区统计]
```

## 2. Python 调用和全部输入输出

```python
from fnit.recon_all.surface_placement_python import (
    run_preaparc_python, run_final_white_python, run_pial_python,
)
preaparc_report = run_preaparc_python(
    subject_dir="/data/subjects/sub01",  # 已有自产 MRI、orig 和 orig.premesh 的目录
    hemi="lh",  # 左半球；右侧为 rh
    device="cuda:0",  # 明确设备，也允许 cpu，不自动回退
)
# 完成 cortex 标签、球面配准和 aparc 注释后，再执行最终放置。
white_report = run_final_white_python(
    subject_dir="/data/subjects/sub01",  # 同一被试的自产前置文件
    hemi="lh",  # 标签、注释和表面顶点顺序一致
    device="cuda:0",  # CUDA 候选；有序优化和 MRI 采样复用 CPU
)
pial_report = run_pial_python(
    subject_dir="/data/subjects/sub01",  # 已生成最终 white
    hemi="lh",  # 与 white 同侧
    device="cuda:0",  # 不启用半精度，不改调用方 TF32 策略
    profile=False,  # 默认不增加剖析屏障；True 同步明确 GPU 并记录子段
)
```

整例候选从原始 T1 和空目录开始：

```python
from fnit.recon_all.native_free import run_recon_all_python

reconstruction_report = run_recon_all_python(
    t1="/data/sub01_T1w.nii.gz",  # 原始三维单幅 T1w NIfTI
    subject_dir="/data/subjects/sub01",  # 不存在或为空的输出目录
    weights_dir="/data/fnit-weights",  # 已校验的项目神经网络权重
    assets_dir="/data/fnit-assets",  # 已校验的图谱、模板和 LUT
    device="cuda:0",  # 当前完整 WM/MNI 接口需显式 GPU
    threads=4,  # 整例总 CPU 预算，双侧 worker 各两线程
    backend="python",  # 固定完整独立候选；不读取原版程序目录
    n4_backend="simpleitk",  # 默认完整CPU N4；可显式选已有完整torch实现
    n4_execution="in-process",  # 默认父进程；torch可选isolated缓存worker
    input_execution="serial",  # 默认串行；parallel用1+3线程并行完整N4/SynthSeg
    profile_stages=False,  # 生产不增加剖析同步；测评设 True
    cuda_allocator_cache="auto",  # 保留已初始化 API 策略及低显存措施
)
```

完整配方默认 SimpleITK N4，显式 `n4_backend="torch"` 复用已有完整 200 轮 CUDA N4；两者均不调用原版阶段程序。单例、batch 与 benchmark 保留该选择，`n4_execution="isolated"` 仅支持 Torch，`input_execution="parallel"` 仅支持 SimpleITK/in-process；非法组合在创建输出前失败。选择、误差和实际验证见 [N4 后端选择](PYTHON_PROFILE_N4_SELECTION.md)。其余固定 Torch GCA/WM、Numba WM 编辑/filled/remesh/Gibbs、完整数字拓扑、标准 Torch inflation 与已有完整 white/pial、paint、curv.stats。双侧 worker 固定为 2，要求 `threads>=2`。CPU 核允许保留，默认 TF32 和既有 FP32 例外保持，不自动半精度。安装依赖 `SimpleITK==2.5.6` 已加入主页 Conda 与 pip，实际新环境安装验收另报。

配方修订 v4 还复用已有 GPU controls/initial-bias、标准 sphere 完整收尾、MNI 后置并行，以及 GCA 隔离 exec、1024 候选分块和同公式 GPU 求逆。仅隔离子阶段启用缓存，不改变父进程低显存策略；算法候选和顺序反馈保留。这些成熟后端此前在 `79a41cdd` 原始整例使用，本轮相同输入回归和修订整例另报，不能把历史耗时当作 v4 实测。v3 漏用这些选项的问题保留在原收据中。

后续调度将 Python 最终放置分为 `final_white` 和 `pial_surface` 两个双侧 exec 组，仍依次复用完整自有函数。仅 final white 新子进程启用缓存；pial 继续继承低显存策略，指标仍在双侧 pial 发布后串行。父 CUDA 和存活张量保持，`mrisps.white.mgz` 仍按右侧最后发布；没有给尚未完成缓存验证的 pial 全局开缓存。实际缓存配对与范围见 [表面分配器验证](SURFACE_CUDA_ALLOCATOR.md)；该拆分当时没有原始 T1 整例结果，后续 cbaffa 两例整例及完成的 v2 事后评估见[两例报告](../../validation/recon_all/optimizations/20261016_accelerator-a-1_cbaffa_two_case/README.md)。

数字拓扑输出 `mri/filled-topology-H.mgz`、`surf/H.orig.premesh` 及原 `orig.nofix` 同顶点序的真实删除区 `H.defect_labels`。标签是新算法切断区域，不声称与 GA 缺陷编号一致；原 `filled`、`orig.nofix` 和 nofix 链保留，继续完整 remesh、严格相交清理、自产 cortex/gyrus ambiens 标签和所有下游阶段。双侧新 MRI 仅发布本侧文件，共享缺陷体积仍按 lh/rh 顺序累计。

| 输入 | 格式、空间和用途 |
|---|---|
| `subject_dir` | 必填路径，含 `mri/`、`surf/`、`label/`；输出写入该目录 |
| `hemi` | 必填 `lh` 或 `rh`；共享文件仍由 pipeline 按确定顺序发布 |
| `device` | 必填 `cpu` 或带索引的 `cuda:N`；非法设备或 CUDA 不可用时报错 |
| `profile` | 仅 pial 接口；默认 `False`；诊断增加目标设备同步 |
| 三张 MRI | `brain.finalsurfs.mgz`、`wm.mgz`、`aseg.presurf.mgz`，同一 conform 网格，后两者保留离散标签语义 |
| preaparc 的网格 | `H.orig` 为优化输入，`H.orig.premesh` 用于阈值估计；`(N,3)` 坐标为 surface RAS/mm，`(F,3)` 有序面 |
| 最终 white 的输入 | `H.white.preaparc`、`autodet.gw.stats.H.dat`、`H.cortex.label`、`H.aparc.annot` 和三张 MRI；label/annot 对应同一顶点顺序 |
| pial 的输入 | `H.white`、阈值表、`H.cortex.label`、`H.cortex+hipamyg.label` 和三张 MRI；固定内侧壁 |

三个输出表面保持输入的顶点数、有序三角面和 surface RAS/mm 坐标。preaparc 另写阈值表和 `mri/mrisps.wpa.mgz`；最终 white 另写 `mri/mrisps.white.mgz`；这些是 uint8 预处理强度诊断图，保留 MRI 网格，并非分割。pial 写 `surf/H.pial.T1`。双侧并行使用私有目录，不能直接同时覆盖共享 MRI。

返回可序列化字典：路径、四轮终点、接受步数、清理统计、后端及秒数，另含 `implementation="fnit-python-numba-torch"`、`native_programs=[]`。阶段报告写入 `scripts/H.{white.preaparc,white,pial}.python.json`。阶段秒数包含影像读写和网格检查；最终报告写入由外层整例计时覆盖。失败抛异常，可能留诊断文件，不回退原版程序。

每个完整优化器最多 400 步，未完成四轮、文件缺失/非有限、顶点顺序改变或残余相交均报错。CUDA 使用 `torch_snapshot`、三单元候选网格、`source_torch` 清理；CPU 使用 `snapshot`、两单元网格、`source_numba`。完整候选扩展、实时碰撞接受、拒绝试步恢复和有序更新保持；`compiled` 为已有 Numba 循环。算法参数见 [preaparc](PYTHON_WHITE_PREAPARC.md)、[最终 white](PYTHON_FINAL_WHITE.md)、[pial](PYTHON_PIAL_PLACEMENT.md)。

## 3. 命令行

```bash
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights \
  --assets-dir /data/fnit-assets \
  --native-bin-dir /data/reference-components/bin \
  --device cuda:0 --threads 4 --hemisphere-workers 2 \
  --surface-placement-backend python
```

`--surface-placement-backend` 当前默认 `native`；`python` 接入完整三个自有阶段，并让 CPU 顶点指标使用已有 PyTorch。batch API 同名参数原样传到每个被试。权重/资产须校验；其他未替代阶段仍需独立构建目录，不能称此命令为无原版程序的整例。

完整独立候选使用同一入口，省略原版程序目录。`--input-execution parallel` 可显式启用完整 N4/SynthSeg 并行，仍要求总四线程；默认串行，前段和整例收益分别验证：

```bash
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights \
  --assets-dir /data/fnit-assets \
  --device cuda:0 --threads 4 --backend python \
  --n4-backend simpleitk --n4-execution in-process
```

`--backend python` 固定上述完整配方，仍处于整例验证阶段；`native` 保留混合对照。`--topology-backend python` 可用于单独定位数字拓扑，但其余后端仍由混合参数选择。旧 `python-gpu` 是严格 GPU 迁移诊断，不代表本次允许 CPU Python 核的目标。

## 4. 原软件调用

固定 FreeSurfer 8.2 配方：preaparc 对应 `mris_place_surface --white --nsmooth 5 --rip-bg-no-annot --rip-bg`；最终 white 对应 `--white --nsmooth 0 --rip-label H.cortex.label --rip-surf H.white.preaparc --aparc H.aparc.annot`；pial 对应 `--pial --nsmooth 0 --rip-label H.cortex+hipamyg.label --pin-medial-wall H.cortex.label --repulse-surf H.white --white-surf H.white`。完整 MRI、阈值和恢复选项见上述专页及独立 benchmark 包装器。候选只读自产输入；参考只在隔离 benchmark 路径运行和比较。

## 5. 当前真实数据验证

2026-10-10 表面适配器 v2 冻结基线为 `54f681dcab210a22c2e6862b2854a7033b5a208d` 加报告中列明 SHA 的改动，28 项接线契约全部通过；这不是影像 benchmark。两例公开 T1 的双侧 preaparc/white/pial 共 12 个冻结同输入阶段全部完成，两个监督进程均 exit 0。

| 被试/阶段 | Python 完整秒 | 固定源码参考完整秒 | 同索引最大距离/mm | 诊断 MRI 不同体素 |
|---|---:|---:|---:|---:|
| sub06 LH preaparc | 249.740 | 216.820 | 0.027004374 | 0 |
| sub06 LH white | 193.157 | 197.780 | 0 | 0 |
| sub06 LH pial | 338.255 | 195.966 | 0 | 无此输出 |
| sub06 RH preaparc | 296.106 | 216.150 | 0 | 0 |
| sub06 RH white | 224.184 | 208.039 | 0 | 0 |
| sub06 RH pial | 279.018 | 220.601 | 0 | 无此输出 |
| sub07 LH preaparc | 370.334 | 271.546 | 0 | 0 |
| sub07 LH white | 197.007 | 151.259 | 0 | 0 |
| sub07 LH pial | 268.994 | 144.700 | 0 | 无此输出 |
| sub07 RH preaparc | 199.116 | 168.168 | 0 | 0 |
| sub07 RH white | 187.732 | 146.363 | 0 | 0 |
| sub07 RH pial | 299.767 | 193.238 | 0 | 无此输出 |

全部比较先确认顶点数与有序面一致；11 个阶段坐标完全一致，sub06 LH preaparc 有 1046 个坐标元素不同，P99 顶点距离为 0，阈值表字节一致。同输入参考重复坐标一致；候选前 27 步坐标一致，第 28 步入口的有效顶点坐标、法线、rip 标记和 sigma 相同。首差为一个顶点的原始边界目标，经五次平滑扩散到 93 个有效顶点，随后传播到坐标。

定位到成熟子函数 `place_surface_border._search` 的通用类型问题：外侧 1 mm 采样中的 `距离 + 1.0` 在 Numba 中提升为双精度，而原公式保持 float32 的逐级运算，改变了阈值分支和边界峰值选择。单因素诊断修正这两个表达式后，四轮有效目标、距离、标记和 sigma 与参考相同；完整几何回归独立报告，不改 TF32、投影公式或阈值。这不是 GPU 碰撞网格误差，也不归因随机性。上述 v2 仍保留修复前的结果；整体等效未判定。它是冻结同输入阶段，共享节点上的一次完整读写和同步观察，不能称为整例提速。

修复后 sub06 LH 的完整 37 步、四轮终点 `[17,27,34,37]` 均与已核实的同输入参考坐标一致，最终坐标和有序面无差异，诊断 MRI 的体素、几何、dtype 及阈值表字节相同，输入 SHA 未变。源码 SHA 为 `199cfd527493a31a8888df47622492a4faf68ee88ce41e57ae75e9c996640bd4`；包含阈值生成和 API 读写为 289.830 秒，共享负载下不是配对提速证据。四侧完整 preaparc 均已通过：sub06 LH/RH 为 289.830/243.209 秒，sub07 LH/RH 为 305.498/190.917 秒，逐步轨迹及最终坐标、有序面、诊断 MRI 与同输入参考一致。

共用这个修复内核的最终 white/pial 八个完整阶段也已完成：sub06 LH/RH white 为 202.051/225.909 秒、pial 为 408.191/301.054 秒；sub07 LH/RH white 为 161.376/175.071 秒、pial 为 355.356/281.216 秒。最终坐标、有序面与冻结同输入原生参考及旧候选无差，white 诊断 MRI 无差，输入 SHA 未变，完整相交清理通过。这八项复用已有带哈希的原生参考，并未新跑原生重复性实验。连同四侧 preaparc，新版 12 个冻结同输入阶段的最终几何均一致；这不替代原始 T1 整例验证，也不证明提速。精确版本、调用和范围见 [边界目标修复与完整回归](WHITE_BORDER_TARGET_PRECISION_20261010.md)。

完整 N4 SimpleITK 同输入 sub07/sub06 分别为 28/99 个 uint8 体素相差 1，参考自身重复无差；完整文件 API 为 186.382/185.290 秒，详见[完整 N4 Python 接口](N4_SIMPLEITK.md)。后续强度与网格传播须在原始整例检查。WM 修复/独立编辑见[两例报告](WM_NATIVE_INDEPENDENT.md)。完整统计四侧八图及文本逐字节一致见[完整曲率统计](FULL_CURVATURE_STATS_PYTHON_20261010.md)，paint 四侧逐字节一致见[图谱采样](PAINT_ATLAS_PYTHON.md)。这些结果分别属于自己的冻结输入，不能相加成为整例通过结论。

显式 Torch/isolated 接线后，两例新的原始 T1→filled 空目录连续链完成。sub06/sub07 API 墙钟为 373.256/360.458 秒；相同主机、亲和性、GPU与四线程的既有 SimpleITK 前段为 464.285/486.859 秒，这是跨版本和跨时段观察。两例 orig 与 SynthSeg/aseg 标签相同，N4 最大量化差为 1；后续 WM 主掩膜 Dice 为 0.991753/0.997255，filled 最小标签 Dice 为 0.995471/0.997227，sub06 LTA 有 4 项不同。严格复现未通过，不能把下游差异称为无影响尾差。完整 19 步耗时、14 图 max/P99、标签体积及脑图见 [本次 Torch N4 前段报告](../../validation/recon_all/optimizations/20261010_torch_n4_prefix_compare/README.md)；默认 SimpleITK 保持，整例表面、最终指标和十分钟目标尚未由这组前段验证证明。

数字拓扑 v8 用相同冻结 `filled-pretess` 完整运行四侧文件 API：92.351/88.282/65.956/67.698 秒。Euler 为 2、单连通、闭合、有向、无非流形边/顶点、无零面积面及无几何自相交均通过；删除前景体素分别为 6942/8245/6062/6982。真实删除区映射回原 `orig.nofix` 顶点，未生成占位缺陷图。它与 GA 网格没有同索引对应关系；候选到原 GA 的完整三角面距离最大为 7.910/3.585/12.341/13.532 mm，反向为 1.844/2.237/1.339/1.000 mm，存在局部形状变化，不能按浮点尾差处理或设为已验证默认。下游解剖指标另测，此处的结构通过不代表整体指标等效。

完整配方的本地接线、线程预算、双侧私有输出、可选程序路径和执行审计共 56 项测试通过。固定 v4 静态审核进一步核对 24 组阶段、71 条函数 SHA；所选路径未发现原版 FS 程序、阶段链接库或原版源码回退的消费，见 [独立迁移表](../../validation/recon_all/optimizations/20261010_python_profile_static/README.md)。这不是完整运行或隔离验收。原始整例源码由 `validation/recon_all/optimizations/20261010_python_profile/freeze_source.py` 捕获到不可变 tar：记录实际 HEAD、工作树差异、逐文件 SHA；不包含原版程序、影像、权重或许可证。整例 benchmark 自动审计 CLI 及继承的 Python 子进程启动，遇原版阶段程序立即失败；这不是操作系统文件/动态库隔离验收。

复现脚本 `validation/recon_all/python_gpu_port/benchmark_python_surface_adapters.py` 记录输入、源码和参考程序 SHA。每侧每阶段隔离输入、候选和参考，避免将 RH 末次覆盖的诊断 MRI 当作 LH 参考。当前脚本新生成一份参考；重复性证据另列。检查顶点对应后报告坐标与 MRI 误差，候选运行时拦截外部进程启动。明确 GPU 前后同步，计时含输出写入。Torch allocated/reserved 不是整个进程树显存，不能据此判定整例 20 GB 门。

早期原始 T1 空目录整例曾用 `43be4a5b0b0215c91865b2292a92244e03547f0a` 加实际工作树冻结 v3 启动两例：773 文件，archive SHA `23b15962c216794bb832b636680b4a06fe026b7854450e0d85044b9d6c411d99`，manifest SHA `ce01deee5ebcf2bac6339cdc28071591aa981155b7f7e8def0e6a870049dcbc2`。每例四线程、明确 GPU，未传原版程序目录，取消 FS_LICENSE/FREESURFER_HOME 环境值，只使用公开原始影像、自产中间结果和声明资产。这段是旧 v3 启动记录，不是当前完成结论；后续 v4 两例最终退出失败，见[完整最终收据](../../validation/recon_all/optimizations/20261010_python_profile/README.md)。整体指标等效未判定，十分钟目标未达到。此前实际结果和图见 [阶段及整例接入记录](TORCH_INTEGRATION_20261009.md)，保持其原版本。

v3 的慢配方使用进程内 GCA、CPU 求逆及 64 候选分块。本轮在其自产 sub06 `nu/brainmask` 上，隔离 CPU/64 完整阶段为 99.822/116.596 秒，隔离 Torch/1024 为 36.611/37.435 秒；四份 LTA 文件 SHA 相同。父进程已初始化 CUDA、关闭缓存及存活张量均保持，仅新子进程启用缓存。它验证修订配方的阶段输出和实际耗时，不将组合变化归因单一算子，也不外推整例提速。共享 GPU 的外部进程归属不能解析，进程树显存尚未验证。

### 原始 T1 连续前段 v4（实测，非官方整例比较）

同原始 T1 对照是已经完成的 FNIT `79a41cdd` 混合链，两例均从原始影像和空目录连续运行。它不是官方参考，也不是冻结官方上游的阶段测试。14 张 MRI 各自成对的 shape、dtype、affine、surface RAS 变换均一致；`rawavg` 在原始导入网格，其余在 256³ conform 网格；LTA 的 16 元素无差，所有体积的绝对差 P99 为 0。

| MRI 文件 | sub06 不同体素 / 最大差 | sub07 不同体素 / 最大差 |
|---|---:|---:|
| `rawavg.mgz` | 0 / 0 | 0 / 0 |
| `orig.mgz` | 0 / 0 | 0 / 0 |
| `tmp/nu0.mgz` | 99 / 1 | 28 / 1 |
| `nu.mgz` | 92 / 3 | 28 / 3 |
| `T1.mgz` | 1549 / 8 | 41 / 3 |
| `brainmask.mgz` | 1103 / 8 | 21 / 3 |
| `synthseg.rca.mgz` | 0 / 0 | 0 / 0 |
| `norm.mgz` | 1563 / 4 | 16 / 3 |
| `brain.mgz` | 13133 / 16 | 25 / 4 |
| `aseg.presurf.mgz` | 0 / 0 | 0 / 0 |
| `wm.seg.mgz` | 10162 / 230 | 40 / 230 |
| `wm.asegedit.mgz` | 7566 / 230 | 32 / 167 |
| `wm.mgz` | 7583 / 230 | 32 / 167 |
| `filled.mgz` | 485 / 255 | 8 / 255 |

两例 SynthSeg 与 `aseg.presurf` 全部非背景标签 Dice 为 1。`filled` 的标签 127/255 Dice 分别为 sub06 **0.999408/0.999644**、sub07 **0.999998/0.999985**。WM 非零前景 Dice 分别为 `wm.seg` 0.998699/0.999983，`wm.asegedit` 0.999459/0.999992，最终 `wm` 0.999452/0.999992。强度差和标签 Dice 分开报告，没有对标签值计算 Pearson r；sub06 的强度差会传播，是否影响最终指标尚未判定。

只读比较 v2 为 34.869/34.222 秒，含 14 张影像读入；这个时间不计入生产整例。v2 保留三份原样生产收据快照，SHA 绑定实际解析字节，避免运行中报告更新造成版本错配；与 v1 的全部 MRI 数值及 LTA 结果相同。公开版本保留数值、哈希与原始/公开 SHA 对照，见 `reports/continuous_v4/PUBLIC_EXPORT_MANIFEST.json`。

v4 sub06 已完成双侧表面生成，但左侧球面配准在非有限采样坐标处失败。回放定位到第一个接受更新后，第 3 次顶点法向调用出现 6 个严格零叉积；单位化产生 4 顶点、12 个非有限元素。原 `mrisNormalize` 仅在长度大于零时相除，自有 PyTorch 内核已修复该分支。真实状态回放中，其他 107766 个有限顶点保持逐位一致；已完成的正常 `79a41cdd` 同输入完整配准，两版 130679 个顶点、有序面及输出文件 SHA 相同。共享负载单配对的冷进程为 156.837/154.180 秒，不作提速结论。

修正角点后，在同一异常整例输入的后续面法向处再次发现零叉积：第 3 次面法向调用有两个法向、六个 NaN。这个函数的原公式不同：`V3_NORMALIZE` 在 `abs(length)<FLT_EPSILON` 时保留叉积，其余相除。新版分别保留两个原有判据，12 项内核测试通过。真实面状态的面积逐位不变，普通面法向逐位不变；278 个非零但小于原判据的面改为保留微小叉积，两个零面不再产生 NaN。这属于恢复原公式，并不声称全部原有限法向都不变。

第二版在异常 sub06 LH 完整执行结束：API 263.914 秒、冷进程 283.389 秒，107770 顶点及 215536 面均有限、Euler 为 2、边界/非流形/零面积为零；但最终仍有 **7507 个负径向面，球面质量未通过**。负面积合计 51.518 mm²、P99 为 0.116918 mm²、最大为 0.413423 mm²，不能归为无实质影响的尾差。正常 79 输入的第二版完整配准再次保持坐标、有序面及文件 SHA，冷进程为 154.484 秒。

保留 1001 项清理轨迹和失败性质，不把执行结束当作可验收配准。固定原源码的 sphere/register 调用者都明确指定 1000，另有 1000 次无进展停止；现有上限属于原语义，不能删掉或增加次数来提高通过数。最后保存态的 7507 与最后一步前 7501 分别报告。新增停止原因和最终负面积诊断不改变坐标或轨迹；Python 配方的标准 sphere 与 sphere.reg 在写出前严格要求无翻折，失败保存机器可读 `quality_failure`，不继续使用无效球面。参考通用接口默认停止语义保持。逐阶段成功、失败和实际版本见 [配准零叉积回归](../../validation/recon_all/optimizations/20261010_registration_zero_corner/README.md)。

全部 incident 法向抵消时的[五轮有序几何修复](NORMAL_PENDING_PYTHON.md)已另行实现，复用成熟 Numba 邻接和单位化，返回坐标及完整状态；尚未接入上述配准内核，该内核遇到全抵消仍明确报错。11 项合同通过，五个真实冻结状态的坐标、法向均无差异，但全部未触发非零 pending；不能据此解释或宣称修复 7507 个翻折。上层接入还须消费返回坐标、失效动态缓存，并验证其余度量步骤。

原 v4 失败产物保持不变：CLI 墙钟为 **1368.894 秒**，API 为 **1363.345 秒**，包含前置检查的完整 harness 为 **1377.794 秒**；监督 shell 为 1379.582 秒。以上是失败运行时间，不能称为完成整例或最终提速。保存的球面本身坐标有限、半径正常；有 6946 个负径向面和微小三角面，动态非有限值由独立回放定位，不用夹紧坐标掩盖问题。sub07 的最终 white/pial 组随后以残余相交失败，CLI11074.201秒、API11069.505秒、harness11083.228秒；左侧失败不属于OOM，右侧worker计算完成但组未发布。独立几何复检为32相交面/35顶点，终态与两次原生参考的坐标及有序面相同。两例都是失败运行，不能称为整例提速；[最终完整收据与逐阶段时间](../../validation/recon_all/optimizations/20261010_python_profile/README.md)保留原字节SHA及执行审计。

两例 N4 实际为 184.698/184.775 秒；GCA 为 36.171/36.143 秒。候选仅执行 Python 阶段，原版程序 SHA 清单为空；源码审查与启动审计不替代无预装软件环境的隔离验收。显存进程归属未解析，目标卡有其他任务；采样请求间隔 0.5 秒，但 sub06 实际最大间隔 11.925 秒，含查询超时。进程树峰值未知，不能将共享卡总占用或缓存关闭的统计 0 当作候选峰值。

公开入口和worker的质量失败接线已做本地CPU回归：30项测试及8项子测试通过，包含严格球面参数透传、写出前失败和机器可读诊断保留。完整[Junit与源码SHA收据](../../validation/recon_all/optimizations/20261010_python_profile/quality_wiring_contracts.json)不替代真实网格或整例结果。

后续补齐双侧组异常的传递：worker 的质量诊断此前只在组报告中保存，公开入口接收到 `HemisphereGroupError` 时不能读取它。现在 `quality_failure` 对双侧组返回 `stage_group` 及按 `lh/rh` 保存的 `hemispheres` 字典；仅包含真实失败 worker 的质量诊断，取消、启动和 I/O 错误不冒充网格质量失败。单侧异常的原有字段保持。新增三项回归与已有接线、阶段拆分测试共 23 项通过，实际源码和 JUnit 见[组质量失败收据](../../validation/recon_all/optimizations/20261010_python_profile/group_quality_contracts.json)。这项修改不改变影像计算或球面停止规则。

N4/SynthSeg并行已完成两例冻结同输入的完整组ABBA，输出无新增差异；调度以 `input_execution="parallel"` 接入，默认serial。API、CLI和batch均传递该参数。随后两例各完成一次串行/并行的原始T1→filled，14图解码数据、几何/dtype和LTA全部0差，逐标签Dice为1。sub07 API486.859→450.809秒，sub06为464.285→503.020秒，未建立稳定前段或整例加速。不能把阶段组缩短8.75%/11.82%当作整例提速；[原始前段报告](../../validation/recon_all/optimizations/20261010_input_n4_synthseg_prefix/README.md)保存全部收据、元数据字节差及脑图，参见[完整并行组](INPUT_N4_SYNTHSEG_PARALLEL.md)。

## 6. 最近更新和 benchmark

原拓扑算法的独立 Python 迁移已补齐 [MRI 似然及零面积诊断](TOPOLOGY_MRI_LIKELIHOOD.md)、[角度伪法向](TOPOLOGY_ANGLE_PSEUDONORMALS.md)、[完整有向距离及体积反馈](TOPOLOGY_SIGNED_DISTANCE.md)、[源精度 fitness 组合](TOPOLOGY_PATCH_FITNESS.md)，以及主曲率、局部 MRI/PDF、完整搜索和写回。早期参考候选冻结测试与后续自产搜索分别保留证据；完整首缺陷已由 FNIT 自产上游连续完成。安装包内[完整GA文件接口](TOPOLOGY_GENETIC_FILES.md)已通过文件和编号域合同，`topology_backend="python-ga"` 的[API/CLI/batch接线](PYTHON_GENETIC_PROFILE.md)保留显式选择、实际半球线程和质量检查。该段阶段记录写成时，全九缺陷及最终 orig 的真实验证仍在运行；后续 cbaffa [两例原始 T1 整例](../../validation/recon_all/optimizations/20261016_accelerator-a-1_cbaffa_two_case/README.md)已退出 0，138 项输出与生产网格门通过。两类验证分别绑定原版本；参考几何和中间卷未接入生产，早期失败的数字拓扑整例仍保留原判定。

- `3644f50c`：修复 float32 步长范数开方后提升顺序，完整 pial 冻结四侧几何一致。
- `6de797b4`：修复最终 white 标签分支的 midline 状态，冻结四侧与重复原版参考一致。
- 2026-10-10：新增生产适配器、API/CLI/batch 接线，跳过不再需要的 white 原版能力探测；本次接入验证独立保存。

## 7. 原实现和参考文献

- [FreeSurfer mris_place_surface 源码](https://github.com/freesurfer/freesurfer/tree/dev/mris_place_surface)，参考实际提交和程序 SHA 以收据为准。
- Dale AM, Fischl B, Sereno MI. Cortical surface-based analysis. I. Segmentation and surface reconstruction. *NeuroImage* 9, 179–194 (1999).
- Fischl B, Dale AM. Measuring the thickness of the human cerebral cortex from magnetic resonance images. *PNAS* 97, 11050–11055 (2000).
