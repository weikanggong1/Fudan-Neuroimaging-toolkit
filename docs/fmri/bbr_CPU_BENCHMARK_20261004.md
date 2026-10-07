# BBR CPU 官方 benchmark（2026-10-04）

## 当前结果与输入输出

`register_bbr` 的输入、全部参数、Python 调用、正常保存、pipeline CLI、原 FSL 命令和参考文献见 [BBR 使用说明](bbr.md)。本轮固定同一真实 EPI、T1、白质分割与 normmi 初始化，保留默认完整 `bbr.sch` 搜索。冻结主仓库基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。

最新 CPU 候选 v4 的完整 4×4 矩阵、T1 网格影像、binary header、边界点、成本及阶段求值次数，在 1/8 线程、首次/热调用下，与冻结基线逐位相同。首次 API＋正常保存为 18.828/18.464 s，本轮官方重测完整进程为 40.817/40.589 s；GPU 完整数值回归及聚合已完成，当前记录为实际观察。默认使用 `grid_search=True`、`execution="batched"`、`candidate_batch_size=128`，没有减少优化步数、网格、点数或影像范围。

2026-10-05 的当前候选为 CPU v4：去掉逐边界点临时数组，小批成本调用使用有序串行内核；CPU 的 `execution="reference"` 也可使用相同有序成本内核，同时保留原搜索顺序。CUDA reference 与 batched 分支继续使用既有实现。v4 的 CPU focused 检查已完成：66 项通过、12 项跳过；详见 [检查报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/focused_suite_v4.public.json)。默认六项中的 BBR 首次/热调用已通过完整输出门槛，详见[完整默认报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/defaults_v4.public.json)。31 项真实功能 job 全部 exit 0；11 组旧新完整输出与计数精确相同，6 组官方完整精度比较和成功退出链均已通过独立门槛；GPU 完整 ABBA 的 12 个进程、36 次完整 API 全部退出 0，18 组完整输出精度门槛已通过，实际时钟与资源见下文。

| 真实输入/输出 | 结构 |
|---|---|
| EPI 参考 | 88×88×64，float32；来自完整 490 帧同源数据 |
| 目标 T1 | 162×215×180，float32 |
| WM 分割 | 与 T1 相同网格和 affine，float32 |
| 初始化 | EPI→T1 的 FLIRT scaled-mm 4×4，固定路径输入 |
| 返回及标准保存 | T1 网格 float32 3D EPI、完整矩阵及 RAS world 矩阵、成本、边界点、阶段钟和求值计数；NIfTI 与 `.mat` 正常保存 |

BBR 不包含场图/GDC 校正。`BBRResult.save` 使用现有分别保存 NIfTI/矩阵的合同，本轮核对正常完整文件；没有新增成对原子保存接口。

## 原软件与可运行复现

现场 FSL 为 6.0.7.22，安装的 FLIRT 包为 2111.4；FNIT vendor 算法来源记录为 2111.2，保留这个版本差异。安装程序 SHA-256 为 `5cffd6bd209f8cb0045fe477815efb71ffbd01f2a529666fc4c1ff30896e54c7`。原软件只供独立 benchmark，生产使用 FNIT 计算实现。

```bash
# 同一真实EPI/T1/WM/init，完整官方搜索和正常输出。
flirt -in "$EPI_REFERENCE" -ref "$T1_REFERENCE" \
  -wmseg "$T1_WHITE_MATTER_SEGMENTATION" -init "$EPI_TO_T1_INITIAL_MATRIX" \
  -dof 6 -cost bbr -schedule "$FSLDIR/etc/flirtsch/bbr.sch" \
  -out "$REGISTERED_EPI" -omat "$EPI_TO_T1_MATRIX"
```

独立 adapter 见 [benchmark_baseline.py](../../validation/fmri_cpu_20261004/task01_motion_bbr/benchmark_baseline.py)；清单保留私有原路径，例子使用占位目录。

```bash
# threads/cpu-list必须由实际资源分配填写；fnit与official使用同一组。
python benchmark_baseline.py --case-json /private/inputs.json --case bbr \
  --function bbr --backend fnit --source /frozen/src \
  --source-revision cc9402734faeba93b3a13c29932fa1392eaccf62 \
  --output-dir /private/new/bbr_cpu8 --threads 8 --cpu-list 8,12,16,20,24,28,32,40 \
  --cpu-lock /private/task01_motion_bbr.lock
```

CPU 实测在共享节点 nodecw8，Python 3.11.16、PyTorch 2.5.1、NumPy 1.26.4、Nibabel 5.4.2。两个预算的 CPU 亲和性分别是 `[8]`、`[8,12,16,20,24,28,32,40]`；OMP、OpenBLAS、MKL、NumExpr、Numba 与 PyTorch 均设置 1/8 线程。源码/输入核验和 API 使用同一任务锁。原软件实际线程数未单独采样；8 指允许的环境线程与亲和性预算。

## 完整精度与耗时

每次调用成本求值均为 2604，其中粗级 1674、细级 928；原求值顺序、停止条件和完整算法保持。首次/热调用是同一进程第一/第二次完整调用，不等同于清空系统缓存。

| CPU 预算 | v4 API＋正常保存 首次 / 热 | v4 导入＋header＋API＋保存 首次 / 热 | 本轮官方重测完整进程 |
|---|---:|---:|---:|
| 1 | 18.828 / 18.017 s | 21.215 / 18.017 s | 40.817 s |
| 8 | 18.464 / 17.869 s | 20.312 / 17.869 s | 40.589 s |

官方重测列已完成，固定同一 WM/init、算法和 CPU 组；六个完整官方默认案例的实际退出链及 10 组全输出精度均已核对，见[本轮重测报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/fresh_native_v4.public.json)。下面保留较早 v2 时钟，不替换其版本标签。

### v5 冻结源复测（2026-10-06，BBR 模块未改）

v5 只修改 MCFLIRT 的 CPU 成本内核，BBR 的两份模块 SHA 与 v4 相同。这一批默认 CPU1/8 首次和热调用的全部矩阵、影像、header、边界点、成本及阶段求值计数仍与 v4 精确相同，成本调用均为 2,604 次。正常文件检查另核对标准影像与矩阵齐全；见[默认 v5 报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/defaults_v5.public.json)。

| CPU 预算 | v5 API＋正常保存 首次 / 热（s） | v5 导入＋header＋API＋保存 首次 / 热（s） |
|---|---:|---:|
| 1 | 23.558 / 18.673 | 28.538 / 18.673 |
| 8 | 19.762 / 18.484 | 21.888 / 18.484 |

该批共享 CPU 调度状态与此前官方参照不同，时钟作为实际观察保留，未计算稳定速度比。MCFLIRT 的同核 CPU 时钟、调度等待及缓存加载诊断见[诊断报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/profile_v4v5_cpu1.public.json)，其记录不能替代 BBR 的正常时钟。

### 较早 CPU v2 记录

| CPU 预算 | 原 FLIRT 完整进程 | 旧版 API＋正常保存 首次 / 热 | v2 API＋正常保存 首次 / 热 | v2 导入＋header＋API＋保存 首次 / 热 |
|---|---:|---:|---:|---:|
| 1 | 41.063 s | 368.257 / 275.157 s | 45.642 / 42.485 s | 48.082 / 42.485 s |
| 8 | 41.353 s | 118.275 / 77.230 s | 43.022 / 45.327 s | 45.311 / 45.327 s |

原程序钟包含子进程启动、读取、算法、输出和退出；FNIT API 包含实际影像读入/解压，正常保存另测并计入本表。输入/源码哈希和未舍入矩阵证据保存均排除。FNIT 的完整应用列含导入/header，不含 Python interpreter startup；正式端到端验收仍需统一进程边界的重复配对。

全目标网格 `162×215×180 = 6,269,400` 个值的官方比较为：RMSE **5.111245**，nRMSE **0.000801142**，max **9391.400391**。nRMSE 为 RMSE / 官方影像 RMS。完整矩阵元素最大绝对差 **0.002272578**；旋转和平移的矩阵元素混合，不能直接解释为毫米位移。这是本轮重新运行官方与冻结 FNIT 的直接比较；v2 和最新 v4 的所有完整输出与冻结 FNIT exact，保留这批官方误差；本轮官方重测对首次/热、1/8 线程的全部输出再次核对，精度与这些指标一致，见[重测报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/fresh_native_v4.public.json)。

以真实去颅骨 T1 的正值区作为精度统计掩膜（不参与算法）时，1,394,970 个脑内值的 RMSE 为 **5.014068**、nRMSE 为 **0.000386847**、最大绝对差为 **79.676758**。全网格最大差位于脑外。把两套完整矩阵转换成 RAS world 后，在全部脑内网格计算逆变换采样位置的位移：均值 **0.002418 mm**、P95 **0.004000 mm**、最大 **0.004694 mm**。详见 [脑内及世界坐标统计](../../validation/fmri_cpu_20261004/task01_motion_bbr/bbr_official_precision_details.public.json)。

FNIT/官方的 affine 与存储 dtype 相同，binary header 有差异。BBR FNIT 的 `xyzt_units=0`（unknown），官方为 10（mm/sec），并有描述、旧 header 字段及 quaternion 最末位差异。v2、v4 与冻结旧版 header 逐位相同；该单位差异是成熟输出 helper 的现存行为，本轮没有在 CPU kernel 内改变它。跨模块修复须另记版本与完整 CPU/GPU 输出回归。

完整 exec/exit/SIGCHLD trace 确认两个预算的实际计算叶进程退出 0，安装包装器退出 255 原样保留。叶进程临时文件退出后清理，无法事后取得其文件 hash；安装 launcher、包版本、完整 trace 和完整产物分别绑定。[匿名报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/cpu_candidate_v2.public.json)记录了源 SHA、输入 SHA、全部精度、header 字段和每次时钟。

## 分步骤与功能覆盖

v4 阶段钟是完整 API 的内部 wall；阶段含依赖，不与整体钟重复相加。固定 init 的初始化读取约 0.0004–0.0006 s；以下数值保持完整默认搜索。

| v4 默认完整阶段 | CPU 1 首次 / 热 | CPU 8 首次 / 热 |
|---|---:|---:|
| 边界准备 | 1.918 / 1.385 s | 1.767 / 1.291 s |
| 粗级 BBR | 0.566 / 0.565 s | 0.563 / 0.565 s |
| 细级 BBR | 15.247 / 15.071 s | 15.187 / 15.083 s |
| 最终重采样 | 0.144 / 0.124 s | 0.130 / 0.120 s |
| 正常保存 | 0.949 / 0.868 s | 0.813 / 0.807 s |

下面为较早 v2 阶段钟，保留其历史版本。


| v2 默认完整阶段 | CPU 1 首次 / 热 | CPU 8 首次 / 热 |
|---|---:|---:|
| 边界准备 | 4.290 / 1.666 s | 1.935 / 1.288 s |
| 粗级 BBR | 1.016 / 0.975 s | 0.968 / 0.975 s |
| 细级 BBR | 38.972 / 38.820 s | 39.167 / 42.131 s |
| 最终重采样 | 0.477 / 0.132 s | 0.136 / 0.124 s |
| 正常保存 | 0.881 / 0.888 s | 0.813 / 0.805 s |

细级有依赖优化的成本调用仍是 CPU 主热点。CPU 专属平滑/成本融合保留 double 坐标与比值、float32 采样、各 tap 舍入、候选顺序及稳定最小值。

| 功能 | 本轮真实执行状态 |
|---|---|
| 路径输入、固定 path init、默认完整网格、batched128 | 完整 1/8 首次/热已执行 |
| `BBRResult` 正常影像/矩阵保存、header、成本计数、阶段钟 | 已执行 |
| 图像对象输入、数组 init、自动 FNIT FLIRT init | 31 项真实功能门槛已通过；11 组旧新完整输出精确相同，6 组官方精度与成功退出链已核对 |
| `execution="reference"`、batch1/其他 batch、`grid_search=False` | 31 项真实功能门槛已通过；11 组旧新完整输出精确相同，6 组官方精度与成功退出链已核对；不能用 no-grid 时钟替代默认速度 |
| 完整 CPU/GPU ABBA、独立进程树 GPU 显存峰值 | GPU 12-process/36-call 完整 ABBA 全部退出 0，18 组完整输出精度门槛通过；共享设备时钟见下文 |

本轮 BBR 数据只发布聚合统计。公开脑图需另使用许可允许的同输入 BBR 配对；MCFLIRT 的公开 180 帧图不能作为这个私有 BBR 案例的精度图。

## 最新 BBR 真实功能变体（v5 冻结源）

下面四组均为本轮完整真实 EPI/T1 网格、CPU 8 的独立调用；v5 冻结源中的 BBR 模块保持 v4 SHA，与冻结 FNIT 的完整影像、矩阵、binary header、边界点、成本和阶段求值次数精确相同。影像对象与数组 init 已覆盖前三组，自动初始化由成熟 FNIT FLIRT 执行。每组观测不能代替默认完整 grid 的速度。详见[最新功能报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/features_v5.public.json)；14 项候选新运行与 17 项此前实际参照逐文件绑定，31 项组合门槛通过。[v4 报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/features_v4.public.json)保留历史时钟。

| 模式 | v5 冻结源应用（s） | 冻结 FNIT 应用（s） | 完整比较 |
|---|---:|---:|---|
| `execution="reference"`、影像对象、数组 init | 24.051 | 20.288 | 全部相同 |
| `execution="batched"`、`candidate_batch_size=1` | 24.974 | 19.609 | 全部相同 |
| `grid_search=False`、影像对象、数组 init | 8.494 | 7.840 | 全部相同 |
| 自动 FNIT FLIRT 初始化 | 56.524 | 104.215 | 全部相同 |

这些参数功能与原 FLIRT CLI 的映射见覆盖表；FNIT reference/batch 的执行选择没有单独对应的原软件选项。本轮六个功能官方参照为完整 MCFLIRT 变体，BBR 的默认官方对照与同批重测分别记录。

## GPU 完整回归（v4）

同一 H100 UUID、TF32 和 20 GB cap，按三个案例分别执行 A1 基线、B1/B2 v4、A2 基线；每个独立进程为一次首次和两次热 API，共 12 个进程、36 次完整调用。18 组完整影像、矩阵、参数、适用的 RMS、binary header 和算法计数均精确相同；[GPU 完整报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/gpu_v4.public.json)记录全部逐项指标、时钟、源码与资源 SHA。

| 案例 / 调用 | 基线 A1 / A2（s） | v4 B1 / B2（s） | v4 / 基线均值比 |
|---|---:|---:|---:|
| MCFLIRT 180 / 首次 | 103.852 / 100.560 | 101.018 / 101.742 | 0.9919 |
| MCFLIRT 180 / 热 1 | 95.334 / 96.661 | 94.973 / 95.139 | 0.9902 |
| MCFLIRT 180 / 热 2 | 95.483 / 97.372 | 94.499 / 95.100 | 0.9831 |
| BBR / 首次 | 4.657 / 4.349 | 4.433 / 4.431 | 0.9843 |
| BBR / 热 1 | 3.242 / 3.243 | 3.379 / 3.244 | 1.0214 |
| BBR / 热 2 | 3.197 / 3.250 | 3.272 / 3.245 | 1.0109 |
| MCFLIRT 490 / 首次 | 294.289 / 177.530 | 294.653 / 290.165 | 1.2395 |
| MCFLIRT 490 / 热 1 | 291.189 / 170.318 | 287.008 / 174.017 | 0.9990 |
| MCFLIRT 490 / 热 2 | 291.080 / 162.203 | 285.958 / 167.685 | 1.0008 |

API 与正常保存计入表内，导入、interpreter、证据与 SHA 比较另计。490 帧首次基线在两进程之间从 294.289 s 变为 177.530 s，首次均值比为 1.2395；两个热调用均值比为 0.9990/1.0008。本次共享设备时钟存在明显运行间波动，冻结 Task01 控制只记录 owned 显存，没有逐调用整体利用率遥测；稳定 GPU 速度结论仍需要相同负载下的重复配对。

| 完整案例 | PyTorch allocation max（GB） | reservation max（GB） | owned 进程树实测 max（GB） |
|---|---:|---:|---:|
| MCFLIRT 180 | 0.211 | 0.283 | 1.351 |
| MCFLIRT 490 | 1.154 | 1.292 | 2.397 |
| BBR | 0.810 | 1.103 | 1.655 |

显存采用十进制 GB。allocator 是每次完整 API 的精确峰值；NVIDIA owned 进程树以 0.5 s 间隔采样，后者含 CUDA context，并分别保存监控 SHA。全部低于 20 GB。该 v4 回归保留其六份源码 SHA；CPU v5 只改变 MCFLIRT CPU 行内核，BBR 两份模块 SHA 保持 v4。最终 v5 已额外完成一次 CUDA MCFLIRT 180 帧的精确完整输出门槛，CPU 成本模块未导入，详见[GPU v5 报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/gpu_v5.public.json)；本节 BBR GPU 时钟仍属于原 36-call ABBA。

## 版本与原实现

| 日期/版本 | 内容 |
|---|---|
| 2026-10-06 v5 冻结源复测 | BBR 模块保持 v4 SHA；默认 1/8 首次/热、标准输出和四组功能变体完整精确回归通过，最新共享 CPU 原始时钟另列。最终改变仅在 MCFLIRT CPU 内核。 |
| 2026-10-05 CPU v4 | 去掉逐边界点数组和小调用线程池切换，CPU reference 使用同一有序成本。默认 1/8 首次/热完整输出及计数与基线精确一致，实际 API＋保存 18.828/18.464 s；31 项功能和六项官方重测门槛通过；GPU 完整数值回归及聚合已完成。 |
| 2026-10-04 CPU v2 | CPU 有序融合平滑/成本；完整输出保持冻结版本，速度接近官方，尚未通过最终性能目标。 |
| `cc940273` | 本轮冻结主仓库基线，默认 BBR 原完整搜索。 |
| 2026-10-01 | CUDA 候选批量融合与真实完整配准链，见 [BBR 原报告](bbr.md#最新真实数据精度耗时与脑图)，不作为本轮 CPU 数据。 |

- [FSL BBR 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)、[FLIRT 源码](https://git.fmrib.ox.ac.uk/fsl/flirt)、[MISCMATHS 优化器](https://git.fmrib.ox.ac.uk/fsl/miscmaths)、[FSL 许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。
- Greve DN, Fischl B. *Accurate and robust brain image alignment using boundary-based registration*. NeuroImage 48:63–72, 2009. [DOI](https://doi.org/10.1016/j.neuroimage.2009.06.060)。
