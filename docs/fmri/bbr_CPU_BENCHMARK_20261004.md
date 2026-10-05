# BBR CPU 官方 benchmark（2026-10-04）

## 当前结果与输入输出

`register_bbr` 的输入、全部参数、Python 调用、正常保存、pipeline CLI、原 FSL 命令和参考文献见 [BBR 使用说明](bbr.md)。本轮固定同一真实 EPI、T1、白质分割与 normmi 初始化，保留默认完整 `bbr.sch` 搜索。冻结主仓库基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。

CPU 候选 v2 的完整 4×4 矩阵、T1 网格影像与 binary header 在 1/8 线程、首次/热调用下，与冻结基线逐位相同。速度已明显缩短，但当前完整 CPU 时钟仍长于官方；本页不宣布目标达成。默认使用 `grid_search=True`、`execution="batched"`、`candidate_batch_size=128`，没有减少优化步数、网格、点数或影像范围。

2026-10-05 的当前候选为 CPU v4：去掉逐边界点临时数组，小批成本调用使用有序串行内核；CPU 的 `execution="reference"` 也可使用相同有序成本内核，同时保留原搜索顺序。CUDA reference 与 batched 分支继续使用既有实现。v4 的 CPU focused 检查已完成：66 项通过、12 项跳过；详见 [检查报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/focused_suite_v4.public.json)。完整默认 API 和真实功能变体仍在验证；本页时钟与完整精度仍绑定已完成的 v2。

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

两个预算的 CPU 亲和性分别是 `[8]`、`[8,12,16,20,24,28,32,40]`；OMP、OpenBLAS、MKL、NumExpr、Numba 与 PyTorch 均设置 1/8 线程。源码/输入核验和 API 使用同一任务锁。原软件实际线程数未单独采样；8 指允许的环境线程与亲和性预算。

## 完整精度与耗时

每次调用成本求值均为 2604，其中粗级 1674、细级 928；原求值顺序、停止条件和完整算法保持。首次/热调用是同一进程第一/第二次完整调用，不等同于清空系统缓存。

| CPU 预算 | 原 FLIRT 完整进程 | 旧版 API＋正常保存 首次 / 热 | v2 API＋正常保存 首次 / 热 | v2 导入＋header＋API＋保存 首次 / 热 |
|---|---:|---:|---:|---:|
| 1 | 41.063 s | 368.257 / 275.157 s | 45.642 / 42.485 s | 48.082 / 42.485 s |
| 8 | 41.353 s | 118.275 / 77.230 s | 43.022 / 45.327 s | 45.311 / 45.327 s |

原程序钟包含子进程启动、读取、算法、输出和退出；FNIT API 包含实际影像读入/解压，正常保存另测并计入本表。输入/源码哈希和未舍入矩阵证据保存均排除。FNIT 的完整应用列含导入/header，不含 Python interpreter startup；正式端到端验收仍需统一进程边界的重复配对。

全目标网格 `162×215×180 = 6,269,400` 个值的官方比较为：RMSE **5.111245**，nRMSE **0.000801142**，max **9391.400391**。nRMSE 为 RMSE / 官方影像 RMS。完整矩阵元素最大绝对差 **0.002272578**；旋转和平移的矩阵元素混合，不能直接解释为毫米位移。这是本轮重新运行官方与冻结 FNIT 的直接比较；v2 的所有完整输出与冻结 FNIT exact，保留同一误差。

以真实去颅骨 T1 的正值区作为精度统计掩膜（不参与算法）时，1,394,970 个脑内值的 RMSE 为 **5.014068**、nRMSE 为 **0.000386847**、最大绝对差为 **79.676758**。全网格最大差位于脑外。把两套完整矩阵转换成 RAS world 后，在全部脑内网格计算逆变换采样位置的位移：均值 **0.002418 mm**、P95 **0.004000 mm**、最大 **0.004694 mm**。详见 [脑内及世界坐标统计](../../validation/fmri_cpu_20261004/task01_motion_bbr/bbr_official_precision_details.public.json)。

FNIT/官方的 affine 与存储 dtype 相同，binary header 有差异。BBR FNIT 的 `xyzt_units=0`（unknown），官方为 10（mm/sec），并有描述、旧 header 字段及 quaternion 最末位差异。v2 与冻结旧版 header 逐位相同；该单位差异是成熟输出 helper 的现存行为，本轮没有在 CPU kernel 内改变它。跨模块修复须另记版本与完整 CPU/GPU 输出回归。

完整 exec/exit/SIGCHLD trace 确认两个预算的实际计算叶进程退出 0，安装包装器退出 255 原样保留。叶进程临时文件退出后清理，无法事后取得其文件 hash；安装 launcher、包版本、完整 trace 和完整产物分别绑定。[匿名报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/cpu_candidate_v2.public.json)记录了源 SHA、输入 SHA、全部精度、header 字段和每次时钟。

## 分步骤与功能覆盖

v2 阶段钟是完整 API 的内部 wall；阶段含依赖，不与整体钟重复相加。

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
| 图像对象输入、数组 init、自动 FNIT FLIRT init | 真实功能队列待执行 |
| `execution="reference"`、batch1/其他 batch、`grid_search=False` | 真实功能队列待执行；不能用 no-grid 时钟替代默认速度 |
| 完整 CPU/GPU ABBA、独立进程树 GPU 显存峰值 | 协调任务安排中；此页没有新 GPU 性能结论 |

本轮 BBR 数据只发布聚合统计。公开脑图需另使用许可允许的同输入 BBR 配对；MCFLIRT 的公开 180 帧图不能作为这个私有 BBR 案例的精度图。

## 版本与原实现

| 日期/版本 | 内容 |
|---|---|
| 2026-10-04 CPU v2 | CPU 有序融合平滑/成本；完整输出保持冻结版本，速度接近官方，尚未通过最终性能目标。 |
| `cc940273` | 本轮冻结主仓库基线，默认 BBR 原完整搜索。 |
| 2026-10-01 | CUDA 候选批量融合与真实完整配准链，见 [BBR 原报告](bbr.md#最新真实数据精度耗时与脑图)，不作为本轮 CPU 数据。 |

- [FSL BBR 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)、[FLIRT 源码](https://git.fmrib.ox.ac.uk/fsl/flirt)、[MISCMATHS 优化器](https://git.fmrib.ox.ac.uk/fsl/miscmaths)、[FSL 许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。
- Greve DN, Fischl B. *Accurate and robust brain image alignment using boundary-based registration*. NeuroImage 48:63–72, 2009. [DOI](https://doi.org/10.1016/j.neuroimage.2009.06.060)。
