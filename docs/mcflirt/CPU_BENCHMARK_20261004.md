# MCFLIRT CPU 官方 benchmark（2026-10-04）

## 当前结果

本轮保留完整公开 180 帧和真实 490 帧、外部参考、8/4/4 mm 三阶段、每阶段默认一次坐标轮回、最终样条及正常输出。冻结基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。CPU 候选 v2 的 10 组完整矩阵、参数、影像及 binary header 与基线逐位一致；CPU 速度目标尚未通过，8 线程 MCFLIRT 正式计时出现回退。v2 是待继续优化的测量版本。

函数介绍、完整参数和 Python/CLI 示例见 [MCFLIRT 使用说明](README.md)。API 的 `resample` 默认关闭；提供 `output` 会生成完整校正影像。此处速度组明确请求最终样条和正常写盘。

2026-10-05 的当前候选为 CPU v4：去掉逐体素临时数组，避免小成本调用在两个 OpenMP 线程池间切换，并将最终线性/样条重采样改为有序 CPU 内核。完整 180/490 帧的 helper 在 1/8 线程下逐值相同，见 [全帧重采样门槛](../../validation/fmri_cpu_20261004/task01_motion_bbr/sampling_full_gate.public.json)。v4 的 CPU focused 检查已完成：66 项通过、12 项跳过；详见 [检查报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/focused_suite_v4.public.json)。默认完整优化、正常文件和功能变体仍在独立队列中验证；下表保持已经完成的 v2 版本，不作为 v4 时钟。

## 输入、原软件与计时方法

| 项目 | 180 帧 | 490 帧 |
|---|---|---|
| 完整输入 | 64×64×42×180，int16，TR 2.1 s | 88×88×64×490，uint16，TR 0.735 s |
| 外部参考 | 保存的实际中间帧，float32 | 同源 3D 参考，float32 |
| 输出 | int16 校正图、180 个 MAT、180×6 `.par` | int32 校正图、490 个 MAT、490×6 `.par`、绝对/相对 RMS 及均值 |
| 精度范围 | 全部 30,965,760 个输出值 | 全部 242,851,840 个输出值；另报既有真实脑掩膜 |

公开 180 帧来自 [OpenNeuro ds001226 v5.0.1](https://openneuro.org/datasets/ds001226/versions/5.0.1)。现场原始 `dataset_description.json` 声明 CC0，SHA-256 为 `84185f9c387ae232ddbe87e20b329bacf82b9b5dafe9bde63b64c9bc78bed6f2`。私有数据只发布聚合统计。

现场 FSL 为 6.0.7.22，MCFLIRT 包为 2111.0；安装程序和来源版本的哈希保存在 [匿名报告](../../validation/fmri_cpu_20261004/task01_motion_bbr/cpu_candidate_v2.public.json)。官方参考仅在独立 benchmark 子进程调用，FNIT 运行使用自己的计算实现。

```bash
# 原完整程序；180帧和490帧均不截取时间或空间。
mcflirt -in "$RAW_BOLD" -reffile "$REFERENCE" -out "$OUTPUT_PREFIX" \
  -mats -plots -stages 3 -spline_final
# 490帧主组另保存全部标准RMS输出。
mcflirt -in "$RAW_BOLD" -reffile "$REFERENCE" -out "$OUTPUT_PREFIX" \
  -mats -plots -rmsrel -rmsabs -stages 3 -spline_final
```

每个进程设置 OMP、OpenBLAS、MKL、NumExpr、Numba 和 PyTorch 的 1/8 线程预算。1 线程亲和性为 CPU 8，8 线程为 8、12、16、20、24、28、32、40；同一任务锁覆盖输入/源码核验和 API。原软件实际线程数未单独采样，8 表示允许的线程与亲和性预算。

原软件计时包含 native 子进程启动、全部读取、优化、最终插值、标准文件和退出。FNIT API 包含完整影像读取与正常写盘；另记录导入、header 加载和额外未舍入证据保存。输入/源码哈希、精度分析和 `.npy` 证据保存均在 API 钟外。首次/热调用指同一进程的第一/第二次完整 API；首次不表示清空系统页缓存或 Numba 磁盘缓存。原软件与 FNIT 的进程边界不同，下面不宣布正式端到端达标。

## 完整默认速度

单位为秒。旧版/v2 的数值是完整 API，MCFLIRT 标准输出写盘已包含其中。180 帧各有一次热调用，490 帧当前各一次完整调用。

| 完整案例 / 预算 | 原 MCFLIRT 完整进程 | 旧版 API 首次 / 热 | v2 API 首次 / 热 | v2 导入＋header＋API＋正常保存首次 |
|---|---:|---:|---:|---:|
| 180 / 1 | 36.309 | 183.217 / 185.082 | 118.577 / 117.999 | 121.678 |
| 180 / 8 | 37.342 | 103.781 / 80.979 | 176.963 / 185.408 | 178.685 |
| 490 / 1 | 306.791 | 1173.808 / — | 700.207 / — | 702.783 |
| 490 / 8 | 307.427 | 434.132 / — | 647.394 / — | 650.683 |

完整 180/490 的 cost 求值分别为 17,456/45,908，旧版和 v2 相同。8 线程 v2 的正式结果慢于旧版；后续优化需重新做完整配对，不能用插桩 profile 的较短时钟覆盖这张表。

## 全输出精度

下表为冻结基线与本轮重新运行的原软件直接比较；v2 与该基线的完整输出逐位相同，所以保持同一官方误差。nRMSE 为 RMSE / 原软件影像 RMS。矩阵 max 是每个完整 4×4 系列元素的最大绝对差，旋转和平移混合，不能当作毫米位移。

| 案例 | 全网格 RMSE | 全网格 nRMSE | 全网格 max | 既有脑掩膜 RMSE | 矩阵 max |
|---|---:|---:|---:|---:|---:|
| 180 | 0.458274 | 0.00139156 | 353 | — | 0.0613768 |
| 490 | 4.297805 | 0.00116170 | 407 | 8.238399 | 0.0807060 |

180 帧 `.par` 的六列最大误差为旋转 `[0.000448533, 0.000683360, 0.000415321] rad`、平移 `[0.00872583, 0.0123806, 0.0127090] mm`；490 帧分别为 `[0.000712307, 0.000395251, 0.000482475] rad`、`[0.0116878, 0.0133450, 0.0143726] mm`。全帧 MAT、参数、RMS、输出有限值、形状和存储 dtype 均已核对；1/8 线程及已测热调用输出相同。

FNIT/官方 affine 完全相同，TR、空间/时间单位和存储 dtype 相同，binary header 有差异：FNIT 使用外部参考构造的 z pixdim，与官方保留输入值相差 `2.38×10⁻⁷ mm`；`dim_info`、`slice_code`、显示范围、描述与 quaternion 最末位亦不同。v2 与旧版 header 逐位相同。逐字段记录见匿名报告。

6 个本轮 MCFLIRT/BBR 官方参照均核对了完整成功 exec、计算叶进程退出 0 和 launcher SIGCHLD 链。外层安装包装器的 255 原样保留。其临时计算文件已在退出后清理，报告不声称事后取得该文件的 ELF/hash；安装程序哈希、包版本、实际完整 trace 与所有输出另外绑定。

## 分步骤 profile 与待验收功能

完整 180 帧的函数插桩只用于定位热点。时钟有嵌套关系，不能逐行相加，也不能替代正式速度。

| 完整 180 帧诊断 | CPU 1 | CPU 8 |
|---|---:|---:|
| cost 全部调用 | 74.080 | 74.705 |
| 其中 ordered CPU 行采样 | 71.727 | 72.316 |
| 最终重采样总计 | 29.506 | 28.313 |
| 其中三次样条采样 | 26.116 | 25.022 |
| 三次样条前滤波 | 1.390 | 1.350 |
| 正常 NIfTI 保存 | 1.569 | 1.749 |

完整默认 path/external/spline、490 RMS、冷/热与 1/8 预算已经执行。图像对象输入、中间帧参考、linear、1/2 阶段、非默认 iteration、只估计、CLI、overwrite/前缀和运动包装器参数的真实功能变体仍待独立队列；[覆盖矩阵](../../validation/fmri_cpu_20261004/task01_motion_bbr/PREPARATION.md#功能与覆盖计划)逐项列出受支持功能及原软件对应关系。GPU 完整 ABBA 由协调任务串行执行，此页尚无本轮 GPU 性能结论。

## 版本与参考

| 日期/版本 | 记录 |
|---|---|
| 2026-10-04 CPU v2 | CPU 专属有序行采样，完整矩阵/参数/影像/header 保持旧版；速度尚未达标，8 线程 MCFLIRT 回退。 |
| `cc940273` | 本轮冻结主仓库基线；上述原版与基线数据均是此次完整实测。 |
| 2026-10-03 | 原无缓存 CUDA 修复、完整 180 帧回归见 [主说明](README.md#5-真实数据精度耗时与脑图)，不作为本轮 CPU 数据。 |

- [FSL MCFLIRT 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)、[2111.0 源码](https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc)、[FSL 许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。
- Jenkinson M, Bannister P, Brady JM, Smith SM. *Improved optimisation for the robust and accurate linear registration and motion correction of brain images*. NeuroImage 17:825–841, 2002. [DOI](https://doi.org/10.1006/nimg.2002.1132)。
