# MCFLIRT、运动包装与 BBR：功能覆盖与复现入口

本轮冻结基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。2026-10-04 已完成完整 180/490 帧、固定初始化 BBR 的原软件与冻结 FNIT 对照，以及 CPU v2 的完整输出比较和 profile。最新结果见 [MCFLIRT CPU 报告](../../../docs/mcflirt/CPU_BENCHMARK_20261004.md)与 [BBR CPU 报告](../../../docs/fmri/bbr_CPU_BENCHMARK_20261004.md)。

截至 2026-10-05，CPU v4 已在服务器完成六份源码、七份测试及控制文件的哈希核验；[focused suite](focused_suite_v4.public.json) 实测 66 项通过、12 项跳过。完整默认 API 和功能变体另行验证，GPU 完整 ABBA 已按固定设备提交。完整 180/490 的线性及样条重采样 helper 已在 1/8 线程下检查所有帧，逐值相同；该结果只验收 helper，不能替代 MCFLIRT 优化与正常输出整链。本页的历史核与排期数字保留其原版本，不改标为 v4 实测。GPU 回归由协调任务统一串行启动和验收。

## 功能与覆盖计划

主速度组保留完整 490 帧、相同参考、全部三阶段与默认每阶段一次轮回。功能组使用完整公开 180 帧，所有帧与原始空间网格保留。参数改变组表示对应受支持功能本身，不代替主速度组。

| 函数/入口 | 实际支持功能 | 官方对照与完整真实覆盖 |
| --- | --- | --- |
| `TorchMCFLIRT.run` / `__call__` | NIfTI 路径或 NiBabel NIfTI1/2；显式 CPU/CUDA；默认中间帧或同网格外部参考 | 490 帧外部参考主组；180 帧中间/外部参考组；路径/图像对象输出一致 |
| MCFLIRT 优化 | `stages=1/2/3`；`stage_iterations=(1,1,1)`；逐阶段增加轮回或用 0 跳过 | `mcflirt -stages 1/2/3` 配对；非默认 iteration 是 FNIT 的 API 功能，与冻结 FNIT 对照，不编造官方 CLI |
| MCFLIRT 重采样 | `resample=False/True`；最终 `linear/spline`；指定 output 强制重采样 | 官方默认三线性或 `-spline_final`；只估计组仅比较矩阵/参数，不能与官方完整输出墙钟混比 |
| MCFLIRT 输出 | `.nii.gz`、`.mat/MAT_####`、`.par`；`rmsrel/rmsabs` 与均值；前缀/`.nii`/`.nii.gz` 归一；overwrite | 同 `-mats -plots -rmsrel -rmsabs`；检查全帧数量、每项数值、dtype/affine/qform/sform/pixdim/TR；重复输出拒绝和显式 overwrite |
| MCFLIRT 类型 | 原 uint16→int32，公开 int16→int16；支持其他已声明实数 NIfTI dtype 转换 | 主真实两例覆盖真实类型；未在真实数据出现的类型以契约单元测试覆盖，不称真实 benchmark |
| `fnit mcflirt` | `-in/-out/-reffile/-mats/-plots/-rmsrel/-rmsabs/-stages/-spline_final/-trilinear_final/--device/--overwrite` | API/CLI 相同完整 180 帧参数与输出；官方对应命令见下 |
| `estimate_motion` | 显式 reference；`mask` 只验证参考网格；`batch_size` 兼容参数；三阶段 iterations；可选 corrected | 矩阵与同参 `TorchMCFLIRT`；包装器参数为原 pull convention，先通过 `matrices_to_mcflirt_parameters` 转成官方 `.par` 再比较，不能直接相减 |
| `matrices_to_mcflirt_parameters` | 全部 T×4×4 FLIRT 矩阵；3D 参考与强度重心；T×6 rad/mm | 官方同次 `.mat`/`.par` 全帧比较；参考有限、正总强度与矩阵形状失败契约 |
| `register_bbr` | 路径/3D影像；T1同网格 WM；init 文本/数组；init=None 自动 FNIT FLIRT；CPU/CUDA | 固定相同 WM/init 的 FSL 完整 BBR；自动 init 组与 FNIT 自产 FLIRT 同算法冻结基线比较，并另报官方链累计差异 |
| BBR schedule | `grid_search=True/False`；`execution=batched/reference`；candidate_batch_size | 主组保留默认完整两起点粗 729×2、微 729 及 Brent→Powell→Brent；reference/batch1/batch128 输出与完整停止流程一致；no-grid 是明确受支持功能，不能用其计时替代主组 |
| `BBRResult` / save | moved/FLIRT matrix/world matrix/cost/boundary数/阶段钟/计数；NIfTI与omat任选保存 | 官方 moved/mat；完整 T1 脑内逆场位移及全部输出体素；世界坐标与 scaled-mm 转换、save/reload 契约 |
| BBR CLI | 没有独立 BBR CLI；pipeline 有 `--bbr-execution` | 不编造独立入口；协调者整链验证覆盖 pipeline 接入 |

明确的当前范围：MCFLIRT 未实现 2D/小于20mm z FOV、meanvol、第四阶段 sinc 优化和其他 cost；BBR 为六自由度、无场图、固定距离/有符号成本。未实现模式不写成支持功能。

## 已核验资源与原软件版本

去标识现场头信息、输入与官方程序哈希见 [v2 匿名报告](cpu_candidate_v2.public.json)。原路径保留在服务器及协调任务的私有清单；临时文件不是持久数据入口。

- 完整真实 490 帧：88×88×64×490，原 uint16；外部参考、同网格脑掩膜、490 个官方矩阵、490 行参数、完整 int32 官方校正图齐全。
- 完整公开 180 帧：64×64×42×180，原 int16；已保存成熟 FEAT 第 90 帧参考。来源为 OpenNeuro ds001226 v5.0.1；本轮官方和 FNIT 完整输出已生成。公开脑图只能从许可允许的公开数据生成。
- BBR：同案例 EPI 88×88×64；T1/WM 162×215×180；固定 normmi init、官方 BBR matrix 与完整 moved 齐全。固定 WM 接口不需要权重或模板，不新增资产下载。自动链中 FAST/SynthStrip 等归协调者或相应任务，不能把官方 WM/init 读入生产链。
- 现场原软件为 FSL 6.0.7.22。MCFLIRT Conda 包 2111.0，和源码参照 tag 2111.0 相符；FLIRT 程序包 2111.4，FNIT vendor 实现参照 tag 2111.2，报告应保留该版本差异。
- FSL 程序仅用于独立官方 benchmark；生产 MCFLIRT/BBR 使用 FNIT 数组计算。已核对项目 vendor 原源码清单和 [官方许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。本轮不复制原软件程序或无关源码，不改变当前许可文件。

## 可运行 baseline adapter

`benchmark_baseline.py` 统一接受私有案例 JSON、冻结 src 和实际 revision，强制明确 1/8 线程、CPU 亲和性、协调锁、fresh输出目录。它支持 FNIT/官方 MCFLIRT、FNIT motion wrapper 及 FNIT/官方固定init BBR。私有 JSON 的通用结构是：

```json
{
  "fsl_root": "/private/fsl",
  "mc490": {"bold": "/private/bold.nii.gz", "reference": "/private/reference.nii.gz", "brain_mask": "/private/mask.nii.gz"},
  "mc180": {"bold": "/private/public_bold.nii.gz", "reference": "/private/public_reference.nii.gz"},
  "bbr": {"epi": "/private/epi.nii.gz", "t1": "/private/t1.nii.gz", "wmseg": "/private/wm.nii.gz", "init": "/private/init.mat"}
}
```

以下为复现模板；CPU 组、锁和输出目录须按现场分配填写。主速度组已经按私有清单实际执行，模板占位路径本身不属于运行证据。

```bash
# 完整490帧、默认三阶段、外部参考、原样条及全部正常文件。
python benchmark_baseline.py --case-json /private/inputs.json --case mc490 \
  --function mcflirt --backend fnit --source /frozen/src \
  --source-revision cc9402734faeba93b3a13c29932fa1392eaccf62 \
  --output-dir /private/new/fnit490 --threads 8 --cpu-list 0-7 \
  --cpu-lock /private/allocated.lock --rms --interpolation spline
# 同主机、同线程与CPU组、同输入的原程序；新的输出目录。
python benchmark_baseline.py --case-json /private/inputs.json --case mc490 \
  --function mcflirt --backend official --source /frozen/src \
  --source-revision cc9402734faeba93b3a13c29932fa1392eaccf62 \
  --output-dir /private/new/official490 --threads 8 --cpu-list 0-7 \
  --cpu-lock /private/allocated.lock --rms --interpolation spline
# 原完整bbr.sch固定WM/init，分别backend fnit/official，使用新目录。
python benchmark_baseline.py --case-json /private/inputs.json --case bbr \
  --function bbr --backend fnit --source /frozen/src \
  --source-revision cc9402734faeba93b3a13c29932fa1392eaccf62 \
  --output-dir /private/new/fnit_bbr --threads 8 --cpu-list 0-7 \
  --cpu-lock /private/allocated.lock
```

原官方调用分别为：

```bash
mcflirt -in "$RAW_BOLD" -reffile "$REFERENCE" -out "$PREFIX" \
  -mats -plots -rmsrel -rmsabs -stages 3 -spline_final
flirt -in "$EPI" -ref "$T1" -wmseg "$WMSEG" -init "$INIT" \
  -dof 6 -cost bbr -schedule "$FSLDIR/etc/flirtsch/bbr.sch" \
  -out "$PREFIX" -omat "$PREFIX.mat"
```

计时将分导入、完整 API、正常写盘、原 native 进程与额外未舍入证据保存。哈希和精度比较不计入算法/正常输出钟。adapter 的 FNIT application 计时不包含 Python interpreter startup，native进程包含程序启动；正式端到端达标应由协调者增加相同进程边界的配对控制，不能把不同时钟直接宣布达标。已有 FSL wrapper 有退出255/实际C++子进程退出0的记录；adapter 不自动把非零退出改成通过，必须另核验实际 child exit 与完整产物。

## 准备阶段的历史证据边界

`validation/mcflirt/full490_cpu.public.json` 是旧核 `f4f18579...`、4 线程、387.598 s，只估计、不最终采样、不写盘；cost 45,794。初始核 SHA 为 `2b344c34ecbc96d14a07132f6c5731a87ebe13178ab5fa1452222ed2299bae78`。`native_exact_latest.public.json` 的完整 490 官方样条/正常文件为 331.535 s，环境 8 线程；实际 C++ 子进程线程峰值未捕获。这两个历史数只曾用于排期，不用于本轮相同线程速度比较。

初始 BBR SHA 为 `0ca14f3343d6b5bf09ea93f6ec94de9f2c0cce19fdf8ea005a87ff8e03f48fc2`。`registration_gpu.current.public.json` 固定同 WM/init 的 GPU 首/热调用 3.581/1.409 s，官方 CPU 完整进程 45.093 s，计时范围和设备不同。CPU v2 的本轮完整同预算数据另见当前 BBR CPU 报告；这些历史 GPU 数字不充当本轮 CPU 预计或 GPU 回归。

## 具体优化热点与GPU保护边界

1. MCFLIRT CPU `FSLMotionNormCorr.__call__` 每个cost生成逐次float32行坐标、大量中间张量、手工8角采样；NCC归约遍历x/y/z并保留连续num/numA。可在CPU专属Numba helper融合独立行采样与原有顺序归约；保留坐标逐步舍入、严格NCC定义、时间传播、cost求值数和Brent停止规则。既有 `flirt/_cpu.py` 与 `_cpu_simd.py` 的float32舍入/线程预算工具可复用，但不能直接拿普通FLIRT cost替代MCFLIRT NCC。
2. BBR `_smooth_wm` 每个tap对完整T1体积进行float64转换、乘加，再缩窄float32；边界和27tap梯度也产生全体积/边界临时张量。CPU专属融合平滑/边界采样可降低内存流量，但每tap舍入、半径、padding和x最快边界枚举必须一致。
3. BBR `_BBRCost.evaluate` CPU每次产生候选×2×边界点×3 double坐标、8角临时数组与tanh；独立候选或点块可用CPU融合，保持double坐标/强度比、float32采样和稳定成本/最小值顺序。依赖的Brent/Powell不能并行改变顺序。
4. `estimate_motion` 后处理每帧4×4逆矩阵/参数转换可做批量NumPy，预期收益小；先量化再优化。

新增helper限device.type=='cpu'分支。GPU模块及调用图原则上保持；最终协调者对完整180/490 GPU候选/基线ABBA配对，核对矩阵/参数/影像/计数、峰值显存及正常缓存/无缓存配置，证明没有影响GPU性能。

## 原实现与文献

- [官方 MCFLIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)与 [2111.0源码](https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc)。Jenkinson M, et al. NeuroImage 17:825–841, 2002。
- [官方 BBR 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)、[FLIRT源码](https://git.fmrib.ox.ac.uk/fsl/flirt)、`flirtsch/bbr.sch` 与 MISCMATHS `optimise.cc`。Greve DN, Fischl B. NeuroImage 48:63–72, 2009，doi:10.1016/j.neuroimage.2009.06.060。
