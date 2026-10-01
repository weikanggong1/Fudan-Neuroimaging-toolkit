# fMRI volume：相同输入与处理步骤的原软件对照

本次比较从同一例真实 BOLD、SBRef 和 T1 开始，分别运行 FNIT 与原 FreeSurfer/FSL 实现。两边都采用 ICA-AROMA，以及相同定义的 WM、CSF 和 24 项运动回归。主对照选择 `registration_backend="fnirt"`，用逐阶段结果区分运动、分割、配准、ICA 和重采样产生的差异。

两边完整运行并通过检查。本次还发现并修复了 T1 组织概率图投到 EPI 网格时缺少 FLIRT 预滤波的问题；主结果来自修复后重新处理的全部 490 帧。

## 数据与代码来源

输入为一例 UKB 发布的原始 490 帧 BOLD 与对应 SBRef。BOLD 网格为 `88×88×64×490`，TR 为 `0.735 s`。T1 为同一个体的存档重建影像，网格 `162×215×180`；本次没有验证它是否为扫描仪直接输出的原始 T1。两边使用同一张 FSL MNI152 T1 2 mm 模板及脑掩膜，标准网格为 `91×109×91`。

| 原始输入 | SHA-256 |
|---|---|
| BOLD | `67717caaa823141c6e59d46db52e92deb4419d232ba699c399c69eb8888a0e92` |
| SBRef | `ff4e0e49a504527468234456cba6ad367d63c1f3f529d8ff3519bc78e2076323` |
| 存档 T1 | `762cb85469a27f8c64555252f97130d2ecb964f8c0d8f67e6c245ca7d9aa17f3` |
| 官方 SynthStrip 权重 | `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33` |

修复后 FNIT 候选运行封存于 `ec57972ceb7d715dd8b91ba32e3512bc7743b9c5`；原软件 benchmark 驱动封存于 `380e23dbedcc36d1e34bd7eed414196089d6fe67`。这次运行使用新结果目录，不复用解剖缓存，也没有跳过某个计算阶段。原始影像、逐体素数组、设计矩阵、完整命令和日志均保留在私有服务器目录；公开内容只有匿名统计、哈希和已获授权的模板空间派生图。

本次配置没有提供原始 B0/GDC 校正输入，因此两边均不估计这些校正；不加入 BET、FEAT 阈值/膨胀、空间平滑、FIX、额外带通或全局信号回归。这些选择与当前 FNIT volume 步骤对应。

## 两边采用的步骤与参数

“相同步骤”表示输入、处理目的、参数与输出空间对应；优化与数值实现的差异需要实测。尤其是当前 volume 的运动估计仍使用 `fmri/motion.py` 的 NCC+Adam，实现不同于原 MCFLIRT，也没有调用包内独立的 `TorchMCFLIRT`。

| 步骤 | 当前 FNIT | 原软件参照与关键参数 | 输出 |
|---|---|---|---|
| EPI/T1 脑提取 | 本包 SynthStrip，GPU，同一官方权重 | 未改写的 FreeSurfer `mri_synthstrip -i INPUT -o BRAIN -m MASK --model WEIGHTS -g -t 8` | 各自原网格的 brain 与二值 mask |
| T1 组织与偏置场估计 | `TorchFAST` 默认 `FASTConfig` | `fast -t 1 -n 3 -I 4 -W 15 -O 4 -f 0.02 -l 20 -H 0.1 -R 0.3 -B -b` | CSF/GM/WM PVE、分割、恢复图与 bias；BBR 用 WM PVE≥0.5 |
| 运动估计/校正 | 脑 mask 加权 NCC，Adam；8 mm、4 mm、原分辨率，迭代 35/25/15；逐帧零初始化；最终样条采样 | `mcflirt -in BOLD -reffile SBREF -mats -plots -rmsrel -rmsabs -spline_final`，默认三阶段优化与帧间初始化传播 | 原生校正 BOLD、每帧 FLIRT 矩阵与六列运动参数 |
| 掩膜与强度 | 校正 BOLD 乘本次 EPI mask；所有 mask 内时间点合并的 p50 缩放到 10000 | `fslmaths CORRECTED -mas MASK`；`fslstats CORRECTED -k MASK -p 50`；`fslmaths MASKED -mul (10000/p50)` | 缩放后的原生 BOLD |
| 时间高通 | `gaussian_highpass(..., preserve_mean=True)`，100 s；时间投影 float64 累积、float32 输出 | `fslmaths SCALED -bptf 68.02721088435375 -1 -add TEMPORAL_MEAN`；sigma=100/(2×TR) | 原生 pre-ICA BOLD，保留时间均值 |
| T1→MNI affine | `TorchFLIRT`，12 DOF、corratio | `flirt -in T1_BRAIN -ref MNI_BRAIN -dof 12 -cost corratio` | FLIRT scaled-mm `.mat` 与重采样图 |
| T1→MNI nonlinear | `TorchFNIRT`，T1 配置，`fnirt_execution="optimized"` | 原 `fnirt`，T1→MNI152 2 mm 配置，并显式覆盖下表参数 | 系数/位移场、MNI→T1 RAS pull 与标准空间 T1 |
| EPI→T1 | 本包六自由度 FLIRT normmi 初始估计，再 BBR；`bbr_execution="batched"` | `flirt -dof 6 -cost normmi`；随后 `flirt -dof 6 -cost bbr -wmseg WM -init INIT -schedule bbr.sch` | BBR FLIRT `.mat` |
| WM/CSF 回归 mask | 用 T1→EPI 的 BBR 逆矩阵，FLIRT 降采样预滤波后做三线性采样；PVE≥0.8，交 EPI mask | 原 `convert_xfm -inverse` 与 `flirt -applyxfm -interp trilinear`，随后同阈值与 mask 交集 | 原生 EPI WM/CSF 二值 mask |
| ICA | 本包 PICA 自动定阶，最大迭代 500，seed=0 | 原 `melodic --dim=0 --dimest=lap --nl=pow3 --eps=0.001 --maxit=500 --seed=0 --Ostats --nobet --mmthresh=0.5 --report --tr=0.735 --mask=MASK` | mixing、频谱、thresholded IC maps |
| AROMA 分类 | 本包四项特征，1000 次采样，seed=0 | 外置作者原 `feature_time_series`、`feature_frequency`、`feature_spatial`、`classification`；IC maps 用原 `applywarp --interp=trilinear` 到 MNI | 特征、噪声成分编号 |
| AROMA 清理 | `aroma_mode="nonaggr"` | 原 `fsl_regfilt --in=PREICA --design=melodic_mix --filter=NOISE_IDS --out=AROMA`；编号从 1 开始，不加 `--aggressive` | 原生 AROMA BOLD |
| 末端混杂回归 | WM/CSF 均值、Friston-24、截距及一次/二次趋势；稳定 float64 联合投影 | benchmark 中独立 NumPy float64 SVD，相同设计和 `rcond=1e-8` | 去均值的原生 clean BOLD，float32 |
| MNI 输出 | 本包复合 BBR 与 T1 非线性 pull，完整 490 帧只采样一次 | 原 `applywarp --in=CLEAN --ref=MNI --warp=T1_COEFF --premat=BBR --interp=spline --mask=MNI_MASK` | `91×109×91×490` clean BOLD |

原 MCFLIRT 在本例默认输出为 int32，后续 FSL 掩膜/缩放输出为 float32；FNIT 运动校正输出为 float32。这个量化差异与运动优化差异一并保留在运动阶段比较中，不能从该阶段总误差单独分离量化贡献。原 MELODIC 与 Torch ICA 的 seed 都为 0，不表示它们采用相同的随机发生器或得到同一个分解。

FNIRT 两边都使用 brain-extracted T1 与完整 MNI 模板乘脑 mask，沿用完整模板 header；这对应当前 FNIT 输入，而不是 UKB 全头 T1 的原始 FNIRT 协议。原 FNIRT 显式参数如下；其中 `numprec`、`minmet` 等是原命令选项，FNIT 使用公开报告中的 T1 配置和 float32/TF32 张量实现，不声称优化轨迹逐步相同：

| 参数 | 值 |
|---|---|
| `subsamp` / `miter` | `4,4,2,2,1,1` / `5,5,5,5,5,10` |
| `infwhm` / `reffwhm` | `8,6,5,4.5,3,2` / `8,6,5,4,2,0` |
| `lambda` / `estint` | `300,150,100,50,40,30` / `1,1,1,1,1,0` |
| `applyrefmask` / `applyinmask` | `1,1,1,1,1,1` / `1` |
| `warpres` / `splineorder` | `10,10,10` mm / `3` |
| `imprefm` / `impinm` / 隐式 mask 值 | `1` / `1` / `0` |
| `ssqlambda` / `regmod` | `1` / `bending_energy` |
| `intmod` / `intorder` | `global_non_linear_with_bias` / `5` |
| `biasres` / `biaslambda` | `50,50,50` mm / `10000` |
| `refderiv` / `numprec` / `minmet` | `0` / `double` / `lm` |
| `jacrange` / `interp` | `0.01,100` / `linear` |

末端设计为 29 列：截距、一次/二次趋势、WM、CSF，以及 `[motion, previous_motion, motion², previous_motion²]` 的 24 列；第一帧的 previous-motion 为零。回归保留截距列，去掉其他常数列，其余列居中并归一化，按有效秩做一次联合投影。末端不重复执行 100 s 高通，不额外带通，不加回时间均值。原 AROMA 的非侵略性清理仍由原 `fsl_regfilt` 计算；末端 SVD 是独立的同定义参考，并非声称某个 FSL 命令天然与全部混杂策略等价。

## 实际软件与资源

安装目录的 `etc/fslversion` 为 `6.0.7.22`，各组件由独立 Conda 包提供，不能据此将整套二进制都称为同一个组件版本。

| 组件 | 实测版本 / build |
|---|---|
| FreeSurfer 原脚本 | `8.2.0-20260314-d932c45`；SynthStrip 源码 SHA `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590` |
| FSL FAST | `fsl-fast4 2111.3 / h8c873e0_9` |
| FSL FLIRT | `fsl-flirt 2111.4 / h28ab0de_3` |
| FSL FNIRT | `fsl-fnirt 2203.2 / ha905fa9_5` |
| FSL MCFLIRT | `fsl-mcflirt 2111.0 / h465c9ab_11` |
| FSL MELODIC | `fsl-melodic 2601.1 / h46763b2_0`；binary banner `2601.1-dirty` |
| 原 SynthStrip GPU 解释器 | Python `3.11.16`，PyTorch `2.5.1 / CUDA 11.8`，surfa `0.6.3` |
| 原 ICA-AROMA functions | 作者源码 SHA `fc48971c2edcf2dd6c7030f29cebb525ad7d84b1c9fcb342dd9d84c86b86f036` |

原 FreeSurfer 的 fspython 缺少 CUDA，因此这次用支持 CUDA 的 Conda Python 直接执行**未修改的原 SynthStrip 脚本**，仍用原权重和 `-g` 参数。首次无法执行 GPU 的尝试保留为失败记录，不计入本次有效整链运行。

AROMA functions 与三张分类 mask 从作者原网站下载到私有外部目录，不复制发布第三方源码或资源。Python 3 兼容副本仅去掉 `future.standard_library` 的兼容导入，并将 `past.utils.old_div` 改为 `operator.truediv`；实际使用的浮点除法公式保持原样。原文件、兼容 diff 和各自 SHA 均保留。兼容副本 SHA 为 `e1ab18e85be963cc14fda1216f0a31382b814f108c89c7e8a68b7112cd339da6`。

本次封存 FNIT 源码中的分类 mask 与另外从作者网站下载的原 mask 已逐张检查：均为 `91×109×91`、2 mm 二值图，affine、体素数组和文件 SHA 全部相同。CSF、edge、outside 的非零体素数分别为 7,396、70,408 和 670,927。[掩膜核对报告](matched_aroma_masks.public.json)记录完整哈希。这排除了两边使用不同分类 mask 文件的差异；配准后的 IC maps 与 ICA 分解仍可能不同。

## 计时与文件验收

整链用[连续驱动](run_native_matched_pipeline.py)依次运行[原生前处理](run_native_matched.py)、[原配准](official_registration.py)、[原 ICA/AROMA 与独立回归](official_denoising.py)和最终原 `applywarp`。私有 JSON 给出实际输入和原软件位置；脚本属于 benchmark，不是 FNIT 运行时依赖。

```bash
# 按 case.json 的单例输入连续运行原软件参照，保存每阶段时间和匿名报告。
python validation/fmri/run_native_matched_pipeline.py \
  --case-json /private/benchmark/case.json \
  --output-dir /private/benchmark/native \
  --source-revision 380e23dbedcc36d1e34bd7eed414196089d6fe67
```

`case.json` 指定 `bold`、`sbref`、`t1w`、`tr`、`mni_template`、`mni_mask`、`synthstrip_weights`、`fsl_root`、`freesurfer_root`、`aroma_functions` 和 `classification_masks_dir`；本次还指定 `threads=8` 与 GPU 版 `synthstrip_python`。相同输入由哈希核对。

计时记录三个范围，避免把纯算法时间与验证时间混在一起：

- 原程序子进程墙钟包含启动、影像读取、计算和写盘，排除外层哈希和文件检查；配准的 exec/exit trace 开销计在该子进程内。
- 各阶段 Python 驱动墙钟包含进程启动和阶段内验证。ICA/AROMA 驱动另记原 FSL 调用总和；该总和不包含 Python 特征计算、NumPy 回归或外层检查。
- 连续整链墙钟从解剖处理开始至最终 MNI 写盘完成，包含阶段内检查与启动，排除运行前输入哈希和最后一次 MNI 复读检查。FNIT API 时间及验证进程墙钟分别报告。

共享服务器上的单次冷运行作为观察值；下表时间来自这次已完成的有效整链，不能把不同输入、旧 FIX 比较或单阶段控制时间拼成总时间。

所有新输出检查 gzip CRC、维度、网格、时间轴及有限值；mask 另检查二值与非空，矩阵检查有限值及可逆性，最终 MNI 检查 mask 外为零。部分原程序的启动器在写出完整结果后返回 255；报告保留原退出码，只有符合显式允许规则且完整输出检查通过才接受，不将 255 改写为 0。配准另保留原 child exit trace。这些文件验收只确认结果可比较，精度由逐阶段数值比较确定。

## 已完成：同分解 AROMA 控制

这个控制固定先前完整真实运行 `3b9b0f8128f10da77d3fbb6c395e9127353c3e46` 的 pre-ICA BOLD、95 成分 mixing、motion 与 52 个噪声成分编号，直接调用原 `fsl_regfilt`，与已经捕获的 FNIT AROMA 输出比较。没有重跑 ICA，也没有换用本次新原软件整链的分解。

| 检查 | 实测 |
|---|---:|
| 完整时间轴 | 490 帧 |
| 脑 mask / 比较值数 | 97,345 体素 / 47,699,050 个值 |
| 原 `fsl_regfilt` vs FNIT AROMA float32 数组 | 全部逐值相同 |
| MAE / RMSE / 最大绝对差 | 0 / 0 / 0 |
| pooled r / 逐体素时间 r 中位数 | 1 / 1 |
| 两边隐式置零状态 | 全图相同；脑 mask 内 213 体素置零 |
| 保留体素的时间均值偏差 RMSE | `1.317217e-5`，两边相同 |
| 原 `fsl_regfilt` 子进程墙钟 | 21.708 s |
| Python 3.11 seed=0 采样顺序 | 1000×441 个有序行索引逐项相同 |
| 原 / FNIT motion feature 最大差 | `3.331e-16` |
| 原 / FNIT high-frequency feature 最大差 | 0 |

这里的“相同”指解码后的 float32 数组，而不是压缩 NIfTI 文件字节；header 或压缩不同仍可产生不同文件哈希。原子进程退出 255，完整新输出通过 CRC、网格和全部有限值检查。Python AROMA 的随机抽样一致性也不延伸为 FSL C++ 与 Torch ICA 的初始化一致性。

结果与来源哈希见[匿名同分解报告](matched_same_decomposition_aroma.public.json)。它支持当前非侵略性回归及运动/频率特征计算在固定分解下的一致性；空间分类、自动定阶和完整分解的差异由下面的新整链比较判断。

## 已完成：固定原运动输出的缩放与高通控制

另外固定本次原 MCFLIRT 的完整 490 帧输出与原 SynthStrip EPI mask，运行 FNIT 生产函数 `grand_mean_scale`、`gaussian_highpass`，与原 `fslmaths` 结果比较。它没有重跑运动估计、配准或 ICA；所用 mask 为 99,372 体素，与上面的旧同分解控制单独记录。

| 控制 | 实测 |
|---|---:|
| FNIT / 原软件 intensity factor | 同为 `1.4359563469270533` |
| 固定原运动结果后，缩放数组 MAE / RMSE / 最大差 | 0 / 0 / 0 |
| 高通比较值数 | 48,692,280 |
| 固定原 scaled 输入后，时间 r 中位数 | `0.9999999999991442` |
| 高通 pooled 去均值时间 r | `0.9999999999991681` |
| 高通 MAE / RMSE / 最大绝对差 | `0.000223763 / 0.000505985 / 0.00390625` |
| 掩膜外最大绝对值 | 0 |
| FNIT 高通调用 / CUDA 分配峰值 | 7.642 s / 0.164 GB |

固定原 intensity factor 的“FNIT 缩放→高通”与“原 scaled 输入→FNIT 高通”得到相同误差。高通调用计时包含主机/设备传输和返回 float32 数组，排除读取、哈希与比较；它只是独立控制时间，不能替代整链耗时。报告同时记录首次控制调用为 7.829 s，共享 GPU 上不解释这两个短测量间的差异。

在这个真实 run 上，缩放完全一致，高通的时间相关接近 1，剩余绝对误差处于 float32 输出尺度。由此可以将时间缩放/高通本身与运动估计产生的 pre-ICA 差异分开。完整数值与 SHA 见[高通隔离报告](matched_highpass_control.public.json)；可复现诊断脚本和私人影像保留在服务器。

## 本次发现并修复的组织图采样问题

旧流程直接对 T1 FAST 概率图做三线性点采样。FSL `flirt -applyxfm` 在较粗的 EPI 网格采样前默认预滤波；少了这一步，0.8 阈值会保留过多 WM/CSF 体素。现流程复用本包 `TorchFLIRT.applyxfm`，输入矩阵为 BBR 的逆，即 T1→EPI 的 FLIRT scaled-mm 矩阵。

下面固定真实 PVE、BBR 和共同 EPI mask，只改变采样方式。这个控制不重新估计配准或 FAST，也不受 ICA 重跑的影响。对原 FSL 组织 mask 的 Dice 如下：

| 概率图 / BBR 来源 | WM：点采样 → FLIRT 预滤波 | CSF：点采样 → FLIRT 预滤波 |
|---|---:|---:|
| 原 FAST / 原 BBR | 0.81803 → 1.00000 | 0.55749 → 0.99989 |
| 原 FAST / FNIT BBR | 0.81657 → 0.98851 | 0.55742 → 0.96811 |
| FNIT FAST / 原 BBR | 0.81626 → 0.99210 | 0.54774 → 0.95819 |
| FNIT FAST / FNIT BBR | 0.81520 → 0.98499 | 0.54751 → 0.94075 |

共同域为 96,820 个体素；估计量来自修复前的封存运行 `86f96b5`，采样实现来自 `ec57972`。同原 PVE、原 BBR 时，修复后 WM/CSF 概率图对 FSL 的 RMSE 为 `8.42e-6 / 7.82e-6`。剩余差异主要涉及 PVE 与 BBR 估计，两项一起变化会影响 0.8 阈值。

另一个控制确认，FLIRT 输出路径的 TF32 坐标矩阵乘法会将概率图 RMSE 增至约 0.0026。现改用已有的 float32 系数逐项计算坐标；GPU 仍默认允许 TF32，既不全局关闭 TF32，也不用 float16。控制脚本和源哈希见[组织采样交叉实验](tissue_sampling_cross.public.json)与[可复现脚本](compare_tissue_sampling.py)。最终 BIDS JSON 用 `FNIT.TissueInterpolation` 标明新路径。

## 本次完整结果

主表比较修复后 `ec57972` 与原软件驱动 `380e23d`。原生共同脑区为 96,823 体素，T1 共同脑区为 1,393,869 体素，MNI 共同域为模板 mask 与双方 warped brain 的交集，共 222,660 体素。没有为计算指标另行调整配准。

| 阶段 | 时间 r 均值 | 中位数 | pooled 去时间均值 r | RMSE |
|---|---:|---:|---:|---:|
| 490 帧运动校正图 | 0.941467 | 0.964133 | 0.931905 | 307.402 |
| 掩膜、强度缩放与高通后的 pre-ICA 图 | 0.951153 | 0.966887 | 0.944958 | 421.236 |
| 原生 AROMA 清理图 | 0.855936 | 0.876957 | 0.823727 | 425.186 |
| 原生最终 clean 图 | **0.850471** | **0.871444** | 0.816525 | 119.877 |
| MNI 最终 clean 图 | **0.853589** | **0.872108** | 0.831237 | 94.972 |

时间 r 在各体素的 490 帧内去均值后计算；常数时序不计算 r。运动/pre-ICA 的有效时间 r 数为 96,823，AROMA/native clean 为 96,644，MNI 为 222,660。RMSE 比较共同 mask 内全部值：原生 47,443,270 个、MNI 109,103,400 个。前两阶段采用原始或缩放后的强度单位；末端回归移除时间均值，设计仍保留截距列；不能跨阶段直接比较 RMSE 的大小。AROMA 的含时间均值 4D r 为 0.996379，而时间 r 均值为 0.855936，说明高静态强度相关并不代表时序相同。

| 脑/组织 mask | FNIT / 原软件体素数 | Dice |
|---|---:|---:|
| EPI SynthStrip | 97,348 / 99,372 | 0.984374 |
| T1 SynthStrip | 1,395,091 / 1,397,746 | 0.998174 |
| MNI brain | 223,166 / 224,952 | 0.993756 |
| EPI WM 回归 mask | 18,201 / 18,393 | **0.987102** |
| EPI CSF 回归 mask | 4,362 / 4,727 | **0.928375** |

FAST 的 T1 CSF/WM PVE 在共同 T1 脑区的空间 r 为 0.989897 / 0.993204，RMSE 为 0.054637 / 0.051097；这些 PVE 的差异仍会影响阈值 mask，不能称为逐体素数值等价。候选没有额外捕获 GM PVE、BBR 初始矩阵或初始重采样图，报告明确列出未测项目，不据 WM/CSF 推造 GM benchmark。

| 变换的实际采样位置差 | RMS mm | p95 mm |
|---|---:|---:|
| 每帧运动矩阵 | 平均 0.349449 | 帧 RMS 的 p95 为 0.411194 |
| T1 affine 逆映射 | 0.080847 | 0.126263 |
| 最终 BBR 逆映射 | 0.092510 | 0.136371 |
| MNI→T1 完整 pull | 0.121560 | 0.247603 |
| MNI→EPI 完整复合 pull | 0.158764 | 0.285088 |

运动在共同 EPI 的 96,823 点评价，T1 affine 逆映射在模板脑区 228,483 点评价，BBR 逆映射在共同 T1 脑区 1,393,869 点评价，完整 pull 在共同 MNI 的 222,660 点评价。表中均为 reference→source 采样位置差，但各项评价域不同。矩阵按 FSL scaled-mm 坐标转成世界坐标再比较，未直接把 `.mat` 数值当毫米距离。MNI→T1 变换场用固定 MNI 网格上的 RAS-mm pull displacement 表示。原 FNIRT 系数已经包含初始 T1 affine；组合只另外加入 EPI→T1 的 BBR，未重复应用 T1 affine。[独立几何审计](matched_native_geometry.public.json)验证了原系数 header、scaled-mm↔RAS 转换、forward 组合方向和阈值规则。该审计的候选几何控制来自修复前 `86f96b5`；上表当前 `ec57972` 数值来自最终比较报告。此主链不需要非线性 invwarp。

完整哈希、版本、有效体素数、坐标定义与所有指标见[最终机器可读报告](matched_pipeline.public.json)。它保留[比较脚本](compare_matched_pipeline.py) SHA；额外比较过程的 479.21 s 包含控制与读取，不算入任一 pipeline 时间。

## 差异主要来自哪里

第一处显著时序差异已出现在运动校正：时间 r 均值为 0.941467。固定原运动输出之后，缩放完全相同，高通时间 r 中位数为 `0.9999999999991442`、RMSE `0.000505985`，见[真实高通控制](matched_highpass_control.public.json)。因此本例 pre-ICA 差异不能归为高通滤波公式错误；当前 NCC+Adam 与 MCFLIRT 的优化、初始化和原输出 int32 量化需分开进一步验证。

完整 ICA 与 AROMA 也不同。修复后 FNIT 为 **96 个成分、70 步收敛、62 个噪声成分**；原 MELODIC 为 **95 个成分、40 步收敛、50 个噪声成分**。将 mixing 列去均值并做绝对 Pearson r 最大化的 Hungarian 匹配，95 对成分的 r 中位数为 0.764251，对齐标签一致率为 73.68%，匹配成分的噪声 Jaccard 为 0.632353；候选另有一个未匹配的噪声成分。见[成分对齐](matched_ica_alignment.public.json)及[脚本](compare_ica_components.py)。两边 pre-ICA 本来就不同，该实验不能单独把分解差异归因于 Torch ICA。

末端回归设计也有影响。双方设计均为 490×29、秩 29；固定候选设计处理两份 AROMA 图，时间 r 中位数为 0.897999，固定原设计时为 0.903744；各自设计为 0.871444。固定同一幅 AROMA 图只换设计，候选/原软件自身 r 中位数分别为 0.963989 / 0.958716。独立 NumPy SVD 对实际候选/原输出的 RMSE 仅 `4.990e-6 / 5.017e-6`，支持投影实现正确、输入混杂设计不同。见[设计控制](matched_nuisance_design_control.public.json)及[脚本](compare_nuisance_designs.py)。

最后，warp 与采样的交叉控制如下，始终使用同一三次 B 样条 sampler 与共同 MNI mask：

| 完整 490 帧控制 | 时间 r 均值 | 中位数 |
|---|---:|---:|
| 两边 clean，固定候选 warp | 0.856642 | 0.874811 |
| 两边 clean，固定原 warp | 0.856725 | 0.874920 |
| 固定候选 clean，换两套 warp | **0.994511** | 0.997152 |
| 固定原 clean，换两套 warp | **0.994728** | 0.997251 |
| 原 FSL `applywarp` 输出 / 相同场的 FNIT sampler | **0.999999999921** | 0.999999999945 |

最后一行比较 109,103,400 个值，RMSE 为 0.002215，最大绝对差 0.113892；候选实际 MNI 与同 sampler 重采样逐值相同。这验证了本次 FSL warp 转换和最终样条采样契约，主要端到端差异仍在前端清理。四个交叉采样共 195.94 s，包含读写，属于验证开销。

修复前的独立冷运行对同一原参照，MNI 时间 r 均值为 0.849717，中位数为 0.872030；WM/CSF mask Dice 为 0.814745 / 0.536688，见[修复前控制](matched_before_tissue_fix.public.json)。新的 mask 一致性明显提高，但两次冷运行的 pre-ICA 与 ICA 文件也有数值变化，自动阶数由 95 变为 96，见[逐文件哈希核对](matched_before_after_preica_hashes.public.json)。不能把所有最终 r 变化仅归为这一处修复；固定 PVE/矩阵的采样控制才隔离了其贡献。

此前约 0.272 的结果比较 FNIT SynthMorph/AROMA 与 UKB FIX、GDC/B0、官方配准处理后的发布图。它与这里的 FNIRT/AROMA 相同步骤参照有不同处理协议。本例相同步骤的 MNI r 均值为 0.853589；这仍是实现差异测量，不是去噪质量的判据。

## 整链时间与显存

gpucw1 的共享 H100 PCIe，8 个 CPU 线程，FNIT float32/TF32。原 SynthStrip 在 GPU 上运行，后续原 FSL 命令在 CPU 上运行。双方都是从新目录开始的单次完整冷调用。

| 阶段 | FNIT 阶段计时，s | 原参照阶段驱动，s |
|---|---:|---:|
| EPI/T1 SynthStrip、模板准备、FAST与缓存核验 | 11.01 | 169.68 |
| FEAT 核心 | 227.37 | 621.12 |
| T1 affine/FNIRT、BBR、场转换及组织采样 | 38.51 | 298.39 |
| PICA/MELODIC、AROMA与混杂回归 | 123.76 | 934.65 |
| 最终 MNI 重采样 | 49.02 | 546.61 |
| **完整调用/连续链墙钟** | **455.62（7.59 min）** | **2570.47（42.84 min）** |
| 独立验证进程墙钟 | 530.35 | 2601.72 |
| CUDA allocated / reserved，GB | 6.239 / 7.317 | 未单独记录 |
| 主机峰值 RSS，GB | 6.053 | 8.785 |

FNIT 表内阶段按所列小阶段相加，API 包含最终 BIDS 保存；捕获额外影像复制 47.517 s 已从 API 和相关阶段扣除。原参照阶段驱动包含启动和阶段内结果检查，最后重采样列为原 `applywarp` 子进程时间；原连续链排除预先输入哈希和最终 MNI 复读。两边计时边界、原 MELODIC 的额外 HTML report 和共享服务器负载不同，因此这些数值用于记录本次实际等待时间，不宣称稳定加速倍数或等价输出的性能优势。

原程序独立子进程中，FAST 为 150.00 s、MCFLIRT 为 326.10 s、高通为 157.57 s、FNIRT 为 219.91 s、MELODIC 为 651.87 s。末端原 `applywarp` 为 546.61 s；相同原场的全 490 帧 sampler 控制已在上表验证精度。逐命令原退出码与时间见[解剖](matched_native_anatomy.public.json)、[前处理](matched_native_feat.public.json)、[配准](matched_native_registration.public.json)、[去噪](matched_native_denoising.public.json)；整链边界见[原连续运行](matched_native_pipeline.public.json)。

[FNIT 实际运行](matched_fnirt_volume.public.json)、[独立输入/输出与来源核对](matched_fnirt_contract.public.json)及[输入预检](matched_input_preflight.public.json)记录完整哈希、源文件和依赖版本。修复后的 46 项目标测试通过，benchmark 的 12 项测试通过，见[测试记录](matched_tests.public.json)。

## 示例图

四行依次为 MNI 解剖模板、FNIT 与原软件清理后 BOLD 的时间标准差，以及两边逐体素时间 Pearson r。中间两行共用色阶；最后一行固定色阶 -1 至 1。切面、共同 mask 与完整时间轴相同，未对显示图做空间平滑；mask 外和常数时序的 r 留空。模板只提供解剖定位。[图像来源记录](matched_figure.public.json)保存 PNG 和绘图脚本 SHA。

![相同步骤 FNIT 与原软件完整490帧结果](../../docs/fmri/figures/fmri_matched_native.png)

## 原实现依据

原软件行为与命令参照 [MCFLIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)、[FAST 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)、[BBR 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)、[FNIRT 用户指南](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)及 [MELODIC / fsl_regfilt 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)。AROMA 特征、分类与原始命令依据[作者代码](https://github.com/maartenmennes/ICA-AROMA/blob/master/ICA_AROMA_functions.py)。

方法参考：Jenkinson et al., NeuroImage (2002), [doi:10.1016/S1053-8119(02)91132-8](https://doi.org/10.1016/S1053-8119(02)91132-8)；Beckmann & Smith, IEEE TMI (2004), [doi:10.1109/TMI.2003.822821](https://doi.org/10.1109/TMI.2003.822821)；Pruim et al., NeuroImage (2015), [doi:10.1016/j.neuroimage.2015.02.064](https://doi.org/10.1016/j.neuroimage.2015.02.064)。
