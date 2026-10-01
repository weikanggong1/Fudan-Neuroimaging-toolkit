# 同输入 fMRI volume 全流程对照

2026-10-01，最新冻结代码 `1eb9c417febebc8bdd450590d4759454e7160141` 从真实 BOLD/SBRef/T1 完整运行 FNIT FNIRT 分支；原参考驱动为 `380e23dbedcc36d1e34bd7eed414196089d6fe67`。两边同一输入、模板、官方 SynthStrip 权重和处理参数，全部 490 帧。最终 MNI 时间相关均值 **0.938768**，中位数 **0.947436**；输出尚未逐体素等价。

## 数据、协议与原命令

一例 UKB 静息态 BOLD：88×88×64×490，TR 0.735 秒；T1 为 162×215×180 的匹配重建存档，经 nibabel 转换保留原体素与 affine。没有证明它是扫描仪直接输出的原始 T1。所有原影像、逐体素数据、完整日志和设计矩阵留服务器；公开匿名汇总、文件/源码哈希及获授权的模板空间 PNG。

不包含 GDC、B0、BET、FIX、空间平滑和额外带通。双方采用 SynthStrip、100 秒高通、自动阶数 PICA、non-aggressive AROMA，再做 WM/CSF/Friston-24 联合回归。原末端回归用独立 NumPy float64 SVD，投影定义与 FNIT 相同。它是按 FNIT 步骤建立的原软件参照，与 UKB 已发布的完整 FIX 结果属于不同协议。

| 阶段 | 当前 FNIT 调用 | 对应原命令/操作 |
|---|---|---|
| EPI/T1 脑提取 | 本包 `SynthStrip`，border=1、原 PT 权重 | `mri_synthstrip -i INPUT -o BRAIN -m MASK` |
| T1 组织/bias | `TorchFAST(execution="fsl")`，三类 T1，无先验 | `fast -t 1 -n 3 -I 4 -W 15 -O 4 -f 0.02 -l 20 -H 0.1 -R 0.3 -b -B` |
| 运动 | 直接 `TorchMCFLIRT.run`；8/4/4 mm、原 NCC、Brent、相邻帧初值、Constant 样条 | `mcflirt -in BOLD -reffile SBREF -mats -plots -rmsrel -rmsabs -spline_final` |
| 强度缩放/高通 | `grand_mean_scale`、`gaussian_highpass` | mask 内 p50 缩放至 10000；`fslmaths -bptf 68.0272108844 -1 -add temporal_mean` |
| T1→MNI | 本包 FLIRT、T1 preset FNIRT | FLIRT 12 DOF normmi；FNIRT `T1_2_MNI152_2mm.cnf` |
| EPI→T1 BBR | 本包 normmi FLIRT 初值、WM PVE≥0.5、BBR | `flirt -dof 6 -cost bbr -wmseg ... -init ... -schedule bbr.sch` |
| EPI WM/CSF | 本包 `TorchFLIRT.applyxfm`，逆 BBR、默认降采样预滤波，PVE≥0.8 | `flirt -applyxfm -init T1_to_EPI.mat -interp trilinear`，阈值并交 EPI mask |
| PICA | 本包 `decompose_spatial_ica`，Laplace/pow3、seed=0 | `melodic --nobet -m MASK -d 0 --dimest=lap --nl=pow3 --eps=0.001 --seed=0 --maxit=500 --Ostats --mmthresh=0.5` |
| ICA-AROMA | 本包原特征规则及 nonaggr | 作者分类函数与 `fsl_regfilt` |
| 联合混杂回归 | 截距、线性/二次趋势、WM/CSF 和 Friston-24，29 列 | 独立 NumPy 去均值/L2 归一化、rank rcond=1e-8 的 SVD 投影 |
| MNI 最终重采样 | 本包周期边界三次 B 样条 | `applywarp --in=CLEAN --ref=MNI --premat=BBR --warp=FNIRT --interp=spline` |

原 uint16 BOLD 经 MCFLIRT 得到 int32，FNIT 也向零截断成 int32，再转 float32 做 FEAT。原 FNIRT 系数已经含初始 T1 affine，组合只另外加入 EPI→T1 BBR，不再重复 T1 affine。`.mat` 按 FSL scaled-mm 转到世界坐标，比较 reference→source 采样点；warp 以 MNI 网格的 RAS-mm pull displacement 比较，不能直接混用 FSL scaled-mm 位移与 RAS 位移。

本轮原软件组件版本与实际可执行文件 hash 保存在[主报告](matched_pipeline.public.json)：

| 组件 | 实际版本 |
|---|---|
| `fast` | 2111.3 |
| `flirt` | 2111.4 |
| `fnirt` | 2203.2 |
| `mcflirt` | 2111.0 |
| `fslmaths` | 2209.6 |
| `applywarp` | FSL FNIRT component 2203.2 |
| `mri_synthstrip` | FreeSurfer 8.2.0-1 original source |
| `ica_aroma` | Original ICA-AROMA Python3 compatibility |
| `melodic` | 2601.1-dirty |

## 本轮子函数修正

| 功能 | 修正 | 固定输入真实验收 |
|---|---|---|
| [SynthStrip](../../docs/synthstrip/README.md) | shape/2 视野中心、离散 LIA、header pixdim、裁剪及回采样边界 | SBRef/T1 的 conform 数组和网络输入相同；同预测回采样 mask 相同；独立 GPU 推理仍有少量边界差异。 |
| [TorchMCFLIRT](../../docs/mcflirt/README.md) | 替换旧 Adam 路径，原 NCC 累加、Brent 容差/初值传播、NEWIMAGE 坐标/样条/整数转换 | 当前完整 GPU motion r 均值 0.999578；矩阵 RMS 均值 0.00881 mm。固定原矩阵文本的 8 帧采样 r 均值 0.999998725。 |
| [MELODIC](../../docs/melodic/README.md) | 空间均值只作用于 PCA covariance，匹配 PPCA、Linux RNG、double 分解、缩放/排序、Gamma 后验尾部 | 固定原 filtered/mask，95 成分/40 步；mix/map r 中位数 0.999999978/0.999999970，阈值 Dice 0.999562。 |
| [TorchFAST](../../docs/fast/README.md) | `execution="fsl"` 保留 radiological X、连续随机流、原位顺序和 bias/PVE 表达式 | 同原 T1_brain，三张分类图相同，PVE r≥0.999999991；仅少量体素相差一档0.01。独立默认 tensor 路径保留。 |

这些修正同时发布在各自源码目录和说明页；volume 直接调用本包实现。完整流程的源码清单、实际依赖与输入 hash 已独立核对：[输出合同](matched_fnirt_contract.public.json)中 101 个公开源文件和两份输出的各 87 项来源清单均无不匹配。实际配置确认 FAST=fsl、motion_iterations=[1,1,1]、无解剖缓存、TF32 开启、未启用低精度。全部最终影像有限、float32、TR/sec 与网格一致，MNI mask 外为0。[116项聚焦测试与12项驱动复测](matched_tests.public.json)通过；这不是全仓库测试全部通过的声明。

## 当前整链精度

原生共同脑区 99,371 体素，T1 共同脑区 1,397,744，MNI 共同域为模板 mask 与双方 warped brain 交集 224,709。各体素的 490 帧内去均值后计算时间 Pearson r；常数时序不计算。RMSE 为共同 mask 内全部值，阶段间强度和时间均值处理不同，不能跨阶段把 RMSE 大小当误差排名。

| 阶段 | 时间 r 均值 | 中位数 | 第 5 百分位 | RMSE |
|---|---:|---:|---:|---:|
| 运动校正 | 0.999578 | 0.999834 | 0.998330 | 8.211 |
| 缩放、高通后的 pre-ICA | 0.999356 | 0.999770 | 0.997351 | 11.745 |
| 原生 AROMA | 0.934092 | 0.944091 | 0.866459 | 87.136 |
| 原生最终 clean | 0.940704 | 0.951029 | 0.877215 | 78.401 |
| MNI 最终 clean | 0.938768 | 0.947436 | 0.880877 | 62.390 |

| 脑/组织 mask | FNIT / 原体素数 | Dice |
|---|---:|---:|
| EPI SynthStrip | 99,372 / 99,372 | 0.999989937 |
| T1 SynthStrip | 1,397,746 / 1,397,746 | 0.999998569 |
| MNI 脑 | 224,882 / 224,952 | 0.999075214 |
| EPI WM | 18,398 / 18,393 | 0.990405262 |
| EPI CSF | 4,746 / 4,727 | 0.970970126 |

完整链自产 T1 的 CSF/WM PVE 空间 r 为 0.999977292/0.999977695，RMSE 0.002594/0.002927；这与固定同一原 T1_brain 的 FAST 控制分开。新 SynthStrip 输入掩膜虽只差4个边界体素，也会影响这些完整链 PVE。GM、初始重采样图和 registered T1 在此次 capture 中未保存，主报告明确 omitted 项，不据其他图推造结果。

| 变换比较 | 评价域 | reference→source 采样位置 RMS（mm） |
|---|---|---:|
| 490 帧运动矩阵 | 共同 EPI；逐帧 RMS 的均值 | 0.008812 |
| T1 affine | 共同 MNI | 0.057998 |
| EPI BBR | 共同 T1 | 0.119144 |
| 完整 MNI→T1 FNIRT pull | 共同 MNI | 0.122997 |
| 合成 MNI→EPI pull | 共同 MNI | 0.174359 |

## 余差的定位

运动和 pre-ICA 的时间相关均值已超过0.999；最大的下降出现在 ICA/AROMA。固定相同原 pre-ICA 输入时，ICA 配对 r 接近1，配对后的95个噪声/信号标签全部一致；完整链 pre-ICA 不完全相同，自动分解得到95/95成分、41/40步，噪声47/50。mix 配对 r 中位数为0.855702，95维子空间 principal cosine 中位数为0.999986892；阈值图配对空间 r 中位数0.638585。低相关配对不能视为同一生理成分。82/95标签一致，42个共同噪声、5个FNIT独有、8个原软件独有。见[ICA 对齐](matched_ica_alignment.public.json)。

这些控制支持小的前处理变化会传播到 ICA 的分解方向、阈值与自动分类；它们没有单独量化每种变化的因果贡献，也不意味着任一去噪结果质量更好。整体尚未只剩浮点输出误差。

固定图像/场的完整490帧交叉重采样进一步区分清理与空间变换：

| 控制 | 时间 r 均值 | 中位数 | RMSE |
|---|---:|---:|---:|
| 固定 FNIT warp，换两份 clean | 0.944884743919 | 0.952832588068 | 59.736828 |
| 固定原 warp，换两份 clean | 0.944921812702 | 0.952880057018 | 59.760197 |
| 固定 FNIT clean，换两套 warp | 0.993487143927 | 0.996117997374 | 18.777447 |
| 固定原 clean，换两套 warp | 0.993497628924 | 0.996110366608 | 18.421769 |
| FNIT 保存 / 同输入 sampler | 1.000000000000 | 1.000000000000 | 0.000000 |
| 原保存 / 原输入 FNIT sampler | 0.999999999921 | 0.999999999945 | 0.002218 |

原场下 sampler 的高一致性排除了最终 applywarp 公式是主要原因。固定同一 warp 后仍约0.945，剩余主要在上游清理；只更换 warp 时约0.9935，配准也有较小贡献。两套采样都使用同一 MNI mask、相同完整时间轴，不额外调整配准。

固定原运动输出的缩放逐值相同，高通时间 r 中位数为0.999999999999144、RMSE0.000505985，见[单步控制](matched_highpass_control.public.json)。组织 PVE 投到 EPI 的路径已保留 FLIRT 降采样预滤波与 float32 逐项坐标，本轮未改变该成熟采样合同。最新[混杂设计控制](matched_nuisance_design_control.public.json)中，两套设计均为490×29、秩29。固定候选或原设计、只换两份AROMA输入，时间r均值分别为0.947048/0.947195；同一份AROMA只换设计为0.990576/0.991250。独立float64 SVD对双方实际clean输出的RMSE为5.17×10⁻⁶/5.02×10⁻⁶，支持末端投影计算一致、主要差异已存在于AROMA输入和各自估计的设计。该控制没有重新估计ICA，也没有替换生产设计。


## 完整耗时

| 范围 | FNIT（s） | 原连续链（s） |
|---|---:|---:|
| EPI/T1 脑提取、模板与 FAST | 33.31 | 169.68 |
| 运动、采样、mask、缩放、高通 | 973.12 | 621.12 |
| T1 affine/FNIRT、BBR、warp 准备 | 72.01 | 298.39 |
| PICA、AROMA、联合混杂回归 | 182.37 | 934.65 |
| 最终 MNI 重采样及该阶段输出 | 50.54 | 546.61 |
| **API含保存 / 原连续链** | **1318.04** | **2570.47** |
| 独立验证进程 | 1372.24 | 2601.72 |

共享 H100、8线程；原 SynthStrip用GPU、FSL用CPU。FNIT API 排除中间 capture 额外复制27.05s，含最终BIDS保存；pipeline total1314.35s的末尾保存边界不同。原连续链含阶段检查和MELODIC HTML输出。FNIT CUDA allocated/reserved为8.316/9.745GB。原FAST、regfilt、最终applywarp有退出255的记录，完整输出通过CRC、形状和有限值检查后由benchmark接受；异常原码未改写为成功。分别见[原解剖](matched_native_anatomy.public.json)、[原去噪](matched_native_denoising.public.json)和[原连续链](matched_native_pipeline.public.json)。这是一例单次测量，不能据此推导稳定加速比。

新MCFLIRT每帧/每阶段顺序Brent优化，需要反复生成坐标、采样和读取标量cost，GPU存在大量同步及Python调度；FEAT因此比旧Adam实现更慢。前8帧暖控制CPU5.41s、GPU11.13s；完整CPU只估计387.60s，原MCFLIRT含采样/写盘326.10s。GPU整链未把fit和sampling从FEAT973.12s独立分离，不把总FEAT时间称运动函数时间。

## 复测与示例图

FNIT单run命令见[验证入口](README.md#单被试复测)。原软件连续驱动读取私有JSON，按上述定义生成中间与最终输出；该驱动属于验证，不被FNIT runtime调用：

```bash
# case-json 指定同一 BOLD/SBRef/T1、模板、权重和原软件位置。
# output-dir 使用新目录；脚本自动在此目录写 pipeline.public.json。
python validation/fmri/run_native_matched_pipeline.py \
  --case-json /private/case.native.json --output-dir /private/native-fresh \
  --source-revision YOUR_REFERENCE_DRIVER_COMMIT
```

私有 `case-json` 的 `bold`、`sbref`、`t1w`、`mni_template`、`mni_mask`、`synthstrip_weights` 为绝对文件路径；`fsl_root`、`freesurfer_root` 为原软件目录。`aroma_functions` 指向作者 Python3 兼容函数文件，`classification_masks_dir` 包含原 `mask_csf`、`mask_edge`、`mask_out` 三图；`tr` 为秒，`threads` 默认8。可选 `synthstrip_python` 指定原 SynthStrip 的解释器。原软件只用于此参照驱动，FNIT API 不读取这份清单。

[`compare_matched_pipeline.py`](compare_matched_pipeline.py)读取私有配对清单，比较各阶段影像、mask、矩阵和pull，并做完整交叉采样；[`compare_ica_components.py`](compare_ica_components.py)做一对一时间/空间配对；[`compare_nuisance_designs.py`](compare_nuisance_designs.py)用独立SVD交叉回归设计。公开前核对报告中脚本SHA和输入文件hash。机器可读报告保存完整定义，不上传逐体素数组或含病例路径的清单。

下图四行是MNI模板、FNIT temporal SD、原软件 temporal SD、逐体素时间r。相同切面/共同mask、SD共用色阶、r色阶−1到1；常数时序留空，不额外平滑。[图示来源](matched_figure.public.json)记录PNG和脚本SHA。

![当前完整490帧对照](../../docs/fmri/figures/fmri_matched_native.png)

原实现与文献：各功能页引用[FSL MCFLIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)、[FAST](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)、[BBR](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)、[FNIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)、[MELODIC](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)及[作者ICA-AROMA代码](https://github.com/maartenmennes/ICA-AROMA/blob/master/ICA_AROMA_functions.py)。
