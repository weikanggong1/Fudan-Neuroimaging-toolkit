# 真实静息态 fMRI 验证

本页记录一例真实 UK Biobank BOLD、SBRef 和 T1 的算法对照。BOLD 为 88×88×64×490，TR 为 0.735 秒。T1 使用同被试已有的去脑 NIfTI；它与 T1 ZIP 中的 `orig/001.mgz` 网格一致、强度不同，具体生成步骤未知。原始 B0 场图和 GDC warp 均不可用，所以 FEAT 对照同时关闭这两步；此结果不能代表原 UKB 全部校正步骤的逐体素复现。FNIT 运行时不调用 FSL。

## FEAT 的原版参照

参照取自 FSL 6.0.7.22 的 `featlib.tcl`。在原有 FSF 中关闭 GDC、B0 场图校正、高分辨率配准和 ICA；保持 MCFLIRT 的 `-spline_final`、BET、10% 强度阈值与掩膜膨胀、单一乘数的 grand-mean 缩放、100 秒高通。原 FSF 的 `st=0`、`smooth=0`、`templp_yn=0`、`perfsub_yn=0`，因此没有 slice timing、空间平滑、低通和灌注相减步骤。

这个 FSL 安装环境的 `feat` 启动器三次均在首次 4D `fslmaths` 拷贝后返回 255。对应输出通过 gzip 完整性检查；仍不能称 `feat` 启动器已完整运行。正式 490 帧参照改用 [原版程序逐条运行脚本](official_feat_no_gdc.sh)中的 MCFLIRT、BET、`fslstats` 和 `fslmaths` 命令；各中间输出均已检查。通用脚本另用同一真实 BOLD 的前 8 帧从头执行成功，产生有限值的 4D 结果。8 帧只检验脚本可执行，不用于正式精度和时间比较。

复现命令（独立验证环境需要安装 FSL；`OUT_DIR` 使用新目录）：

```bash
FSLDIR=/path/to/fsl bash official_feat_no_gdc.sh \
  /absolute/path/bold.nii.gz /absolute/path/sbref.nii.gz \
  /absolute/path/out 0.735
```

第四个位置参数是原始 BOLD 的 TR，单位秒。命令及原版二进制 SHA、几何/有限值检查和分步计时见 [FEAT 标量摘要](feat_summary.json)。

## 490 帧 FEAT 对照

下面的主对照使用同一原始 BOLD/SBRef、同一 EPI 网格，FNIT 高层整链先对 SBRef 用 SynthStrip 求掩膜，并把它显式传给 FEAT；运动重采样使用三次 B 样条。独立 `run_feat_core` 的默认行为是从运动校正后的 EPI 均值求 SynthStrip 掩膜，本次主对照并未使用该默认入口。体素强度比较只取官方掩膜与 FNIT 掩膜的交集，共 98,733 个体素；逐体素时间相关先在各体素 490 帧内计算，再报告中位数。

| 检查 | 结果 |
|---|---:|
| 官方 / FNIT 掩膜体素 | 113,881 / 99,401 |
| 掩膜 Dice | 0.925845 |
| 运动校正均值图 Pearson r；MAE（原强度单位） | 0.997652；141.918 |
| 滤波后 4D Pearson r；MAE（归一化强度单位） | 0.997429；426.525 |
| 4D 各体素去时间均值后的 pooled r | 0.949149 |
| 逐体素时间相关中位数 | 0.973472 |
| 运动参数差：平移 / 旋转中位数 | 0.23975 mm / 0.20952° |

两份 4D 结果均为 88×88×64×490，affine 完全相同，所有 242,851,840 个 FNIT 数值均有限。固定官方 MCFLIRT 的 4D 结果、官方掩膜和缩放中位数后，只用 FNIT PyTorch 执行缩放与高通，最终与 FSL 的 MAE 为 0.000192、Pearson r 接近 1。

另保持 FNIT 最终整链的 spline 运动矩阵和重采样，仅改用官方最终掩膜及膨胀前掩膜取缩放中位数：在与默认整链相同的 98,733 个交集体素上，4D MAE 从 426.525 降至 206.084，均值从 10423.126 接近官方的 10068.613（控制结果 10072.464）；去均值时间 r 保持约 0.94915。此固定官方掩膜运行仅用于归因，不是默认 SBRef-SynthStrip 输出。两项控制分别显示时间滤波公式吻合，以及掩膜/缩放解释了大部分强度偏差；剩余时间差异仍与运动估计和重采样有关。

FSL MCFLIRT 实测 397.54 秒；其余 11 个已单独计时的影像命令合计 428.12 秒，另有两次未单独计时的 `fslstats`。因此 825.66 秒是分步耗时下界，**不是**一次完整 `feat` 运行时间。最终 490 帧 FNIT 成功整链的 FEAT 核心为 243.06 秒，默认 SynthStrip 另需 8.59 秒。FSL 与 FNIT 使用的处理器不同，且官方 `feat` 启动器未提供完整运行时间，因此不将分步下界解释为严格配对加速比。

## 分步与其他模块

- [固定官方运动矩阵的插值核对照](motion_spline_summary.json)：8 个真实时间点；区分插值误差与运动矩阵误差。
- [MCFLIRT 运动求解差异](mcflirt_difference.public.json)：同一真实 490 帧的矩阵误差、搜索步骤及计时范围；FNIT 求解器与 MCFLIRT 未达到数值等价。
- [EPI→T1 BBR](bbr_summary.json)：同一初始矩阵和同一白质分割的受控对照，以及 FNIT 白质分割的独立影响。
- [MELODIC/PICA](pica_summary.json)：同一真实 4D 输入与掩膜的组件数、重建和耗时检查；原版程序状态保留在摘要中。
- [独立 MELODIC BIDS 入口](melodic_bids_current.json)：同一真实 BOLD 的前 64 帧，在完成 volume 回归后检查成分图、混合矩阵、收敛状态、BIDS 来源链接及 CPU 耗时。
- [当前 T1→MNI152 2 mm FNIRT 对照](t1_fnirt_20260929.public.json)：同一真实 T1、模板和脑掩膜的 FSL 配对精度、时间、显存及输入/源码 SHA256。此前 FSL 与 SynthMorph 标量保留在[参照摘要](registration_summary.json)。
- [ICA-AROMA 与完整 BIDS→MNI152 2 mm 结果](e2e_summary.json)：最终运行的组件数、噪声分类、输出完整性和各阶段耗时。
- [当前 fMRI volume FNIRT 整链](fmri_volume_fnirt_20260929.public.json)：同一例真实 490 帧 BOLD 的退出码、全体素有限值、模板网格、掩膜外零值、TR、GPU 显存、时间和输入/输出 SHA256。
- [重构后 volume BIDS 入口](volume_bids_current.json)：同一真实 BOLD 的前 64 帧，在 CPU 上从原始 BIDS 到最终 BIDS Derivatives 的文件结构、有限值和耗时检查。
- [fMRIPrep 表面路径标量摘要](surface_current.json)与[函数、输出和官方命令对照](../../docs/fmri/surface.md)：真实 490 帧 BOLD 的 T1w 皮层投影及 MNI 皮层下组装；固定官方球面时 CIFTI 与 NiWorkflows 官方源码逐值一致，并单独量化 FNIT HOCR/FastPD 与官方 newMSM 球面产生的每点时间相关差异。旧 MNI 表面流程的单帧对照已移除。
- [重构后 surface BIDS 入口](surface_bids_current.json)：把此前完成回归的完整 490 帧真实 volume 结果按 BIDS Derivatives 路径接入，验收 MSMSulc、fsLR32k GIFTI、91k CIFTI 及 JSON 写出。

## 最终 MNI 输出检查

默认 SynthMorph 流水线此前退出状态为 0，得到 91×109×91×490 的 float32 4D 影像，TR 0.735 秒；442,288,210 个值有限，掩膜外为 0。ICA 收敛于 96 个成分，AROMA 判定 48 个噪声成分。进程墙钟 532.22 秒，CUDA 峰值 reserved 17.58 GB；详见[原有整链摘要](e2e_summary.json)。

联合优化版本的 FNIRT volume 流水线退出状态为 0，输出尺寸和 TR 相同，模板网格及 gzip CRC 检查通过；442,288,210 个数值全部有限，掩膜外最大绝对值为 0。ICA 收敛于 95 个成分，AROMA 判定 63 个噪声成分。分步耗时合计 1194.98 秒，进程墙钟 1202.18 秒；CUDA 峰值 reserved 7.31 GB。两次运行均处于共享 GPU 环境，不作速度排序。FNIRT 结果及源文件、输入和输出哈希见[当前整链报告](fmri_volume_fnirt_20260929.public.json)。两个分支均使用 AROMA 代替 FIX，清理后的 MNI 影像没有可直接逐体素比较的 UKB FIX 输出。
