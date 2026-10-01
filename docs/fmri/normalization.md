# T1→MNI152 2 mm 配准与 BOLD 重采样

`register_t1_to_mni` 先用 FNIT `TorchFLIRT` 求 T1→模板的 12 自由度初始矩阵，再从 `SynthMorph` 或 `TorchFNIRT` 中选一个方法估计非线性形变。几何与变换由 FNIT 的 `AffineTransform`、`DenseWarp` 管理，重采样在 PyTorch 中计算，影像由 NiBabel 读写。两个后端都把最终变换写成 **MNI 网格上指向 T1 的位移场**，供 `resample_world` 与 EPI→T1 的 BBR 合成。清理后的原生 EPI BOLD 通过这个合成变换一次重采样到 MNI，完整 volume 流程使用 GPU 三次 B 样条。前序运动校正单独进行一次重采样。

## 输入与输出

| `register_t1_to_mni` 参数 | 含义 |
|---|---|
| `t1_brain` | 待配准的同被试 3D 去颅骨 T1 NIfTI 文件。由 SynthStrip 生成；这里不接收 4D 影像。 |
| `mni_brain` | 3D MNI152 2 mm 脑模板文件，定义配准的目标影像、输出形变网格。调用者应提供 2 mm 模板；此低层函数只检查它是 3D。 |
| `output_dir` | 输出文件夹；函数创建目录并写出 `.mat` 与 3D 向量 NIfTI。 |
| `backend` | `"synthmorph"` 使用 FNIT SynthMorph deform 权重；`"fnirt"` 使用 FNIT PyTorch FNIRT。默认 `"synthmorph"`。 |
| `synthmorph_weights` | deform 权重文件的绝对路径；只对 `backend="synthmorph"` 有效。`None` 时按 FNIT 权重配置解析。 |
| `reference_mask` | 可选的 `mni_brain` 网格二值掩膜，只供 PyTorch FNIRT 使用；SynthMorph 不读取它。 |
| `fnirt_config` | FNIRT 预设名称或 `FNIRTConfig` 对象；`backend="fnirt"` 时默认 `"t1"`。可用 `dataclasses.replace(T1FNIRTConfig(), ...)` 更改参数；SynthMorph 分支不接收此选项。 |
| `fnirt_execution` | `"optimized"`（默认）用缓存、GPU 平滑与系数空间算子；`"reference"` 保留串行平滑、dense 弯曲算子及原 PCG 执行方式。两者使用相同配置、精度和停止准则；切换只适用于 FNIRT。 |
| `device` | 如 `"cuda:0"` 或 `"cpu"`；`None` 时优先 CUDA。GPU 默认允许 TF32，图像与形变使用 float32，不启用 float16。 |

返回 `T1MNIResult`。`affine` 是 **T1→MNI 的 FSL scaled-mm 初始矩阵文件** `T1_to_MNI152_2mm_affine.mat`；`moving_to_fixed_world` 是对应的 RAS world 4×4 NumPy 数组。`pull_ras` 是 `MNI152_2mm_to_T1_pull_ras.nii.gz`，shape 为 MNI `X×Y×Z×3`，三个分量单位是 RAS 毫米。对某个 MNI world 坐标 `p`，对应 T1 world 坐标为 `p + pull_ras(p)`。这个位移场已经包含初始仿射与非线性形变；应用它时**不要再叠加 `affine`**。`backend` 记录选择的方法；`qc` 在 FNIRT 分支返回逐层优化、强度多项式、偏置场范围和显存精度设置，SynthMorph 分支为 `None`。`timing_seconds` 分开返回 `t1_to_mni_affine`、`t1_to_mni_nonlinear`、`warp_conversion`，仅在阶段边界同步 GPU。完整流程在 BIDS BOLD 的 JSON 中记录 QC 与这些阶段耗时。

| `resample_world` 参数 | 含义 |
|---|---|
| `source` | 要重采样的 3D/4D NIfTI；对最终 BOLD 是已清理的原生 EPI 空间 4D 文件。 |
| `reference` | 3D 目标模板 NIfTI；决定输出的空间维度、affine 与体素大小。 |
| `reference_to_source_world` | 4×4 RAS world 矩阵。若 `source` 是 EPI，传 EPI→T1 BBR world 矩阵的逆；若 `source` 是 T1，传单位矩阵。 |
| `output` | 写出的绝对路径；3D 输入生成 3D NIfTI，4D 输入生成 4D NIfTI，时间步长来自 `source` header。 |
| `pre_affine_pull_ras` | 可选的目标网格 `X×Y×Z×3` 位移 NIfTI；传 `T1MNIResult.pull_ras` 时，先将目标 MNI world 坐标加上位移，再应用上面的 4×4 矩阵。 |
| `output_mask` | 可选的目标网格 3D 二值 NIfTI；掩膜外输出强制为 0，默认 `None`。 |
| `interpolation` | `"linear"`（低层函数默认）、`"nearest"` 或 `"spline"`。完整 volume 的最终 MNI BOLD 固定用 `"spline"`；连续组织图仍用线性，二值标签用最近邻。 |
| `batch_size` | 4D 输入每次送入 GPU 的帧数，默认 8；样条系数按帧计算，不改变时间轴或输出网格。 |
| `device` | PyTorch 设备；`None` 时优先 CUDA。 |

`resample_world` 返回写出的 `Path`。输出数组是 float32，空间 header 来自 `reference`，4D 输出的 TR 和时间单位来自 `source`。位移场的形状和 affine 必须与 `reference` 一致。`spline` 在 float32 下用 PyTorch FFT 求各帧的空间三次 B 样条系数，再复用 FNIT 的 GPU 采样核；不会过滤时间轴。样条系数采用周期边界，回归后 BOLD 的负值保留。组织概率和 ICA 图用线性采样，二值掩膜用最近邻采样。

三种插值共享同一套源网格边界规则。对某轴长度 `N`，有效坐标范围为 `[−1e-6, N−1+1e-6]`，单位是**源体素**；落在容差内的微小越界坐标夹回 `0` 或 `N−1`。这样可消除斜切 affine 求逆时产生的边界舍入误差。超过容差的真实图像外坐标、以及输出掩膜外位置仍置零，容差不会扩展输出视野。边界修复与验证见[重采样报告](../../validation/fmri/resampling.md#边界回归测试)。

2026-10-01 复核了当前整链 `3b9b0f8` 的实际输出：取前 8 个真实 BOLD 时间点，固定该版本估计的 BBR、非线性位移场及目标脑掩膜，与原 FSL `applywarp --rel --interp=spline` 比较。脑掩膜内 r=0.999999999929，MAE=0.001461、RMSE=0.002063，最大绝对差 0.05496；输出网格、float32、TR 0.735 s、有限值及掩膜外零值均通过检查。对应的[8 帧报告](../../validation/fmri/volume_fixed_resampling.public.json)记录输入和源码哈希、FSL 耗时及退出码。该报告验证固定变换下的插值；当前 BBR/FNIRT 执行优化与解剖缓存的测量见[当前配准报告](../../validation/fmri/registration_gpu.current.public.json)。

```python
import numpy as np
from fnit.fmri.bbr import register_bbr
from fnit.fmri.normalization import register_t1_to_mni, resample_world

bbr = register_bbr(
    epi="/absolute/path/example_func.nii.gz",  # 同被试 3D EPI 参考影像
    t1="/absolute/path/sub-0001_T1w_brain.nii.gz",  # 同被试 3D 去颅骨 T1
    wmseg="/absolute/path/sub-0001_T1w_wmseg.nii.gz",  # T1 网格 3D 白质二值分割
    init=None,  # EPI→T1 初始 FLIRT 矩阵；None 由 TorchFLIRT 估计
    device="cuda:0",  # BBR 的计算设备
    grid_search=True,  # 先进行白质边界粗网格搜索
    execution="batched",  # 在 GPU 上同时计算相互独立的候选矩阵代价
    candidate_batch_size=128,  # 每批候选矩阵数量
)
registration = register_t1_to_mni(
    t1_brain="/absolute/path/sub-0001_T1w_brain.nii.gz",  # 同被试 3D 去颅骨 T1
    mni_brain="/absolute/path/MNI152_T1_2mm_brain.nii.gz",  # 3D MNI152 2 mm 脑模板
    output_dir="/absolute/path/sub-0001/reg",  # 仿射矩阵、MNI→T1 位移场输出目录
    backend="fnirt",  # 使用 T1FNIRTConfig；改成 "synthmorph" 使用 deform 网络
    synthmorph_weights=None,  # fnirt 不读取该权重；synthmorph 分支填写 deform 权重或使用缓存
    reference_mask="/absolute/path/MNI152_T1_2mm_brain_mask.nii.gz",  # fnirt 使用的模板脑掩膜
    fnirt_config="t1",  # FNIRT 预设；省略时仍为 t1，可传修改后的 T1FNIRTConfig 对象
    fnirt_execution="optimized",  # GPU 执行优化；reference 可逐项复核同一算法
    device="cuda:0",  # 计算设备；无 GPU 时填写 "cpu"
)

clean_mni = resample_world(
    source="/absolute/path/filtered_func_data_clean_epi.nii.gz",  # 原生 EPI 空间的清理后 4D BOLD
    reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 3D MNI152 2 mm 最终输出网格
    reference_to_source_world=np.linalg.inv(bbr.moving_to_fixed_world),  # T1 world→EPI world 的 BBR 逆矩阵
    output="/absolute/path/filtered_func_data_clean_MNI152_2mm.nii.gz",  # 输出 4D BOLD 文件
    pre_affine_pull_ras=registration.pull_ras,  # MNI world→T1 world 的完整位移场
    output_mask="/absolute/path/MNI152_2mm_brain_mask.nii.gz",  # 目标网格二值掩膜，掩膜外置零
    interpolation="spline",  # 最终 MNI BOLD 使用三次 B 样条，减少采样位置造成的 SD 格纹
    batch_size=8,  # 每批 8 个时间帧；显存紧张可减小
    device="cuda:0",  # 与配准相同的 GPU，也可使用 CPU
)
```

上例的 `bbr.moving_to_fixed_world` 是 EPI→T1 的 4×4 RAS world NumPy 数组。`reference` 和 `mni_brain` 可以是整头模板及去颅骨模板，但两者必须处于**同一体素网格**。`registration.affine` 是初始配准记录，不能同时代入 `resample_world` 的第三个参数。

## 官方对照命令

FSL 的常规 T1→MNI 流程使用整头 T1 和整头 MNI 模板，并以两张去颅骨影像求 FLIRT 初始矩阵。以下命令中的 FSL 仅用于独立 benchmark；FNIT 运行时不调用它。FSL [FNIRT 指南](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)推荐整头输入，`T1_2_MNI152_2mm.cnf` 包含多尺度、强度映射和偏置场参数。

```bash
flirt -in T1_brain.nii.gz -ref MNI152_T1_2mm_brain.nii.gz \
  -dof 12 -omat T1_to_MNI_affine.mat
fnirt --in=T1.nii.gz --ref=MNI152_T1_2mm.nii.gz \
  --aff=T1_to_MNI_affine.mat --config=T1_2_MNI152_2mm \
  --cout=T1_to_MNI_coeff.nii.gz --iout=T1_in_MNI.nii.gz
applywarp --in=filtered_func_data_clean_epi.nii.gz \
  --ref=MNI152_T1_2mm.nii.gz --premat=example_func2highres.mat \
  --warp=T1_to_MNI_coeff.nii.gz \
  --out=filtered_func_data_clean_MNI152_2mm.nii.gz --interp=spline
```

`--interp=spline` 选择三次样条；不指定时 FSL `applywarp` 默认用三线性。下面的配准对照比较两种方法估计出的形变；[固定 warp 的插值对照](../../validation/fmri/resampling.md)则固定输入 BOLD 和变换，只比较重采样器。

固定**与 FNIT 完全相同的两张去颅骨输入**时，可把上面 `fnirt` 的 `--in`、`--ref` 换成 `T1_brain.nii.gz`、`MNI152_T1_2mm_brain.nii.gz`，并添加 `--refmask=MNI152_T1_2mm_brain_mask.nii.gz`。下面的实测对照使用这一组输入；它与 FSL 推荐的整头 FNIRT 输入不同。

## 两个后端的对应边界

SynthMorph 使用学习得到的 deform 网络。PyTorch FNIRT 复用 FNIT 的 B 样条和 Gauss–Newton/LM 核心，`T1FNIRTConfig` 采用官方六级采样、平滑、正则化及 `intorder=5` 的强度设置；后者表示常数项至四次项共 5 个系数。T1 分支在前五级用同一 LM/PCG 法方程联合优化形变、强度多项式和 50 mm 三次 B 样条乘性偏置场，最后一级固定强度参数。两后端共享输出网格及位移场定义，实际形变差异用下面的配准后 T1 强度、脑支持区和 RAS 坐标差量化。

## 当前真实数据 benchmark

固定一例真实已处理、去颅骨 T1 与 MNI152 2 mm 脑模板，使用完全相同的 FSL 初始矩阵和模板掩膜，2026-10-01 重跑官方 FSL 6.0.7.22 与 FNIT T1 六级非线性阶段。该 T1 的更早处理来源未知，不能视为扫描仪原始 T1。

| 指标 | 修改前 | 当前 optimized |
|---|---:|---:|
| FNIRT 首次 / 热调用 | 69.350 / 71.501 s | 32.595 / 30.422 s |
| 与 FSL warped T1 的 Pearson r | 0.99784173 | 0.99771788 |
| MAE / RMSE，原强度单位 | 4.97974 / 16.37197 | 4.87159 / 16.83218 |
| 脑支持 Dice | 0.99924899 | 0.99922162 |
| 完整 MNI→T1 pull median / p95 | 0.05322 / 0.23351 mm | 0.05176 / 0.23294 mm |
| 热调用峰值 allocated / reserved | 1.091 / 1.474 GB | 1.178 / 1.491 GB |

此表隔离 FNIRT，不包含 FLIRT 初始化或最终 4D BOLD 重采样。FSL CPU 命令观测为 217.558 s，实际子进程退出 0、输出与固定参照逐位一致；包装器 255 单独记录。FNIT 在共享 H100 上运行，函数钟包括输入解压和 CPU 输出转换，排除写盘/事后比较，不能与 CPU 命令直接计算稳定加速倍数。

当前 `reference` 在本例复现修改前的图像、系数、pull 和完整 Jacobian；默认 optimized 的 Gram 弯曲算子改变 FP64 求和顺序。图像 r/Dice 略降、RMSE 略增，MAE、pull median/p95 和 nonlinear Jacobian 改善，保留完整精度表及仅换回 dense 算子的逐位消融。需要原优化轨迹可设置 `fnirt_execution="reference"`。指标定义、系数/header 契约、profile 与局限见 [FNIRT 功能页](../fnirt/README.md#t1w-专用预设当前-gpu-修复版)及 [当前配准报告](../../validation/fmri/registration_gpu.current.public.json)。

### 跨 BOLD run 复用解剖预处理

完整 `fMRIVolume_pipeline` 默认缓存当前被试/会话的 T1 SynthStrip、FAST、模板准备与 T1→MNI。真实 T1 的首次解剖调用为 41.118 s，第二次完整输入/权重/输出哈希核验为 0.0785 s；所有产物 SHA-256 相同，命中后上述计算阶段均为 0。首次分段为 T1 提取 5.042 s、模板准备 0.081 s、FAST 1.957 s、T1→MNI affine 4.331 s、FNIRT 29.041 s、warp 转换/保存 0.502 s；峰值分配 4.683 GB。

该缓存测试包含 FNIT 自身的 FLIRT 初始化，因此不是上面固定 FSL affine 的同输入 FNIRT 比较。BBR 按 BOLD run 单独计算。缓存位置、失效条件与 `reuse_anatomical=False` 见[volume 输入输出](README.md#输出)。SynthMorph 可继续使用；本次没有重跑该后端，不混列其旧计时与新的 FNIRT 测量。
