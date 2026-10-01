# T1→MNI152 2 mm 配准与 BOLD 重采样

`register_t1_to_mni` 先用 FNIT `TorchFLIRT` 求 T1→模板的 12 自由度初始矩阵，再从 `SynthMorph` 或 `TorchFNIRT` 中选一个方法估计非线性形变。两个后端都写出 **MNI 网格上指向 T1 的位移场**，供共享的 `resample_world` 与 EPI→T1 BBR、逐帧运动变换组合。影像由 NiBabel 读写，重采样由 PyTorch 完成。

`resample_world` 同时用于以下两种输出：

- `preproc`：从原始或 minimal BOLD 直接采样到 T1w/MNI，将每帧 HMC 与固定变换组合，只做一次空间插值。FNIT 默认关闭 slice timing；显式开启时，minimal BOLD 是已完成 STC、尚未做 HMC 的序列。
- `clean`：读取已在原生 EPI 空间完成运动校正和去噪的 BOLD，再采样到目标网格；这一步不再次应用 HMC。

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

| `resample_world` 参数 | 含义与默认值 |
|---|---|
| `source` | 要重采样的有限值 3D/4D NIfTI。`preproc` 使用原始/minimal BOLD；`clean` 使用已校正、已去噪的原生 EPI 序列。 |
| `reference` | 3D 目标 NIfTI，定义输出的空间 shape、affine 和体素大小；图像强度不参与采样。 |
| `reference_to_source_world` | 有限的 4×4 RAS 毫米矩阵。EPI 输入传 `np.linalg.inv(bbr.moving_to_fixed_world)`；T1 输入应用 MNI→T1 位移时传单位矩阵。 |
| `output` | 输出 NIfTI 路径，父目录自动创建；返回写出的绝对 `Path`。 |
| `pre_affine_pull_ras` | 默认 `None`。目标网格 `X×Y×Z×3` 位移 NIfTI，三个分量为 RAS 毫米；先加到目标 world 坐标，再应用固定矩阵。可传 `T1MNIResult.pull_ras`。原生场保持 float32；外部 float64 场读入既有 float64 坐标张量时保留原值。 |
| `output_mask` | 默认 `None`。与目标同网格的 3D NIfTI，值大于 0.5 的体素保留；其余位置强制输出零。 |
| `interpolation` | 默认 `"linear"`；也可用 `"nearest"` 或 `"spline"`。BOLD 的 `preproc`/`clean` 用三次 B 样条，连续组织图用线性，标签和二值掩膜用最近邻。 |
| `boundary` | 默认 `"grid-constant"`，源图像外按零扩展。`"periodic"` 保留既有 clean/FSL 样条系数边界及源 FOV 裁剪行为，见下文。 |
| `motion_pull_world` | 默认 `None`，不应用逐帧运动。需要 HMC 时传 `(帧数, 4, 4)` 有限数组，每个矩阵从参考 EPI world 指向该原始帧 world；3D 输入对应一帧。源文件应尚未做 HMC，避免重复校正。 |
| `coordinate_precision` | 默认 `"float64"`，保持既有 world 坐标组合。`"fmriprep"` 遵循固定 fMRIPrep 25.2.4/nitransforms 25.1.0 的坐标顺序：目标 world 和每次仿射入口转 float32，形变查询使用网格 deformation，逐帧 HMC 在源体素坐标中组合。FNIT 运行时只使用 NumPy/PyTorch。 |
| `spatial_chunk_size` | 默认 `262144`，每次样条查询的目标空间体素数，必须为正整数；只改变查询分块，不改变输出网格。 |
| `batch_size` | 默认 `8`，每批处理的时间帧数，必须为正整数；不混合或过滤时间轴。 |
| `device` | PyTorch 设备，如 `"cuda:0"` 或 `"cpu"`；默认 `None` 时优先 CUDA。 |

3D 输入生成 3D float32 NIfTI，4D 输入生成同帧数的 float32 NIfTI。空间 header 来自 `reference`，4D 的 TR 和时间单位来自 `source`。位移场和输出 mask 的 shape、affine 必须与目标一致；位移和源影像中的非有限值会报错。BOLD 的有限负值保留。

`grid-constant` 样条按 SciPy 的三次 B 样条零扩展定义：先在三个空间轴补 12 个零体素，再用 float64 镜像系统求系数和查询；输入影像及保存结果为 float32。`linear`/`nearest` 使用零扩展的 `grid_sample`。这一分支不把微小图像外坐标夹回边界。

`periodic` 样条使用 float32 周期系数。三种插值在这个显式分支都保留源 FOV 规则：源体素坐标落在 `[−1e-6, N−1+1e-6]` 时夹回 `[0, N−1]`，更远的图像外位置置零。该规则消除斜切 affine 求逆产生的边界舍入误差；[既有 clean 重采样报告](../../validation/fmri/resampling.md#边界回归测试)使用这个分支。

共享子函数 `sample_cubic_periodic_fast` 原先在长度为 1/2 的镜像轴、长度为 1 的周期轴上会因两体素 padding 报错；长度为 1 的镜像坐标还会产生零周期。现在这些短轴使用显式反射/环绕索引，单体素镜像轴保持常数。常规尺寸仍执行原来的 `F.pad`，FSL 负坐标索引规则保持原实现定义。三个短轴网格和一个常规网格、两种边界共 8 项独立 SciPy 系数查询对照通过；另有 2 项仿射/HMC 与形变网格精度控制通过。该修复同时作用于复用这个子函数的 EDDY 与 fMRI 路径。

`resample_world` 读入外部位移场时曾一律转 float32，丢失 float64 场的坐标精度；现在保留原值后进入既有 float64 坐标张量。内部生成的 float32 场仍取完全相同的数值。对默认关闭 STC 的真实 490 帧完整流程，使用同一原始 BOLD、HMC、BBR 和 MNI 场，仅重跑修改后的两个 `preproc` 插值阶段：T1w 的 136,917,760 个样本与 MNI 的 442,288,210 个样本均逐值复现已完成输出，RMSE 和最大绝对差均为 0。两个阶段另与镜像内串行 `resample_image` 比较，最大绝对差均为 0.00048828125。输入、源码 SHA 和完整指标见[float32 原生场不变性报告](../../validation/fmri/fmriprep/native_float32_resampler_invariance_full490.public.json)。

2026-10-01 复核了历史 clean 整链 `3b9b0f8` 的实际输出（`boundary="periodic"`）：取前 8 个真实 BOLD 时间点，固定该版本估计的 BBR、非线性位移场及目标脑掩膜，与原 FSL `applywarp --rel --interp=spline` 比较。脑掩膜内 r=0.999999999929，MAE=0.001461、RMSE=0.002063，最大绝对差 0.05496；输出网格、float32、TR 0.735 s、有限值及掩膜外零值均通过检查。对应的[8 帧报告](../../validation/fmri/volume_fixed_resampling.public.json)记录输入和源码哈希、FSL 耗时及退出码。该报告验证固定变换下的插值；当前 BBR/FNIRT 执行优化与解剖缓存的测量见[当前配准报告](../../validation/fmri/registration_gpu.current.public.json)。

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
    interpolation="spline",  # 三次 B 样条
    boundary="periodic",  # 显式保留上述 clean/FSL 对照的边界口径
    motion_pull_world=None,  # clean 输入已做 HMC，这里不重复应用
    coordinate_precision="float64",  # 保留原 clean 的 world 坐标组合
    spatial_chunk_size=262144,  # 每批空间查询的体素数
    batch_size=8,  # 每批 8 个时间帧；显存紧张可减小
    device="cuda:0",  # 与配准相同的 GPU，也可使用 CPU
)
```

### 原始/minimal BOLD 的单次空间插值

下面复用上例的 `bbr` 和 `registration`，将尚未做 HMC 的同一 run 直接采样到 MNI。运动矩阵须由同一 run 生成，并与全部源帧一一对应。

```python
motion_pull_world = np.load(
    "/absolute/path/motion_pull_world.npy",  # shape 为 (时间帧数, 4, 4)
    allow_pickle=False,
)
preproc_mni = resample_world(
    source="/absolute/path/minimal_bold.nii.gz",  # 默认使用原始 BOLD；主动开启 STC 时使用 STC 后序列
    reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 最终 3D 目标网格
    reference_to_source_world=np.linalg.inv(bbr.moving_to_fixed_world),  # T1 world→参考 EPI world
    output="/absolute/path/preproc_MNI152_2mm_bold.nii.gz",  # 保留全部源帧和 TR
    pre_affine_pull_ras=registration.pull_ras,  # MNI world→T1 world
    output_mask=None,  # 保留完整目标网格的插值结果
    interpolation="spline",  # 三次 B 样条空间采样
    boundary="grid-constant",  # 源图像外零扩展
    motion_pull_world=motion_pull_world,  # 固定变换之后应用每帧 HMC
    coordinate_precision="fmriprep",  # 按固定官方 resampler 的坐标精度与组合顺序
    spatial_chunk_size=262144,  # 查询分块
    batch_size=4,  # 每批四帧
    device="cuda:0",  # 计算设备
)
```

这段示例定义固定变换的组合和插值。`preproc` 在完整 volume 流程中使用此模式；`clean` 保留上述显式 `periodic` 与 float64 world 组合。

### 固定变换的真实 490 帧对照

2026-10-01，固定默认关闭 STC 的真实 run 全部 490 帧、FNIT 已估计的 HMC/BBR/MNI 变换与目标网格，比较更新后的 GPU 单次插值和用户提供的 fMRIPrep 25.2.4 镜像内串行 `resample_image`。输入为原始 BOLD，未重建 STC 文件。两个阶段同时逐值复现 `c3c921cc` 已完成的 `preproc`，最大绝对差均为 0。

| 目标网格 | 完整样本数 | Pearson r | RMSE，原强度单位 | 最大绝对差 | FNIT 耗时，含读写 | PyTorch 峰值 allocation | 官方函数线程 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1w，59×74×64×490 | 136,917,760 | ≈1.0 | 4.80×10⁻⁸ | 0.00048828125 | 37.97 s | 0.922 GB | 1 |
| MNI 2 mm，91×109×91×490 | 442,288,210 | ≈1.0 | 5.51×10⁻⁸ | 0.00048828125 | 99.19 s | 1.062 GB | 1 |

两张输出均通过完整帧数、目标 shape/affine、float32、有限值和原 BIDS TR 0.735 s 检查；预设门禁为最大绝对差 ≤0.01、相对 RMSE ≤10⁻⁶、r ≥0.999999。候选使用共享 H100 PCIe、8 CPU 线程、TF32 和 20 GB allocator 上限。此表绑定更新后的 `normalization` SHA `268f61b2`；原完整 API 的 697.313 s 绑定 `c3c921cc`。逐文件 SHA、坐标模式和各空间指标见[默认关闭 STC 的聚合报告](../../validation/fmri/fmriprep/native_float32_resampler_invariance_full490.public.json)。

另从真正完成的官方流程读取两个保留的 `resample` 节点，固定其输入、目标网格、变换文件及逆向参数。官方 `in_file` 与原始 BOLD 的完整有序 float32 数组 SHA 相同，490 帧逐值一致、affine 相同。导出的 float64 场在整个目标网格恢复有效 float32 world 坐标和源体素坐标的误差均为 0。该控制使用官方 T1w 目标 57×73×60，与上表 FNIT 的 T1w 网格不同。

| 固定官方目标及变换 | FNIT 对实际保存节点的 r | RMSE | 最大绝对差 | 原预设数值门禁 | FNIT 耗时，含读写 |
|---|---:|---:|---:|---|---:|
| T1w，57×73×60×490 | 1.0 | 8.86×10⁻⁸ | 0.0009765625 | 通过 | 32.96 s |
| MNI 2 mm，91×109×91×490 | 0.999958709 | 33.6142 | 22930.4144 | 未通过 | 107.94 s |

T1w 的独立串行源码回放与实际节点 8 线程输出逐值一致。MNI 的实际输出有 12/490 帧无法由相同保留输入和源码严格复现；这些帧均有限且不为空，没有排除任何帧。最差帧索引 22（从 0 计）在 BLAS 1/8 线程的直接函数控制中均逐值复现串行回放，有限空间采样未检出精确换帧，原因尚未确定。FNIT 对独立串行 MNI 回放的 r=1、RMSE=1.46×10⁻⁸、最大绝对差 0.000244140625；对实际保存节点的验收仍记为未通过。

[实际节点完整报告](../../validation/fmri/fmriprep/actual_node_interpolation_full490.public.json)分开记录两个空间的门禁；[MNI 失败诊断](../../validation/fmri/fmriprep/actual_node_mni_replay_failure.public.json)保留全部输入与变换哈希、12 帧质量统计和有限控制。上述测试固定变换，仅检验组合与插值。早期显式开启 STC 的控制及其参考可重复性记录见[历史报告](../../validation/fmri/fmriprep/held_interpolation_full490.public.json)。

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

该缓存测试包含 FNIT 自身的 FLIRT 初始化，因此不是上面固定 FSL affine 的同输入 FNIRT 比较。BBR 按 BOLD run 单独计算。缓存位置、失效条件与 `reuse_anatomical=False` 见[volume 输入输出](README.md#输出)。本节独立 FNIRT 与缓存测试未测量 SynthMorph；当前 SynthMorph 完整 volume 的重跑结果见[全流程 benchmark](README.md#全流程-benchmark)，两者分别计时。

## 原实现与参考文献

- 固定 fMRIPrep 25.2.4 [单次重采样源码](https://github.com/nipreps/fmriprep/blob/25.2.4/fmriprep/interfaces/resampling.py)。其原始文件 SHA 与实际镜像一致；官方软件只作为上述独立对照。
- 固定 NiTransforms 25.1.0 [仿射坐标精度与映射](https://github.com/nipy/nitransforms/blob/25.1.0/nitransforms/linear.py)、[形变网格查询](https://github.com/nipy/nitransforms/blob/25.1.0/nitransforms/nonlinear.py)。FNIT 用自己的 NumPy/PyTorch 路径实现对应坐标组合，运行时不导入 NiTransforms。
- Esteban et al. *fMRIPrep: a robust preprocessing pipeline for functional MRI*. Nature Methods (2019), [doi:10.1038/s41592-018-0235-4](https://doi.org/10.1038/s41592-018-0235-4)。
- FSL [FNIRT 原文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)；配准算法、原实现与文献详见 [FNIRT](../fnirt/README.md)、[FLIRT](../flirt/README.md)与 [SynthMorph](../synthmorph/README.md)功能页。
