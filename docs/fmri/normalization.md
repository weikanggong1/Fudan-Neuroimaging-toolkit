# T1→MNI152 2 mm 配准与 BOLD 重采样

`register_t1_to_mni` 先用 FNIT `TorchFLIRT` 求 T1→模板的 12 自由度初始矩阵，再从 `SynthMorph` 或 `TorchFNIRT` 中选一个方法估计非线性形变。运行时不启动 FSL 或 FreeSurfer 可执行程序；现有 FNIT SynthMorph/FNIRT 后端仍以 Surfa `Volume`、`Affine`、`Warp` 管理几何与变换，SynthMorph 内部还用 Surfa 在 CPU 上重采样 T1；MNI 位移场与最终 BOLD 由 NiBabel 写出。两个后端都把最终变换写成 **MNI 网格上指向 T1 的位移场**，供 `resample_world` 与 EPI→T1 的 BBR 合成。BOLD 最终只插值一次。

## 输入与输出

| `register_t1_to_mni` 参数 | 含义 |
|---|---|
| `t1_brain` | 待配准的同被试 3D 去颅骨 T1 NIfTI 文件。由 SynthStrip 生成；这里不接收 4D 影像。 |
| `mni_brain` | 3D MNI152 2 mm 脑模板文件，定义配准的目标影像、输出形变网格。调用者应提供 2 mm 模板；此低层函数只检查它是 3D。 |
| `output_dir` | 输出文件夹；函数创建目录并写出 `.mat` 与 3D 向量 NIfTI。 |
| `backend` | `"synthmorph"` 使用 FNIT SynthMorph deform 权重；`"fnirt"` 使用 FNIT PyTorch FNIRT。默认 `"synthmorph"`。 |
| `synthmorph_weights` | deform 权重文件的绝对路径；只对 `backend="synthmorph"` 有效。`None` 时按 FNIT 权重配置解析。 |
| `reference_mask` | 可选的 `mni_brain` 网格二值掩膜，只供 PyTorch FNIRT 使用；SynthMorph 不读取它。 |
| `device` | 如 `"cuda:0"` 或 `"cpu"`；`None` 时优先 CUDA。GPU 默认允许 TF32，图像与形变使用 float32，不启用 float16。 |

返回 `T1MNIResult`。`affine` 是 **T1→MNI 的 FSL scaled-mm 初始矩阵文件** `T1_to_MNI152_2mm_affine.mat`；`moving_to_fixed_world` 是对应的 RAS world 4×4 NumPy 数组。`pull_ras` 是 `MNI152_2mm_to_T1_pull_ras.nii.gz`，shape 为 MNI `X×Y×Z×3`，三个分量单位是 RAS 毫米。对某个 MNI world 坐标 `p`，对应 T1 world 坐标为 `p + pull_ras(p)`。这个位移场已经包含初始仿射与非线性形变；应用它时**不要再叠加 `affine`**。`backend` 记录选择的方法。

| `resample_world` 参数 | 含义 |
|---|---|
| `source` | 要重采样的 3D/4D NIfTI；对最终 BOLD 是已清理的原生 EPI 空间 4D 文件。 |
| `reference` | 3D 目标模板 NIfTI；决定输出的空间维度、affine 与体素大小。 |
| `reference_to_source_world` | 4×4 RAS world 矩阵。若 `source` 是 EPI，传 EPI→T1 BBR world 矩阵的逆；若 `source` 是 T1，传单位矩阵。 |
| `output` | 写出的绝对路径；3D 输入生成 3D NIfTI，4D 输入生成 4D NIfTI，时间步长来自 `source` header。 |
| `pre_affine_pull_ras` | 可选的目标网格 `X×Y×Z×3` 位移 NIfTI；传 `T1MNIResult.pull_ras` 时，先将目标 MNI world 坐标加上位移，再应用上面的 4×4 矩阵。 |
| `output_mask` | 可选的目标网格 3D 二值 NIfTI；掩膜外输出强制为 0，默认 `None`。 |
| `interpolation` | `"linear"`（默认）或 `"nearest"`。BOLD 用线性，二值标签宜用最近邻。 |
| `batch_size` | 4D 输入每次送入 `grid_sample` 的帧数，默认 8；不改变输出网格。 |
| `device` | PyTorch 设备；`None` 时优先 CUDA。 |

`resample_world` 返回写出的 `Path`。输出数组是 float32，空间 header 来自 `reference`，4D 输出的 TR 和时间单位来自 `source`。它要求位移场的形状和 affine 与 `reference` 一致。

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
)
registration = register_t1_to_mni(
    t1_brain="/absolute/path/sub-0001_T1w_brain.nii.gz",  # 同被试 3D 去颅骨 T1
    mni_brain="/absolute/path/MNI152_T1_2mm_brain.nii.gz",  # 3D MNI152 2 mm 脑模板
    output_dir="/absolute/path/sub-0001/reg",  # 仿射矩阵、MNI→T1 位移场输出目录
    backend="synthmorph",  # 可改成 "fnirt"；两者输出结构相同
    synthmorph_weights="/absolute/path/synthmorph.deform.3.h5",  # deform 模型权重；fnirt 不使用
    reference_mask="/absolute/path/MNI152_T1_2mm_brain_mask.nii.gz",  # fnirt 使用的模板脑掩膜
    device="cuda:0",  # 计算设备；无 GPU 时填写 "cpu"
)

clean_mni = resample_world(
    source="/absolute/path/filtered_func_data_clean_epi.nii.gz",  # 原生 EPI 空间的清理后 4D BOLD
    reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 3D MNI152 2 mm 最终输出网格
    reference_to_source_world=np.linalg.inv(bbr.moving_to_fixed_world),  # T1 world→EPI world 的 BBR 逆矩阵
    output="/absolute/path/filtered_func_data_clean_MNI152_2mm.nii.gz",  # 输出 4D BOLD 文件
    pre_affine_pull_ras=registration.pull_ras,  # MNI world→T1 world 的完整位移场
    output_mask="/absolute/path/MNI152_2mm_brain_mask.nii.gz",  # 目标网格二值掩膜，掩膜外置零
    interpolation="linear",  # 4D BOLD 使用三线性插值
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
  --out=filtered_func_data_clean_MNI152_2mm.nii.gz
```

固定**与 FNIT 完全相同的两张去颅骨输入**时，可把上面 `fnirt` 的 `--in`、`--ref` 换成 `T1_brain.nii.gz`、`MNI152_T1_2mm_brain.nii.gz`，并显式提供模板掩膜。这是实现差异测试；它与 FSL 推荐的整头 FNIRT 输入并不相同。

## 两个后端的对应边界

SynthMorph 使用学习得到的 deform 网络，在初始 FLIRT 仿射上估计形变；它不是 FNIRT 优化器。PyTorch FNIRT 复用 FNIT 的 B 样条和 Gauss–Newton/LM 核心，并按 FSL `T1_2_MNI152_2mm.cnf` 设置多尺度采样、平滑和正则化；现有实现仍使用全局线性强度映射，而 FSL T1 配置使用非线性强度及偏置场模型。因此二者输出文件契约一致，不应声称与官方 FNIRT 逐体素一致。两后端都要求检查配准图像和脑缘重合程度，不能仅凭算法名称认定配准有效。

## 真实数据 benchmark

本次用同一例 UK Biobank T1：BIDS NIfTI 由该例 FreeSurfer `orig/001.mgz` 转换，**不能核实它与扫描仪原始 T1 文件逐字节相同**。FNIT 和主对照 FSL 共用这张 T1 经 SynthStrip 提取的脑图，以及同一张 MNI152 2 mm 脑模板。FSL 版本为 6.0.7.22。另运行了 FSL 整头 T1→整头模板；该补充对照的 FLIRT 初始矩阵也用两张整头影像估计，不能与同输入 brain-to-brain 结果逐项比较，也不是上面的推荐命令。

| 同输入指标 | PyTorch SynthMorph | PyTorch FNIRT | FSL FLIRT+FNIRT |
|---|---:|---:|---:|
| 初始仿射 + 非线性配准耗时 | 156.32 s | 240.18 s | 11.25 + 150.53 = 161.78 s |
| 加上 FNIT 结果图重采样 | 156.71 s | 240.67 s | `fnirt --iout` 已包含输出图 |
| 峰值内存 | GPU allocated 12.38 GiB，reserved 18.12 GiB | GPU allocated 0.653 GiB，reserved 0.900 GiB | FNIRT CPU RSS 0.798 GiB |
| MNI 脑内输出强度与 FSL Pearson r | 0.8323 | 0.9135 | 参照 |
| 输出脑支持区与 FSL Dice | 0.9782 | 0.9902 | 参照 |
| MNI→T1 pull 坐标差 | 中位 1.75 mm，95 百分位 5.01 mm | 中位 1.02 mm，95 百分位 3.41 mm | 参照 |
| 与 MNI T1 模板强度 Pearson r | 0.7938 | 0.8158 | 0.8025 |

Pearson r 在官方 MNI 脑掩膜内计算。脑支持区把每张重采样 T1 的正值第 99 百分位乘以 0.05 作阈值；两张图的二值区求 Dice。坐标差用 FSL `applywarp` 对 T1 的三个 RAS world 坐标图重采样，再与 FNIT 的完整 pull 场比较；只纳入双方均有有效脑信号的体素。相比输出强度，坐标差能直接检验仿射与非线性形变的合成方向。FSL 整头输入单独耗时为 FLIRT 17.41 s、FNIRT 194.34 s，其配准后脑内强度与模板 r=0.7833；整头结果含头皮，故未与去颅骨结果计算脑支持 Dice。

这是一例真实数据。PyTorch FNIRT 行为当前源码在同输入上的复测；TorchFLIRT 本次更新仅涉及 6 自由度分支，T1→MNI 使用的 12 自由度分支未改。复测的初始仿射矩阵、完整位移场和重采样 T1 与此前同输入结果逐值相同（最大绝对差均为 0）。SynthMorph 行来自该后端未变更时的同输入运行。FNIT 在共享 GPU 上与其他作业同时运行；上表是观察到的耗时，**不能用来作公平的 CPU/GPU 速度排序**。本机 FSL 的 FLIRT/FNIRT 在产出文件后返回 255；已保留该退出码，并核验了新产出 NIfTI 的 gzip CRC、网格、有限值及仿射矩阵可逆性，条件接受为数值参照。两支 FNIT 输出均通过相同的 NIfTI 有限值和 gzip CRC 检查。完整标量、定义及源码 SHA256 见 [`registration_summary.json`](../../validation/fmri/registration_summary.json)；公开文件不含原图、被试标识或逐体素结果。
