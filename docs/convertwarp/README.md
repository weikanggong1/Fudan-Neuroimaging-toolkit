# TorchConvertWarp：组合线性矩阵与非线性场

`TorchConvertWarp` 生成标准空间网格上的 FSL pull 位移场。它支持两类输入：FSL dense/FNIRT 系数场及可选的前后 FLIRT 矩阵；FNIT MMORF 的参考图像轴毫米场及其独立的 FA→MNI FLIRT 矩阵。前一类按 `premat → warp1 → postmat` 组合；后一类先把 MMORF 坐标约定换成 FSL scaled-mm，再把 affine 和非线性变换合为一张场。矩阵不能直接用 NIfTI 的 RAS affine 代替。运行时只使用本包代码、Nibabel、NumPy 和 PyTorch，不调用 FSL。

FNIT dMRI pipeline 的 **TBSS** 输出 `registration/dti_FA_to_MNI_warp.nii.gz` 已经含 FA→MNI 线性部分；将它交给 `warp1` 时**不要**再传 `dti_FA_to_MNI_affine.mat`。**MMORF** 输出 `registration/mmorf_warp.nii.gz` 则是另一种坐标约定，必须使用下述 `from_mmorf` / `--mmorf-warp` 和同目录的 `dti_FA_to_MNI_affine.mat`。这两个分支共用后续 [TorchInvWarp](../invwarp/README.md)。

## Python 单被试调用

```python
from fnit import TorchConvertWarp

result = TorchConvertWarp(device="cuda:0").run(
    reference="/data/MNI152_T1_2mm.nii.gz",  # 输出场的 MNI 网格，对应 --ref
    warp1="/data/T1_to_MNI_warp.nii.gz",  # T1→MNI 的 FSL dense warp 或 FNIRT 三次样条系数，对应 --warp1
    premat="/data/diff_to_T1.mat",  # diffusion→T1 的 FLIRT 4×4 scaled-mm 矩阵，对应 --premat
    postmat=None,  # 可选的 warp 目标→最终参考空间矩阵，对应 --postmat
    warp_convention="auto",  # 按 FSL intent 自动识别 warp1；也可指定 relative 或 absolute
    output_convention="relative",  # 输出相对位移；absolute 输出绝对 pull 坐标
    output="/data/diff_to_MNI_warp.nii.gz",  # 输出 4D NIfTI，末轴长 3，float32，单位 mm
)
print(result.image.shape, result.valid_fraction, result.qc)
```

`result.image` 是保存后的 NIfTI 对象；`valid_fraction` 是采样 `warp1` 时落在其网格内的输出体素比例；`qc` 记录坐标约定和变换顺序。输出 shape 是 `reference.shape + (3,)`，affine、qform、sform 沿用 `reference`。相对场保存为 FSL `convertwarp --relout` 的 dense 形式；`warp1` 可为 FSL intent-2006 dense 位移或 intent-2007 三次样条系数。当前实现覆盖一张非线性场及其前后矩阵，不提供 FSL `--warp2`、shiftmap 或 Jacobian 选项。

### 直接使用 FNIT pipeline 的两类输出

```python
from fnit import TorchConvertWarp

model = TorchConvertWarp(device="cuda:0")  # PyTorch 计算设备
tbss = model.run(
    reference="/data/subject_tbss/registration/standard/FA.nii.gz",  # TBSS MNI 网格
    warp1="/data/subject_tbss/registration/dti_FA_to_MNI_warp.nii.gz",  # 已含 affine 的 FNIRT 系数
    premat=None,  # TBSS 不再次应用 dti_FA_to_MNI_affine.mat
    output="/data/subject_tbss/diff_to_MNI_FSL_warp.nii.gz",  # dense 相对场
)
mmorf = model.run_mmorf(
    reference="/data/subject_mmorf/registration/standard/FA.nii.gz",  # MMORF 的 MNI 网格
    source="/data/subject_mmorf/native/dti_FA.nii.gz",  # FLIRT 矩阵的 native FA 输入网格
    mmorf_warp="/data/subject_mmorf/registration/mmorf_warp.nii.gz",  # 参考图像轴 mm 场
    affine="/data/subject_mmorf/registration/dti_FA_to_MNI_affine.mat",  # native FA→MNI 矩阵
    output="/data/subject_mmorf/diff_to_MNI_FSL_warp.nii.gz",  # FSL scaled-mm dense 相对场
)
```

`run_mmorf` 的 `source` 是产生 affine 的 FA 原图；若概率追踪使用的 BEDPOSTX mask 不与它同 shape 和 affine，须先取得两者之间的配准，不能直接复用这个场。`mmorf.image`、`mmorf.valid_fraction`、`mmorf.qc` 与标准 `run()` 的返回结构相同。MMORF 原场不能直接输入 FSL `convertwarp`；`run_mmorf` 是 FNIT 额外提供的坐标转换。单被试 CLI 对应 `fnit convertwarp --ref ... --source ... --mmorf-warp ... --premat ... --out ... --device cuda:0`，其中 `--premat` 在此模式指 FA→MNI 矩阵。

## 命令行及 FSL 对应

```bash
# --ref 指定 MNI 输出网格；--warp1 指定 T1→MNI 非线性场。
# --premat 指定 diffusion→T1 线性矩阵；--out 写出组合场。
fnit convertwarp --ref /data/MNI152_T1_2mm.nii.gz \
  --warp1 /data/T1_to_MNI_warp.nii.gz \
  --premat /data/diff_to_T1.mat \
  --out /data/diff_to_MNI_warp.nii.gz --relout --device cuda:0

convertwarp --ref=/data/MNI152_T1_2mm.nii.gz \
  --warp1=/data/T1_to_MNI_warp.nii.gz \
  --premat=/data/diff_to_T1.mat \
  --out=/data/fsl_diff_to_MNI_warp.nii.gz --relout
```

独立入口 `fnit-convertwarp` 接受相同选项。`--absout` 改写为绝对 pull 坐标；`--rel` / `--abs` 可显式声明输入 dense warp 的约定；`--overwrite` 允许覆盖已有输出。输入系数场的 intent 仍决定其解释方式。

## 真实数据对照

在 gpucw1 上，用同一例真实 DWI 已保存的 FNIT pipeline 输出核对两条分支。TBSS 系数单独转换的 dense 场与 FSL `convertwarp --relout` 场均为 `182×218×182×3`、float32，affine 相同；位移分量 MAE `1.984×10⁻⁶ mm`，最大差 `9.54×10⁻⁶ mm`。FSL 完整命令耗时 49.73 秒，FNIT GPU 单次 Python 调用 10.02 秒，计时边界不同，不据此计算加速比。TBSS 场重采样 FA 与 pipeline 标准 FA 的非零并集 `r=0.994361`、MAE `0.001231`；标准 FA 在重采样后另乘模板非零掩膜，因此这里还包含掩膜末步差异。

MMORF warp 加同目录 FLIRT 矩阵转换后重采样 FA，与 pipeline 原 `apply_mmorf_warp` 标准 FA 的 `r=0.999999999996`、MAE `2.14×10⁻⁷`。转换 Python 调用耗时 7.27 秒。FSL `convertwarp` 不能直接读取 MMORF 原场，转换后的 dense 场可供 FSL `invwarp` 和 `applywarp` 使用。[验证记录](../../validation/convertwarp/README.md)列出命令与输入边界；反变换结果及脑图见 [TorchInvWarp](../invwarp/README.md)。
