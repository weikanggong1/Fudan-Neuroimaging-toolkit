# TorchConvertWarp

组合 FSL scaled-mm `premat → warp1 → postmat`，在 `reference` 网格保存三分量 pull 场。常用输入是 diffusion→T1 FLIRT `.mat` 与 T1→MNI FNIRT 系数；默认输出相对位移，单位 mm。运行时只依赖本包的 TorchApplyWarp 几何函数、Nibabel、NumPy、PyTorch。

```python
from fnit import TorchConvertWarp

result = TorchConvertWarp(device="cuda:0").run(
    reference="/data/MNI.nii.gz",  # 输出场的 MNI 网格
    warp1="/data/T1_to_MNI_warp.nii.gz",  # T1→MNI 非线性场
    premat="/data/diff_to_T1.mat",  # diffusion→T1 FLIRT 矩阵
    output="/data/diff_to_MNI_warp.nii.gz",  # 4D float32 NIfTI，末轴为 xyz
)
```

`result.image`、`result.valid_fraction`、`result.qc` 分别为输出图、warp1 网格有效比例和坐标约定记录。命令行 `fnit convertwarp --ref ... --warp1 ... --premat ... --out ... --relout --device cuda:0` 对应 FSL `convertwarp --ref=... --warp1=... --premat=... --out=... --relout`。完整参数、真实 DWI 对照和局限见[功能页](../../../docs/convertwarp/README.md)。

FNIT dMRI pipeline 的 TBSS `registration/dti_FA_to_MNI_warp.nii.gz` 已含线性部分，用 `warp1` 时不加 `premat`。MMORF 的 `registration/mmorf_warp.nii.gz` 是参考图像轴 mm 位移，不能直接交给 FSL；使用 `TorchConvertWarp(device).run_mmorf(reference=..., source=..., mmorf_warp=..., affine=..., output=...)` 将它和 `dti_FA_to_MNI_affine.mat` 合为 FSL dense 场。命令行对应 `fnit convertwarp --ref ... --source ... --mmorf-warp ... --premat ... --out ...`，该 MMORF 输入没有同名的原版 FSL `convertwarp` 命令。
