# TorchInvWarp

在 `reference` 指定的输出网格上求 FSL dense 或 FNIRT 三次样条 pull 场的反场。常用场景是把 MNI 网格上的 diffusion→MNI 组合场反转到 diffusion 网格，再用 TorchApplyWarp 重采样 MNI 掩膜。PyTorch 固定点求逆的结果与 FSL 的拓扑约束算法存在数值差异。

```python
from fnit import TorchInvWarp

result = TorchInvWarp(device="cuda:0").run(
    reference="/data/nodif_brain_mask.nii.gz",  # diffusion 输出网格
    warp="/data/diff_to_MNI_warp.nii.gz",  # MNI 网格上的前向 pull 场
    output="/data/MNI_to_diff_warp.nii.gz",  # diffusion 网格上的反向三分量位移场
)
```

`result.image`、`result.valid_fraction`、`result.qc` 分别为反场、原场网格有效比例和收敛信息。命令行 `fnit invwarp --ref ... --warp ... --out ... --rel --device cuda:0` 对应 FSL `invwarp --ref=... --warp=... --out=... --rel`。完整参数、真实 DWI 对照和示意图见[功能页](../../../docs/invwarp/README.md)。
