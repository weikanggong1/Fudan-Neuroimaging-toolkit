# 代码结构

[返回首页](../README.md) · [新增功能](ADDING_FUNCTIONS.md)

每个功能位于 `src/fnit/` 的独立目录，完整用法、原版对应关系和最新验证位于 `docs/` 的同名子页面。共享的 `cli.py` 提供单被试命令行入口，`weights.py` 定位并校验官方权重。除本轮未调整的 `recon_all` 子包外，公开接口只接受单被试输入：Python 每次调用一个对象，命令行每次处理一个病例。

FastVBM 的配准链位于 `flirt/`、`fnirt/`、`applywarp/`、`fast_vbm/registration.py` 和 `fast_vbm/synthmorph_backend.py`。`registration.py` 先执行共同 TorchFLIRT，只在非线性形变估计处分到 SynthMorph 或 TorchFNIRT，随后回到共同 FSL warp conversion、TorchApplyWarp、Jacobian 和 modulation。

| 功能 | 详细说明 |
|---|---|
| SynthStrip | [脑提取](synthstrip/README.md) |
| SynthMorph | [配准](synthmorph/README.md) |
| WMH-SynthSeg | [结构及 WMH 分割](wmh_synthseg/README.md) |
| 33 类 SynthSeg | [T1 结构分割](synthseg/README.md) |
| SynthSR | [合成 T1w](synthsr/README.md) |
| TorchFAST | [三组织分割及偏置校正](fast/README.md) |
| TorchFLIRT | [FSL 12-DOF affine](flirt/README.md) |
| TorchFNIRT | [FSL GM nonlinear registration](fnirt/README.md) |
| TorchApplyWarp | [FSL warp application](applywarp/README.md) |
| FastVBM | [原始 T1w 到 modulated GM](fast_vbm/README.md) |
| TorchTOPUP | [UKB AP/PA b0 畸变校正](topup/README.md) |
| GPU recon-all | [单 T1 皮层重建](recon_all/README.md) |

## 公开 Python API

```python
from fnit import (
    SynthStrip, SynthMorph, WMHSynthSeg, SynthSeg, SynthSR,
    TorchFAST, FastVBM, FastVBMResult, VBMRegistrationResult,
    TorchFLIRT, FLIRTResult, TorchFNIRT, TorchFNIRTResult,
    TorchApplyWarp, ApplyWarpResult,
    TorchTOPUP, TOPUPResult, TOPUPConfig,
    prepare_ukb_topup, run_ukb_topup,
    flirt_to_world_affine, flirt_to_world_pull,
    voxel_to_fsl_scaled_mm, world_to_flirt_affine,
    apply_transform,
)
```

学习模型在构造时加载权重并选择 `device="cpu"` 或 `device="cuda:0"`；TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp 和 TorchTOPUP 不加载权重。单次调用返回带几何信息的结果对象，由调用者选择保存字段。FastVBM 组合 SynthStrip、TorchFAST、TorchFLIRT 和一个可选非线性后端。`registration_backend="synthmorph"` 延迟加载官方 deform checkpoint；`registration_backend="fnirt"` 构造无 checkpoint 的 TorchFNIRT。

权重查找顺序为显式路径、`FNIT_WEIGHTS`、配置脚本保存的目录、用户缓存目录、已设置的 `FREESURFER_HOME/models/`；见[权重说明](WEIGHTS.md)。各功能的单被试 Python 返回值、保存方式和命令行参数见上表链接。
