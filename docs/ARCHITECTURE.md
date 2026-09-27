# 代码结构

[返回首页](../README.md) · [新增功能](ADDING_FUNCTIONS.md)

每个功能位于 `src/fnit/` 的独立目录，完整用法、原版对应关系和最新验证位于 `docs/` 的同名子页面。共享的 `cli.py` 提供单被试命令行入口，`weights.py` 定位并校验官方权重。公开命令行每次处理一个病例；`recon_all` 另提供仅限 Python 的多被试并行调用。

FastVBM 的配准链位于 `flirt/`、`fnirt/`、`applywarp/`、`fast_vbm/registration.py` 和 `fast_vbm/synthmorph_backend.py`。`registration.py` 先执行共同 TorchFLIRT，只在非线性形变估计处分到 SynthMorph 或 TorchFNIRT，随后回到共同 FSL warp conversion、TorchApplyWarp、Jacobian 和 modulation。

| 功能 | 详细说明 |
|---|---|
| SynthStrip | [脑提取](synthstrip/README.md) |
| SynthMorph | [配准](synthmorph/README.md) |
| WMH-SynthSeg | [结构及 WMH 分割](wmh_synthseg/README.md) |
| 33 类 SynthSeg | [T1 结构分割](synthseg/README.md) |
| SynthSR | [合成 T1w](synthsr/README.md) |
| TorchFAST | [三组织分割及偏置校正](fast/README.md) |
| TorchFLIRT | [FSL 12-DOF affine / 6-DOF normmi rigid](flirt/README.md) |
| TorchFNIRT | [FSL GM nonlinear registration](fnirt/README.md) |
| TorchApplyWarp | [FSL warp application](applywarp/README.md) |
| FastVBM | [原始 T1w 到 modulated GM](fast_vbm/README.md) |
| TorchTOPUP | [UKB AP/PA b0 畸变校正](topup/README.md) |
| TorchEDDY | [UKB dMRI 运动、EC 和离群切片校正](eddy/README.md) |
| TorchDTIFIT | [FSL 默认 OLS tensor fit](dtifit/README.md) |
| TorchAMICONODDI | [不依赖 DIPY 的 NODDI fitting；当前数值验证状态见文档](amico_noddi/README.md) |
| run_mmorf / TorchMMORF | [scalar/tensor shared-warp registration](mmorf/README.md) |
| dMRI pipeline | [optional TOPUP 到九张标准空间参数图](dmri_pipeline/README.md) |
| TorchBEDPOSTX | [体素内纤维方向估计](bedpostx/README.md) |
| TorchProbtrackX | [概率纤维束追踪与连接矩阵](probtrackx/README.md) |
| UKBConnectome | [已校正 DWI 与 T1w 到四张结构连接矩阵](connectome/README.md) |
| Python recon-all（近似版） | [单 T1 皮层重建](recon_all/README.md) |

## 公开 Python API

```python
from fnit import (
    SynthStrip, SynthMorph, WMHSynthSeg, SynthSeg, SynthSR,
    TorchFAST, FastVBM, FastVBMResult, VBMRegistrationResult,
    TorchFLIRT, FLIRTResult, TorchFNIRT, TorchFNIRTResult,
    TorchApplyWarp, ApplyWarpResult,
    TorchTOPUP, TOPUPResult, TOPUPConfig,
    UKBConnectome, ConnectomeResult,
    prepare_ukb_topup, run_ukb_topup,
    TorchEDDY, EDDYResult, EDDYConfig,
    TorchDTIFIT, DTIFITResult,
    TorchAMICONODDI, AMICONODDIResult, AMICONODDIConfig,
    run_mmorf, TorchMMORF, MMORFResult, MMORFConfig, apply_mmorf_warp,
    DMRIPipeline, DMRIPipelineResult, TorchTBSS, TBSSResult,
    prepare_ukb_eddy, run_ukb_eddy,
    TorchBEDPOSTX, BedpostXResult,
    TorchProbtrackX, ProbTrackXResult,
    flirt_to_world_affine, flirt_to_world_pull,
    voxel_to_fsl_scaled_mm, world_to_flirt_affine,
    apply_transform,
)
```

学习模型在构造时加载权重并选择 `device="cpu"` 或 `device="cuda:0"`；TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp、TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI、TorchMMORF、TorchBEDPOSTX 和 TorchProbtrackX 不加载预训练权重。dMRI pipeline 的 MMORF 分支调用 SynthStrip，因此该分支需要 SynthStrip 权重。UKBConnectome 默认使用 CUDA，从已校正 DWI 起步；其 SynthSeg 分割可由调用方提供，或在单次调用时加载官方权重；也可读取官方 FreeSurfer `aparc+aseg.mgz`。单次调用返回带几何信息的结果对象，由调用者选择保存字段。FastVBM 组合 SynthStrip、TorchFAST、TorchFLIRT 和一个可选非线性后端。`registration_backend="synthmorph"` 延迟加载官方 deform checkpoint；`registration_backend="fnirt"` 构造无 checkpoint 的 TorchFNIRT。

权重查找顺序为显式路径、`FNIT_WEIGHTS`、配置脚本保存的目录、用户缓存目录、已设置的 `FREESURFER_HOME/models/`；见[权重说明](WEIGHTS.md)。各功能的单被试 Python 返回值、保存方式和命令行参数见上表链接。
