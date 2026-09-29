# 验证索引

本页只索引当前发布代码采用的证据。机器可读报告记录候选源码或调用链的 SHA-256、输入边界、参考软件版本、计时范围和限制。真实数据报告保留实际测量源码 hash。FLIRT 的 [12-DOF 测量源码范围核对](runtime_dependencies/flirt_profile_source_equivalence.public.json)记录本次修订前后的精确差异和一例真实病例矩阵逐字节复核；6-DOF 则由最终源码直接重跑。其他保留的继承链有 [SynthMorph registration linear 路径](runtime_dependencies/synthmorph_linear_source_equivalence.public.json)和 [FNIRT/SynthMorph/dMRI 报告的公共包入口](runtime_dependencies/package_entry_source_equivalence.public.json)。这些记录不能写成当前 hash 的完整多例重跑，也不能外推到未列出的配置。

官方程序只用于生成参考结果。FNIT 候选运行不调用已安装的 FSL、FreeSurfer、SPM、MRtrix3、AFNI、DIPY 或工作流封装包。没有人工真值的比较衡量的是与参考实现的一致性，不代表生物学准确度。

## sMRI 与通用配准

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [SynthSeg+](../docs/synthseg_plus/README.md) | [公开 T1w 对照](synthseg_plus/README.md) | GPU 合并图逐体素一致率 0.9999758；CPU 仅 6 个体素不同，软体积最大差 0.293 mm³ | 官方与 FNIT 的 CPU/GPU 完整命令计时、GPU Python 调用计时、逐标签报告及切面图 |
| [TorchGEMS 脑干亚区](../docs/subregions/README.md) | [两张真实 T1 的 BrainstemSS 对照](subregions/README.md) | 两例四区均达逐区 Dice ≥0.95、体积差 ≤5% | 官方/FNIT 命令时间、逐区体积差、阶段耗时和切面图 |
| [GEMS 丘脑、海马与杏仁核](../docs/subregions/nuclei.md) | [同输入真实 T1 对照](subregions/nuclei.md) | 新 Conda 环境前景 Dice 0.9819–0.9957；逐区阈值未通过 | 官方/FNIT 阶段时间、逐标签 JSON 和切面图 |
| [SynthStrip](../docs/synthstrip/README.md) | [12 例 T1w GPU 对照](synthstrip/report.real.current.json) | shape、affine、dtype 一致；脑掩膜、脑图和距离场为近似一致 | 报告含逐例 FreeSurfer/FNIT 时间、RSS、峰值显存；功能页展示三视图 |
| [SynthMorph](../docs/synthmorph/README.md) | [12 例 GPU](synthmorph/report.real.current.gpu.json)、[12 例 CPU](synthmorph/report.real.current.cpu.json)与[公开 T1w 示例](synthmorph/public_example.current.json) | moved image 与 RAS-mm pull warp 接近参考；测量 hash 通过 linear-only 证明继承，nearest 另有定向测试 | 两份报告使用同一计时边界；没有 fresh current-hash 全量重跑；功能页展示公开配准图 |
| [WMH-SynthSeg](../docs/wmh_synthseg/README.md) | [3 例公开 FLAIR GPU 与 1 例 CPU](wmh/report.public.json) | 标签、WMH、软体积及 NIfTI 合同与 FreeSurfer 参考近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示 WMH overlay |
| [SynthSeg](../docs/synthseg/README.md) | [3 例公开 T1w](synthseg/report.public.json) | shape、affine、int32、qform/sform 合同一致；标签与软体积近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示标签与差异 |
| [SynthSR](../docs/synthsr/README.md) | [12 例 T1w 与公开 FLAIR 示例](synthsr/report.public.json) | shape、affine、uint8 合同一致；输出与 FreeSurfer CPU 参考近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示合成 T1w |
| [TorchFAST](../docs/fast/README.md) | [真实 brain-only T1w](fast/report.public.json) | 分割、PVE、bias field 与 bias-corrected 图按图比较；不是逐体素等价实现 | 报告含 FSL/FNIT 时间和峰值显存；功能页展示 GM、差值和偏置校正 |
| [FastVBM](../docs/fast_vbm/README.md) | [单例真实 GM 掩膜定位](fast_vbm/README.md) | 关闭隐式零值掩膜后，固定 FSL GM 输入的三项相关性约 0.996；完整链仍有上游差异 | 本版单例完整链已复测；FSL 缺少同边界耗时，多例尚待复测 |
| [TorchFLIRT](../docs/flirt/README.md) | [12-DOF GM CPU 10 例](flirt/report.cpu.current.json)、[H100 4 例](flirt/report.public.json)与[6-DOF b0→T1 一例](connectome/original_ukb_flirt.public.json) | 6-DOF 相对 FSL 矩阵位移 RMS 为 CPU 0.00993 mm、H100 0.00998 mm；12-DOF CPU 9/10 例通过 0.05 mm 门限 | 共享节点时间只报观察值；公开 T1w 图和完整边界见功能页 |
| [TorchFNIRT](../docs/fnirt/README.md) | [当前真实 FA matched-input 对照](fnirt/report.real.current.json) | coefficient、warped FA、两类 Jacobian 和标准网格合同分别核对；连续值差异超过浮点误差 | 报告含三阶段 FSL 与 FNIT 配准时间、显存；功能页展示真实 FA 对照 |
| [TorchApplyWarp](../docs/applywarp/README.md) | [真实 FA 与 intent-2007 coefficient warp](applywarp/report.real.current.json) | shape、affine、dtype 一致；连续值误差按 union support 报告 | 报告含三次 FSL/FNIT 计时和峰值显存；功能页展示 FA 与差值 |
| [TorchConvertWarp](../docs/convertwarp/README.md) | [真实 DWI 组合场](convertwarp/README.md)及[TBSS/MMORF pipeline 分支](probtrackx/README.md) | 默认 TBSS 系数场与 FSL 分量 MAE 1.98×10⁻⁶ mm；MMORF 转场后重采样 FA 与原 pipeline r≈1 | FSL TBSS 完整命令 49.73 s，FNIT Python 调用 10.02 s；MMORF FNIT 调用 7.27 s，计时边界不同 |
| [TorchInvWarp](../docs/invwarp/README.md) | [真实 TBSS/MMORF 同输入 FSL 对照](invwarp/README.md) | 脑内反场向量均差 0.0058/0.0670 mm，MNI 掩膜 Dice 0.9917/0.9912；不是逐体素等价 | FSL 完整命令 110.82/137.49 s，FNIT Python 调用 0.79/0.81 s；功能页展示两条真实掩膜图 |
| [PyTorch fslmaths 常用运算](../docs/fslmaths/README.md) | [真实 T1w、tensor 与 BOLD 配对对照](fslmaths/real_data_20260929.json) | 48 组裁剪块、5 组完整影像、2 组 BOLD 滤波和 3 组 3D/4D 混合运算逐体素比较 | 报告记录 FSL CPU、FNIT 拥挤 H100 的耗时，主要运算另记录 CUDA 峰值分配 |

## dMRI

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [TorchTOPUP](../docs/topup/README.md) | [真实 UKB 格式 AP/PA dMRI](topup/report.public.json) | Hz 场、校正图、运动参数与 FSL 文件合同逐项比较 | 报告含三次 FSL CPU/FNIT GPU 墙钟和显存；功能页展示场图与校正图 |
| [TorchEDDY](../docs/eddy/README.md) | [真实 UKB 格式 dMRI](eddy/report.public.json) | 对 `eddy_cuda10.2` 的 4D r=0.999738，14/15 离群切片重合；不是逐体素等价实现 | FSL GPU 10:21.19（进程），FNIT GPU 10:38.75（CUDA 初始化后调用）；切片图尚未公开 |
| [TorchDTIFIT](../docs/dtifit/README.md) | [真实 UKB 格式 dMRI](dtifit/report.public.json) | tensor、FA、MD、特征值、特征向量和 S0 的合同与数值分别报告 | 报告含三次 FSL/FNIT 墙钟；功能页展示 FA 与差值 |
| [TorchAMICONODDI](../docs/amico_noddi/README.md) | [真实 EDDY 校正 dMRI](amico_noddi/report.public.json)与[无 DIPY 核心证明](amico_noddi/no_dipy_equivalence.public.json) | NDI、ODI、FWF、方向与 RMSE 对 AMICO 2.0.3 逐体素比较 | 报告含参考/FNIT 分阶段时间和峰值显存；功能页展示参数图与差值 |
| [TorchMMORF](../docs/mmorf/README.md) | [当前双标量真实数据对照](mmorf/report.public.json) | T1、FA 两组标量与 DTI tensor 共享 warp；记录自动 PyTorchFLIRT、warp、Jacobian、两张 warped scalar 的配对精度 | 分别记录线性、非线性、官方 MMORF 时间及共享 GPU 负载；功能页提供当前对照图 |
| [dMRI 参数图 pipeline](../docs/dmri_pipeline/README.md) | [TBSS 原软件对照](dmri_pipeline/tbss_e2e.real.current.json)、[真实 BIDS 输入预检](dmri_pipeline/bids_preflight.real.final.json)、[无 T1w 整链](dmri_pipeline/bids_tbss.real.current.json)、[带 T1w 失败阶段](dmri_pipeline/bids_mmorf.shared_gpu_oom.json) | BIDS 无 T1w 路径生成九张标准图及 skeleton；带 T1w 路径已生成九张 native 图，但 MMORF 因共享 GPU 显存耗尽，标准图待验收 | BIDS 报告记录完整命令时间和阶段时间；FSL 数值对照仍以原 TBSS 报告为准 |
| [TorchBEDPOSTX](../docs/bedpostx/README.md) | [真实 dMRI ROI 与 crossing-fibre 检查](bedpostx/report.public.json) | 比较纤维数、fraction、方向轴和角度；采样随机流不同，不要求后验体积逐元素相同 | 报告含 FSL CPU、FNIT CPU/GPU 时间；功能页展示真实方向轴 |
| [TorchProbtrackX](../docs/probtrackx/README.md) | [默认追踪](probtrackx/report.default.latest.public.json)、[当前 GPU](probtrackx/report.current.latest.public.json)、[matrix/target 汇总](probtrackx/README.md)与[MNI seed 两分支自动转换](probtrackx/README.md) | 比较密度、路径长度、稀疏矩阵和 ROI 矩阵；真实 MNI seed 自动转换与直接传入转换结果的追踪图完全相同 | 原追踪报告含 FSL/FNIT CPU/GPU 计时；功能页展示连接矩阵和真实掩膜 |

## fMRI

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [BIDS→MNI152 2 mm fMRI](../docs/fmri/README.md) | [FEAT](fmri/feat_summary.json)、[BBR](fmri/bbr_summary.json)、[PICA](fmri/pica_summary.json)、[T1 FNIRT 配准](fmri/t1_fnirt_20260929.public.json)、[FNIRT volume 整链](fmri/fmri_volume_fnirt_20260929.public.json)与[默认整链](fmri/e2e_summary.json) | FEAT 子阶段分别比较；最终 AROMA 输出没有可逐体素配对的 UKB FIX 参考 | 报告记录阶段时间、显存和限制；功能页的真实数据图同时展示 FEAT mean BOLD、BBR、T1→MNI 配准和 PICA |
| [MS-HBM 17 网络](../docs/mshbm/README.md) | [真实 fsLR32k 静息态时序](mshbm/report.public.json) | 输入 profile、网络标签和 Dice/ARI 按顶点比较 | 报告含 CBIG/FNIT 的匹配计时与内存；功能页展示网络标签与差异 |

recon-all 与 Connectome 的验证边界由各自功能页维护，不纳入本页审计表。

公开报告不含账号、私有绝对路径、源病例编号、权重或临床原图。可再分发样例及来源校验见 [T1w 示例](../examples/README.md)与 [FLAIR 示例](../examples/WMH.md)。独立环境求解与固定 NumPy wheel 的证据见[环境验证](environment/README.md)。
