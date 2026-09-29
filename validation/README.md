# 验证索引

本页只索引当前发布代码采用的证据。机器可读报告记录候选源码或调用链的 SHA-256、输入边界、参考软件版本、计时范围和限制。真实数据报告保留实际测量源码 hash；若当前源码只改动报告未覆盖的分支，则另以机器可读证明记录旧、新 hash、适用配置和 AST 指纹。当前有两条这种继承链：[FLIRT 的 QC-only 第一段](runtime_dependencies/flirt_qc_source_equivalence.public.json)与[12-DOF/corratio 第二段](runtime_dependencies/flirt_profile_source_equivalence.public.json)，以及 [SynthMorph registration linear 路径](runtime_dependencies/synthmorph_linear_source_equivalence.public.json)和 [FNIRT/SynthMorph/dMRI 报告的公共包入口](runtime_dependencies/package_entry_source_equivalence.public.json)。这些证明均明确 `fresh=false`，不能写成当前 hash 的完整真实数据重跑，也不能外推到 FLIRT 6-DOF/normmi、SynthMorph nearest 或公共入口证明未列出的命令。

官方程序只用于生成参考结果。FNIT 候选运行不调用已安装的 FSL、FreeSurfer、SPM、MRtrix3、AFNI、DIPY 或工作流封装包。没有人工真值的比较衡量的是与参考实现的一致性，不代表生物学准确度。

## sMRI 与通用配准

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [SynthSeg+](../docs/synthseg_plus/README.md) | [公开 T1w 对照](synthseg_plus/README.md) | 同网格与官方 `--parc` 比较，98 个出现的前景标签最低 Dice 0.99834 | 单例 H100 命令时间和峰值显存；逐标签报告 |
| [TorchGEMS 脑干亚区](../docs/subregions/README.md) | [两张真实 T1 的 BrainstemSS 对照](subregions/README.md) | 两例四区均达逐区 Dice ≥0.95、体积差 ≤5% | 官方/FNIT 命令时间、逐区体积差、阶段耗时和切面图 |
| [SynthStrip](../docs/synthstrip/README.md) | [12 例 T1w GPU 对照](synthstrip/report.real.current.json) | shape、affine、dtype 一致；脑掩膜、脑图和距离场为近似一致 | 报告含逐例 FreeSurfer/FNIT 时间、RSS、峰值显存；功能页展示三视图 |
| [SynthMorph](../docs/synthmorph/README.md) | [12 例 GPU](synthmorph/report.real.current.gpu.json)、[12 例 CPU](synthmorph/report.real.current.cpu.json)与[公开 T1w 示例](synthmorph/public_example.current.json) | moved image 与 RAS-mm pull warp 接近参考；测量 hash 通过 linear-only 证明继承，nearest 另有定向测试 | 两份报告使用同一计时边界；没有 fresh current-hash 全量重跑；功能页展示公开配准图 |
| [WMH-SynthSeg](../docs/wmh_synthseg/README.md) | [3 例公开 FLAIR GPU 与 1 例 CPU](wmh/report.public.json) | 标签、WMH、软体积及 NIfTI 合同与 FreeSurfer 参考近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示 WMH overlay |
| [SynthSeg](../docs/synthseg/README.md) | [3 例公开 T1w](synthseg/report.public.json) | shape、affine、int32、qform/sform 合同一致；标签与软体积近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示标签与差异 |
| [SynthSR](../docs/synthsr/README.md) | [12 例 T1w 与公开 FLAIR 示例](synthsr/report.public.json) | shape、affine、uint8 合同一致；输出与 FreeSurfer CPU 参考近似一致 | 报告含 GPU/CPU 时间和峰值显存；功能页展示合成 T1w |
| [TorchFAST](../docs/fast/README.md) | [真实 brain-only T1w](fast/report.public.json) | 分割、PVE、bias field 与 bias-corrected 图按图比较；不是逐体素等价实现 | 报告含 FSL/FNIT 时间和峰值显存；功能页展示 GM、差值和偏置校正 |
| [FastVBM](../docs/fast_vbm/README.md) | [单例真实 GM 掩膜定位](fast_vbm/README.md) | 关闭隐式零值掩膜后，固定 FSL GM 输入的三项相关性约 0.996；完整链仍有上游差异 | 本版单例完整链已复测；FSL 缺少同边界耗时，多例尚待复测 |
| [TorchFLIRT](../docs/flirt/README.md) | [12-DOF 真实 MRI 对照](flirt/report.public.json) | FSL scaled-mm 矩阵合同和 reference-grid 输出均独立比较；仅 12-DOF/corratio 可通过两段证明继承 | 报告区分 CPU/GPU 与共享节点计时边界；没有 fresh current-hash 全量重跑；功能页展示公开 T1w 配准图 |
| [TorchFNIRT](../docs/fnirt/README.md) | [当前真实 FA matched-input 对照](fnirt/report.real.current.json) | coefficient、warped FA、两类 Jacobian 和标准网格合同分别核对；连续值差异超过浮点误差 | 报告含三阶段 FSL 与 FNIT 配准时间、显存；功能页展示真实 FA 对照 |
| [TorchApplyWarp](../docs/applywarp/README.md) | [真实 FA 与 intent-2007 coefficient warp](applywarp/report.real.current.json) | shape、affine、dtype 一致；连续值误差按 union support 报告 | 报告含三次 FSL/FNIT 计时和峰值显存；功能页展示 FA 与差值 |

## dMRI

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [TorchTOPUP](../docs/topup/README.md) | [真实 UKB 格式 AP/PA dMRI](topup/report.public.json) | Hz 场、校正图、运动参数与 FSL 文件合同逐项比较 | 报告含三次 FSL CPU/FNIT GPU 墙钟和显存；功能页展示场图与校正图 |
| [TorchEDDY](../docs/eddy/README.md) | [真实 UKB 格式 dMRI](eddy/report.public.json) | 对 `eddy_cuda10.2` 的 4D r=0.999738，14/15 离群切片重合；不是逐体素等价实现 | FSL GPU 10:21.19（进程），FNIT GPU 10:38.75（CUDA 初始化后调用）；切片图尚未公开 |
| [TorchDTIFIT](../docs/dtifit/README.md) | [真实 UKB 格式 dMRI](dtifit/report.public.json) | tensor、FA、MD、特征值、特征向量和 S0 的合同与数值分别报告 | 报告含三次 FSL/FNIT 墙钟；功能页展示 FA 与差值 |
| [TorchAMICONODDI](../docs/amico_noddi/README.md) | [真实 EDDY 校正 dMRI](amico_noddi/report.public.json)与[无 DIPY 核心证明](amico_noddi/no_dipy_equivalence.public.json) | NDI、ODI、FWF、方向与 RMSE 对 AMICO 2.0.3 逐体素比较 | 报告含参考/FNIT 分阶段时间和峰值显存；功能页展示参数图与差值 |
| [TorchMMORF](../docs/mmorf/README.md) | [正常路径 FSL 对照](mmorf/report.public.json)与[当前源码恢复试验](mmorf/recovery.real.current.json) | pull warp、Jacobian、warped scalar 和九张参数图合同通过；正常路径测量早于恢复逻辑，当前源码尚无 fresh 同输入 FSL 重测 | 分别记录 FSL/FNIT 正常路径时间和当前源码真实数据恢复、整链时间；功能页保留原对照图 |
| [dMRI 参数图 pipeline](../docs/dmri_pipeline/README.md) | [TBSS 分支](dmri_pipeline/tbss_e2e.real.current.json)与[MMORF 分支](dmri_pipeline/mmorf_e2e.real.current.json) | 两条路径从 raw AP/PA 生成九张标准空间图；网格/dtype 合同通过，连续值未达到逐体素等价 | 报告分开记录各阶段时间、显存和比较边界；新版病例图仍保存在计算节点 |
| [TorchBEDPOSTX](../docs/bedpostx/README.md) | [真实 dMRI ROI 与 crossing-fibre 检查](bedpostx/report.public.json) | 比较纤维数、fraction、方向轴和角度；采样随机流不同，不要求后验体积逐元素相同 | 报告含 FSL CPU、FNIT CPU/GPU 时间；功能页展示真实方向轴 |
| [TorchProbtrackX](../docs/probtrackx/README.md) | [默认追踪](probtrackx/report.default.latest.public.json)、[当前 GPU](probtrackx/report.current.latest.public.json)与[matrix/target 汇总](probtrackx/README.md) | 比较密度、路径长度、稀疏 voxel 矩阵和 ROI 连接矩阵；随机流不同，不要求逐轨迹相同 | 报告含 FSL/FNIT CPU/GPU 计时；功能页展示真实五区连接矩阵 |

## fMRI

| 功能 | 当前真实数据报告 | 输出一致性结论 | 时间、显存与示意图 |
|---|---|---|---|
| [BIDS→MNI152 2 mm fMRI](../docs/fmri/README.md) | [FEAT](fmri/feat_summary.json)、[BBR](fmri/bbr_summary.json)、[PICA](fmri/pica_summary.json)、[配准](fmri/registration_summary.json)与[整链](fmri/e2e_summary.json) | FEAT 子阶段分别比较；最终 AROMA 输出没有可逐体素配对的 UKB FIX 参考 | 报告记录阶段时间、显存和限制；功能页的真实数据图同时展示 FEAT mean BOLD、BBR、T1→MNI 配准和 PICA |
| [MS-HBM 17 网络](../docs/mshbm/README.md) | [真实 fsLR32k 静息态时序](mshbm/report.public.json) | 输入 profile、网络标签和 Dice/ARI 按顶点比较 | 报告含 CBIG/FNIT 的匹配计时与内存；功能页展示网络标签与差异 |

recon-all 与 Connectome 的验证边界由各自功能页维护，不纳入本页审计表。

公开报告不含账号、私有绝对路径、源病例编号、权重或临床原图。可再分发样例及来源校验见 [T1w 示例](../examples/README.md)与 [FLAIR 示例](../examples/WMH.md)。独立环境求解与固定 NumPy wheel 的证据见[环境验证](environment/README.md)。
