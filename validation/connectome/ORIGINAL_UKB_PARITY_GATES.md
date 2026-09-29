# 原 UKB-connectomics 流程的验收边界

目标是用同一名 UKB 受试者、同一输入和固定随机种子，对原脚本的每个中间产物及最终矩阵进行配对验证。原仓库使用 FreeSurfer 7.1.1、FSL 6.0.3 和 MRtrix `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`；公开 ds004666 的现有适配参考使用 FreeSurfer 8.2、另一版 MRtrix、20 区 SynthSeg atlas 和 1 万次播种，不能代表原 UKB 流程已经通过。

| 验收阶段 | 原流程固定输入、输出与所需比较 | 当前状态 |
|---|---|---|
| DWI 与响应 | UKB `data_ud`、`bvecs/bvals`；`dwi2response dhollander` 未传 `-mask`，MRtrix 内部调用 `dwi2mask`。比较掩膜逐体素及 WM/GM/CSF 响应逐元素；记录时间、RAM/VRAM。 | 同一真实 UKB DWI 的默认 `dwi2mask legacy` 掩膜 XOR 0；Dhollander 11 个选择掩膜 XOR 0，WM/GM/CSF 响应最大误差分别为 6.28e−11 / 2.55e−10 / 8.73e−11，见[掩膜](default_dwi_mask_stage_20260927.md)和[完整响应](original_ukb_dhollander_stage_20260927.md)。全脑 212,831 个掩膜体素的 WM/GM/CSF FOD 最大绝对误差分别为 2.98e−8 / 2.84e−14 / 9.31e−10；H100 求解 269.457 s，MRtrix CPU 完整命令 107.04 s，见[全脑 FOD 报告](original_ukb_fod_stage_20260927.md)。BET 掩膜的两次膨胀／侵蚀均 XOR 0；归一化三组织全图最大误差均 <2e−7，见[归一化报告](original_ukb_mtnormalise_stage_20260927.md)。 |
| T1 与 5TT | 同名 T1 的官方 `recon-all`、FIRST；`5ttgen freesurfer -first T1_first -nocrop -sgm_amyg_hipp`、`5tt2gmwmi`。比较 5TT 每组织体素、GMWMI、affine。 | 同一真实 UKB T1 的官方 FIRST 14 个 VTK 已固定；当前 PyTorch 网格 PVE 有 52 个体素差异（最大 0.001），5TT 有 92 个元素差异，GMWMI 有 228 个体素差异，见[FIRST/5TT 报告](FIRST_MESH_5TT_STAGE_20260927.md)。尚未逐值一致。 |
| 配准与 atlas | `flirt -cost normmi -dof 6`；皮层 annot 经 pial/white 最近邻贴至 ribbon；Tian S1–S4 经 FSL `invwarp`/`applywarp --interp=nn` 映射。逐 atlas 比较 label XOR、Dice、affine、LUT 顺序及合并优先级。 | 同一 UKB b0/T1 的当前 6DOF 求解与 FSL 世界坐标位移 RMS 为 0.009932 mm，尚未逐值一致，见[配准报告](ORIGINAL_UKB_FLIRT_STAGE_20260929.md)。固定 FSL 变换后皮层 annot→ribbon 标签 XOR 0；同一新生成的 FNIRT coefficient 下，FNIT 逆场相对 FSL 全域最大误差 1.144e−5 mm，Tian S1 标签 XOR 0；S2–S4 尚未验证，见[atlas 算子报告](../../docs/connectome/ORIGINAL_ATLAS_OPERATORS.md)。 |
| 追踪与 SIFT2 | 原版 iFOD2、GMWMI、ACT、`-seeds 10M -select 0 -samples 3 -power 0.5`；以固定输入、种子和官方重复包络比较长度/密度、连接支持和误差。固定官方 TCK 时逐轨验证 SIFT2、FA 与端点赋值。 | 公开样本 1 万次种子的固定官方 TCK 下 SIFT2/FA/赋值已有高精度对照；独立追踪的随机采样仍在改进。10M 显存和运行时间尚未验证。 |
| 最终矩阵 | 一次追踪及 SIFT2 共用于七套皮层+Tian atlas；每套至少核对 count、SIFT2 FBC、mean length、mean FA，共 28 张核心矩阵。逐张记录维度、行列标签、相关性、相对 L1、共同边误差、时间及示例图。 | 公开适配样本仅有单个 20 区 atlas 的四矩阵三种子对照；没有原版七套 atlas/10M 的闭环结果。 |

已在授权服务器找到同一 UKB 受试者的 DWI 与 T1 包，并核对 DWI、b 值和 FreeSurfer 分割/皮层表面输入。逐文件哈希、受试者编号、私有路径和所有体素影像只保留在服务器验证目录，不进入公开仓库。FIRST 与 `T1_to_MNI_warp_coef` 不在提供的存档中；当前用官方 FSL 从同一 T1 生成并固定参考，新生成的形变不会标作 UKB 存档产物。可公开的脑部示例图使用 OpenNeuro ds004666。

原始命令以 [UKB-connectomics 总流程](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/UKB_connectivity_mapping_pipeline.sh)、[追踪脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/probabilistic_tractography_native_space.sh) 以及仓库 README 的版本为准。每个新算子应在真实数据上固定中间输入与官方比较，并归档命令、输入/代码 SHA-256、数值指标、墙钟时间、峰值显存和示例图；随机追踪应同时报告官方自身固定种子重复范围。本阶段按用户要求以未完成验证状态推送 `main`，记录见[阶段性发布](STAGE_RELEASE_20260928.md)；只有这些缺口全部通过后，才能将“输出与原官方 pipeline 一致”作为结论。
