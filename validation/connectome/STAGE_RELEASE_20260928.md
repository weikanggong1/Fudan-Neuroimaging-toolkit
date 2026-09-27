# dMRI connectome 阶段性记录（2026-09-28）

本次提交保留已实现的算子、真实数据同输入对照、运行时间和公开示例图，供后续复核。**阶段性发布不表示原 UKB-connectomics 的七套 atlas、1,000 万次播种和 28 张矩阵已与官方结果一致。**原版命令只在独立 benchmark 环境运行；FNIT 接口仍从已校正 DWI、旋转后的梯度及外部官方 FreeSurfer `recon-all` 产物起步。

## 已确认的结果

| 固定输入阶段 | 数值结果 | 时间、显存与证据 |
|---|---|---|
| 真实 UKB 与公开 ds004666 的 4D DWI → mean b0 → `bet -R` | 两例的 778,752 个 mean b0 值逐值一致，最终二值掩膜各 XOR 0；CPU/GPU 均核验 | H100 两例 PyTorch 核心分别 7.327/7.845 s，峰值分配 0.354 GiB；[BET 专页](../../docs/connectome/BET_B0_OPERATORS.md)、[公开图](../../docs/connectome/figures/ds004666_bet_mask_comparison.png) |
| 默认响应掩膜、Dhollander | UKB DWI 掩膜 XOR 0；11 个组织选择掩膜 XOR 0，WM/GM/CSF 响应最大差 6.28e−11 / 2.55e−10 / 8.73e−11 | [掩膜](default_dwi_mask_stage_20260927.md)、[响应](original_ukb_dhollander_stage_20260927.md) |
| 全脑 MSMT-CSD、mtnormalise | UKB 212,831 个体素 WM 9,577,395 系数最大差 2.98e−8；归一化三组织全图最大差均小于 2e−7 | FOD H100 核心 269.457 s/1.354 GiB，MRtrix CPU 完整命令 107.04 s；[FOD](original_ukb_fod_stage_20260927.md)、[归一化](original_ukb_mtnormalise_stage_20260927.md) |
| Tian S1 逆场、皮层 ribbon 标注 | 对固定官方 FNIRT 系数，Tian S1 标签差 0/6,269,400；皮层 annot→ribbon 标签差 0/16,777,216 | 逆场 H100 27.78 s/0.367 GiB，FSL CPU 41.43 s；[atlas 报告](../../docs/connectome/ORIGINAL_ATLAS_OPERATORS.md) |
| 固定官方流线的 SIFT2、FA、矩阵赋值 | SIFT2 权重 r=0.999999903、MAE 3.44e−5；FA 逐轨 r=0.9999999949、MAE 5.58e−7；count 矩阵 400/400 元素一致 | [SIFT2](ds004666/sift2_fmls_stage_20260927.md)、[FA](ds004666/tcksample_precise_stage.md)、[四矩阵](ds004666_fsl_act_real_tracks_assignment_report.json) |
| 公开 ds004666 的 10,000 次播种整链（固定旧版掩膜/变换） | seed 0 count/FBC 相关 0.98864/0.98527，连接支持 Dice 0.73585；mean length/FA 全边相关 0.57720/0.62107 | H100 266.83 s/2.720 GiB；[CSV、图和三种子重复范围](ds004666/README.md)；未包含本次新增的自动 BET 分支 |

合并远端更新前的 `0.12.1` 阶段快照 wheel 已在现有 Conda 环境中以 `pip --no-deps --target` 隔离安装；BET、FOD、FIRST/5TT、Tian 在真实输入的 CPU 最小调用均成功，详见[安装记录](ds004666/anatomy_conda_install.public.json)。本次 GPU 初始化发生显存不足，未重试；该安装检查不代替合并后源码的重新安装或全链数值验证。

表中的核心计算时间与独立命令墙钟的计时边界不同，不能直接当作整链加速比。本次 connectome 的私人 UKB 路径、受试者编号、影像和逐文件哈希留在授权服务器；connectome 报告只发布脱敏指标与公开 ds004666 示例。

## 未完成的官方一致性门槛

1. **T1/5TT：** 官方 `recon-all` 输入保留为外部产物。FIRST 网格 PVE 仍有 52 个体素差异（最大 0.001），5TT 有 92 个元素差异，GMWMI 有 228 个体素差异；需固定完整 FIRST 几何、修复并重跑。见 [FIRST/5TT](FIRST_MESH_5TT_STAGE_20260927.md)。
2. **配准及 atlas：** 同输入 6DOF/normmi FLIRT 的世界坐标位移均值仍为 0.186655 mm；8 mm 搜索候选已出现分叉，需定位搜索路径。Tian S2–S4、七套皮层加亚皮层 atlas 的标签、顺序及合并优先级尚未全量核对。见 [FLIRT](ORIGINAL_UKB_FLIRT_STAGE_20260927.md)与[atlas](../../docs/connectome/ORIGINAL_ATLAS_OPERATORS.md)。
3. **追踪：** 独立 iFOD2/ACT 的连接支持、FA 和速度仍与官方有明显差异。公开 10,000 次播种的 PyTorch 追踪 58.45 s、MRtrix 2.96 s，播种/接受口径见[追踪报告](ds004666/tracking_act_stage.md)；原流程的 1,000 万次播种没有跑过，需报告随机重复包络、时间和小于 20 GiB 的显存。
4. **最终输出：** 尚无相同 UKB 输入下七套 atlas、四指标共 28 张矩阵的完整闭环比较。需固定同一追踪产物，逐矩阵报告标签、维度、相关、误差、稳定性、运行时间和连接图；当前四张 20 区公开矩阵不能外推为原流程一致。
5. **文件几何和安装：** 自动 BET 数组已与官方一致，但当前 NIfTI 保存仍使用原 DWI affine/header；公开样本官方 mean b0 的 x/y zoom 相差 9.54e−7 mm，sform x 平移相差 −8.39233e−5 mm。应独立校验输出头与 affine。新版本还需从干净 Conda 环境安装、运行正式整链并核对所有最终产物。

[逐阶段验收表](ORIGINAL_UKB_PARITY_GATES.md)保留原软件命令、输入约束和下一轮复现范围。本阶段应作为开发中结果使用；后续只有在上述门槛全部通过后，才能声称输出与原官方 pipeline 一致。
