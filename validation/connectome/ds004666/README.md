# ds004666 配对 T1/DWI：当前 connectome 验证

[OpenNeuro ds004666](https://openneuro.org/datasets/ds004666) `sub-01/ses-2mm` 提供配对 T1w 与 AP/PA DWI。原始输入的 SHA-256 见 [download_manifest.tsv](download_manifest.tsv)。本例以 FSL TOPUP/EDDY 生成校正 DWI 和旋转 bvec；元数据缺少实测总读出时间，因而参考处理假定 0.05 s，详见[输入来源](corrected_input_provenance.public.json)。T1 分割取已完成的官方 FreeSurfer `recon-all` 目录；FNIT 正式运行不调用 FreeSurfer。

![配对 T1、校正前后 b0 与真实 atlas 切面](../../../docs/connectome/figures/ds004666_t1_raw_vs_topup_eddy_atlas.png)

## 与官方流程逐项对照

原 [UKB-connectomics](https://github.com/sina-mansour/UKB-connectomics) 以 UKB `data_ud`、FIRST、七套皮层+Tian 图谱及 1,000 万次播种为输入。本公开样本起初用 MRtrix3 3.0.3 的 FreeSurfer 5TT、20 节点 atlas 和每次 10,000 次播种隔离追踪误差；后续已扩展到 100k 播种和七套图谱矩阵。适配流程仍不等于原 UKB 的 FIRST/FNIRT/10M 整链。[早期参考命令和输出](corrected_mrtrix_fs5tt_act_adapted/)可核对。

| 步骤 | 当前同输入证据 |
|---|---|
| 5TT、GMWMI、DWI↔T1、atlas | [解剖报告](ANATOMY_STAGE_20260927.md)；固定官方分割的 5TT/GMWMI 逐值一致 |
| 掩膜、响应、FOD、mtnormalise | [掩膜](maskfilter_stage_20260927.md)、[响应/FOD](response_fod_stage_20260927.md)、[归一化](mtnormalise_stage_20260927.md)；固定输入的主要数值误差与脑图分别列明 |
| iFOD2/ACT | [当前拒绝采样与 ACT 报告](ifod2_rejection_20260929.md)；固定单弧最大概率误差 1.58×10⁻⁶，12,600 个真实 5TT 采样点 ACT 状态零差异；独立随机轨迹未全面进入官方重复包络 |
| GMWMI 播种位置 | [旧算法的 100k 空间分布和脑图](gmwmi_seed_100k_20260929.md)；[精确仿射与现行播种的复核](act_geometry_accepted_seeds_20260929.md)使 ACT 单向比例达到 0.926149，官方两次为 0.925899/0.925910 |
| 100k 播种规模 | [三次官方与三次 FNIT 的全链四矩阵、轨迹分布和脑图](tracking_100k_three_seed_20260929.md)固定相同体素数组；NIfTI-1 5TT 的世界仿射仍有微小舍入差。[精确仿射检查](act_geometry_accepted_seeds_20260929.md)另列。FNIT 编译核追踪 782.49–806.95 s、全链 Torch 峰值 2.468–2.473 GiB；count 相对 L1 有 6/9 个跨软件配对进入官方范围，长度/端点/TDI 均为 0/9 |
| 可选 CUDA 圆弧编译核 | [同输入 128/1k/100k 对照与图](tracking_compile_20260929.md)；100k 首次编译计入的追踪 782.49 s、保留 27,353 条、全链 Torch 峰值 2.468 GiB。不同时间的共享 GPU 负载不能用于稳定加速比；长度/端点/TDI 仍超出官方自身重复范围 |
| 七套 atlas 的 100k 四矩阵 | [固定与独立 TCK 对照](seven_atlas_100k_20260929.md)；84–1054 节点的七套 count 在同一官方 TCK/逐轨数值下全部逐元素一致；独立 FNIT 轨迹的五项矩阵指标仍未全面进入三次官方互比范围，Tian 标签使用 SynthMorph |
| SIFT2、FA、双端赋值 | 固定同一官方 TCK，[SIFT2](sift2_mapping_stage.md) 逐轨权重相关 0.999999903、[FA](tcksample_precise_stage.md) 逐轨相关 0.9999999949、[count](integrated_seed0_fixed_tck_assignment.public.json) 400/400 元素一致 |
| atlas 自动生成 | [Schaefer200+Tian S1/S4](atlas_synthmorph_20260929.md)、[Schaefer500/1000+Tian S4](atlas_schaefer_multi_20260929.md)、[原生 aparc/a2009s+Tian S1](atlas_native_aparc_20260929.md)和[Glasser+Tian S1/S4](atlas_glasser_20260929.md)；七套原 UKB 图谱的皮层体积与原脚本逐体素一致。Glasser 有两个仅 1–2 个 T1 体素的节点在 DWI 网格消失。SynthMorph 与原 FNIRT 路线存在明确差异，给定同一 FNIRT coefficient 可使 Tian S1 逐体素一致 |
| 配准影响 | [固定流线敏感性实验](registration_sensitivity_20260929.md)；只换 atlas 时 count 相关 0.9986、相对 L1 0.0157；各自重跑追踪会放大矩阵差异 |

## 当前整链和随机重复

`fnit UKBConnectome_pipeline --atlas schaefer200+tian-s1` 已直接读取校正 DWI、旋转梯度和官方 `recon-all` 目录，自动生成 216 节点 atlas、追踪、SIFT2 及四张矩阵。真实 1000 次播种的[输出检查与四矩阵图](atlas_synthmorph_20260929.md)显示 216 个节点全部存在、四矩阵有限且对称；这一小样本仅验收接口和文件结构。

固定真实 FOD/5TT、10,000 个 GMWMI 位置以及同一 20 节点 atlas 后，FNIT 三次分别接受 2,713、2,727、2,691 条，MRtrix 三次为 2,758、2,728、2,727 条。官方自身 count 上三角相对 L1 为 0.2283–0.2583，跨软件九组为 0.2300–0.2933，其中 4/9 进入官方范围；共同边 mean FA 归一化 MAE 官方为 0.0777–0.1014，跨软件为 0.0643–0.1039，其中 6/9 进入范围。完整矩阵、输入哈希、时间、显存及脑图见[当前追踪报告](ifod2_rejection_20260929.md)。尚不能称最终 connectome 与官方流程匹配。

100k 的[三次对三次 20 节点对照](tracking_100k_three_seed_20260929.md)使用同一 FOD/5TT/GMWMI/FA/atlas。count 全边相对 L1 官方自身为 `0.094–0.117`、跨软件为 `0.076–0.119`，其中 6/9 进入官方范围；FNIT 自身为 `0.084–0.099`。跨软件长度 KS `0.0090–0.0148`，高于官方内部 `0.0054–0.0070`；端点和 8 mm TDI 相关也全部低于官方内部范围。[七套图谱的 100k 比较](seven_atlas_100k_20260929.md)进一步确认固定轨迹下的矩阵赋值逐值接近，但独立轨迹仍未全面匹配；1,000 万次播种尚未验收。

[历史 10k 起始组织分层](sgm_tracking_20260929.md)把长轨偏差定位到已接受流线内部：两套实现各接受 308 条皮层下起始流线，FNIT 的 75% 长度短 6.00 mm。后续合并的精确几何与掩膜线性播种修正消除了种子落在白质侧的错误，但没有消除皮层下长度差异。

[已接受种子的精确仿射检查](act_geometry_accepted_seeds_20260929.md)发现普通 NIfTI-1 使 FNIT 与官方的 ACT 单向标志在 2,795 个 FNIT 种子中有 93 个分歧；保持同体素、精确仿射的 NIfTI-2 后分歧为 0。合并掩膜线性播种和双精度 affine 后，独立 100k 追踪接受 27,549 条，count 相对 L1 与支持 Dice 的三次跨软件比较均进入官方重复范围；长度、端点和 TDI 尚未全面匹配。精确 FOD 仿射的 10k/100k 单项检查和脑图也在该报告中，不能以单一矩阵指标宣称整个 connectome 已一致。

100k 的[可选编译核复跑](tracking_compile_20260929.md)降低了这次实测的追踪墙钟时间；FNIT 保留数与未编译运行相差 48 条，独立轨迹群体偏差未消除。当前未测 100 万及 1,000 万次播种。

![当前三种子四矩阵误差比较](ifod2_rejection_20260929/rejection_act_3x3.png)

## 复跑和输出

安装、每个参数及输出结构见[中文函数文档](../../../docs/connectome/README.md)。当前单模块追踪和 3×3 随机对照入口为 [`benchmark_connectome_ifod2_rejection_matrices.py`](../../../tools/benchmark_connectome_ifod2_rejection_matrices.py)；配准隔离实验为 [`benchmark_connectome_registration_sensitivity.py`](../../../tools/benchmark_connectome_registration_sensitivity.py)。官方参考命令留在相应报告的独立 benchmark 脚本中，不进入 FNIT 运行时。
