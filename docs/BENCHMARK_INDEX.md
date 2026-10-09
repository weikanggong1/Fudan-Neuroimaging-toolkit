# FNIT 最新 CPU/GPU benchmark 索引

> 更新：2026-10-09。这里集中列出各子功能说明页中**当前可公开、可复核**的真实数据结果。原有功能页的输入、参数、原软件命令、历史版本和脑图结构不变；本页只提供统一入口和最新聚合数字。共享节点、冷启动、I/O 和不同流程范围会单独标注，不能把不同口径的墙钟直接相加或外推为稳定加速比。

## 读表规则

- `CPU` 和 `GPU` 只表示该行实际使用的执行设备；没有可复核的实测值写为“未测”，不以模拟数据补齐。
- “逐位/逐值一致”只适用于表中列出的输出或固定阶段；“科学内容一致”不等于每个浮点值逐位一致。
- 共享 GPU 利用率为 99–100% 或显存监测失败的结果，只能作为观察或数值回归，不能作为稳定资源/加速结论。
- 所有公开表均不包含被试 ID、UKB 字段、原始影像、表型名称或服务器地址。

## fMRI 与群体分析

| 子功能 | 最新 CPU 结果 | 最新 GPU 结果 | 精度范围与公开记录 |
|---|---|---|---|
| `fMRIVolume_pipeline` | 已有 1/8 线程 FNIRT、SynthMorph 配对；MNI 全网格 RMSE/r：FNIRT CPU1 `67.169521/0.525382`、CPU8 `68.034841/0.500816`；SynthMorph CPU1 `39.852096/0.742280`、CPU8 `40.927701/0.732729`。 | H100 当前完整 API 1225.84 s，PyTorch peak allocation 13.31 GB；FEAT 700.033 s、PICA/AROMA/混杂回归 171.502 s、clean MNI 重采样 47.151 s、T1w/MNI 单次插值与写出 254.356 s。 | 同源码优化前后 395,140,404 个科学输出逐值一致；fMRIPrep 25.2.4 体积-only、STC 关闭、8 线程为 3775.517 s，范围和硬件不同，不能直接作通用倍率。见 [fMRI 体积页](fmri/README.md)。 |
| `fMRISurface_pipeline` | 固定490帧采样CPU1/8历史：FNIT412.364/128.419 s，官方466.650/268.903 s。最新实际球面官方CPU4采样/CIFTI：MSMSulc85.654 s、MSMAll88.394 s；仅投影时钟。 | 最新冻结源码10/10：完整MSMSulc-only API均值270.562 s、MSMSulc＋MSMAll API345.813 s；自产21C8.113 s，连续两分支外层626.908 s；最大allocation/reserved2.390/11.044 GB。 | 两分支所有保存阶段absolute翻折0/0、新增正常面翻折0、全180帧finite。首例独立官方全帧采样56,255,760值exact，科学轴/intent/NIfTI头一致；完整XML节点顺序及GIFTI provenance不同。C-only；复用已完成volume/recon，未将旧NMI整链记为等价。见 [最新surface验证](../validation/msm/native_qc_20261009/a100/README.md#最终修复版完整-surface)。 |
| `MSMSulc` / `MSMAll` | 最新同机总预算8、左右各4：原版507.392/507.310 s，FNIT347.914/312.104 s（完整配准API、含读写），本次观测比1.458/1.625。 | 最终冻结版A100完整核心130.669/97.685 s，allocation2.241/2.266 GB；缓存优化历史ABBA均值91.109→83.939/72.162→68.859 s。本次自产21C完整GPU139.522 s/2.275 GB。 | FNIT CPU8/GPU保存坐标、faces对严格CPU1逐位一致；最新原版CPU8自身Sulc R/All L相对严格CPU1角差mean0.120077/0.032034°，不声称同线程输出都exact。report保留2/0；显式repair全十例0/0。见 [A100与CPU验证](../validation/msm/native_qc_20261009/a100/README.md)。 |
| `MS-HBM` | 真实 490 帧、8 线程：FNIT 603.380 s，官方 1741.111 s；优化版输出与官方在 CPU8 全部标签一致（CPU1 有 1 个标签差异）。 | 本轮没有新的可复核 GPU 端到端时钟。 | MSC02 100 帧配对：FNIT 141.29 s、CBIG 145.71 s，64,984 标签逐值一致、Dice=1。见 [MSHBM 页](mshbm/README.md)。 |
| `MELODIC/PICA+ICA-AROMA` | 本轮没有新增同口径 CPU 全链时钟。 | 固定 490 帧/95 成分/40 步：FNIT fit+write 117.49 s，原始完整命令 651.87 s（含 HTML）；时间/空间成分中位相关 0.999999978/0.999999970，阈值 Dice 0.999562。见 [MELODIC 页](melodic/README.md)。 |
| `BWAS` | ABIDE II 32 人 CPU 核心 1.05 s（仅统计核心）；完整 FNIT run_bwas（I/O、平滑、聚类、写出）1.79 s，范围不同。 | ABIDE I+II 1748 人、6,296,047,005 对连接：FNIT GPU run_bwas 3828.74 s，peak allocation 11.66 GB。 | 32 人 130,816 对：z MAE 3.70e-6、max 7.75e-5、CDT disagreement 0；1748 人抽样 224,428 对：MAE 3.34e-6、max 3.78e-5、CDT3/5 disagreement 0。共享 GPU 单次观察，不外推稳定倍率。见 [BWAS 页](bwas/README.md)。 |
| `SuperBigFLICA` | 5000 人公开真实运行是 GPU-only；本轮无同输入、同线程 CPU 全链对照。 | H100、20 成分、3000/1000/1000、50 epoch：完整 API 660.04 s，训练/选模/预测 101.37 s，空间统计/脑图 18.30 s，allocation/RSS 12.25/1.39 GiB；连续目标测试 r/R² 0.1731/0.0254，二分类 AUC/balanced accuracy 0.7226/0.6716。 | 新旧 HDF5 输入的模型/course/预测逐值相同，空间图最大差 4.77e-7；原仓库只覆盖 8 人前向/梯度控制，没有 5000 人端到端原软件对照。见 [SuperBigFLICA 页](superbigflica/README.md)。 |

## dMRI

| 子功能 | 最新 CPU 结果 | 最新 GPU 结果 | 精度范围与公开记录 |
|---|---|---|---|
| `DMRIPipeline` 原始 DWI→TBSS | CPU8 旧版 36,821.45 s → 优化版 25,313.29 s；47 张影像逐位一致，另外 19 个结果科学内容一致。EDDY 阶段约 24,460 s 受共享节点竞争，仅作观察。 | TOPUP 新版 GPU 场图 RMSE 0.0108 Hz、API 中位 10.82 s；完整链观测 6.75 min。 | 标准空间九张图相关 0.9880–0.9987，非逐体素等价；十人正式整链和本轮 CPU 报告分别见 [dMRI pipeline 页](dmri_pipeline/README.md)。 |
| `TorchAMICONODDI` | 本轮未提供同输入 CPU wall clock。 | H100 单被试：AMICO 81.79 s、classic Watson 209.55 s；两模式 PyTorch peak allocation 均 9.95 GB。 | 输入 104×104×72×105、全脑 mask 242,261 体素；输出有限值、同网格、mask 外为零；24 个真实体素与 MATLAB NODDI 核对。测试时共享 GPU 利用率 99–100%，不宣称稳定加速比。见 [AMICO 页](amico_noddi/README.md)。 |
| `TorchProbtrackX` | 两组真实完整流程：旧版 6818/6086 s，CPU 优化版 4275/4317 s；完整输出严格一致。 | 单 way-point/avoid H100 公开配对仍见原页，peak allocation 2.81 GB；随机数流不同，非逐轨迹等价。 | 两组 CPU 结果约缩短 37%/29%，限定为同输入、同线程两次观察。见 [ProbtrackX 页](probtrackx/README.md)。 |
| `TorchDTIFIT` | 真实第二壳层 CLI（含 I/O）3.43 s；官方同范围秒数当前未形成可复核公开表。 | 本轮没有独立、完整资源记录。 | FA max abs 1.19×10⁻⁷、tensor/eigenvalue max 2.33×10⁻¹⁰；MO/方向图仍有单体素较大差异，不能写成全输出通过。见 [DTIFIT 页](dtifit/README.md)。 |
| `UKBConnectome_pipeline` CSD | 本轮没有可复核 CPU 端到端速度表。 | CSD 完整 API 旧/新 3436.72/3376.50 s；WM/GM/CSF 三套数组及 NPY 逐位一致。 | CUDA 进程显存监测失败，因此精度回归通过但资源/速度验收不通过；不得写成稳定 GPU 加速。见 [Connectome 页](connectome/README.md)。 |
| `UKBConnectome_pipeline` iFOD2/ACT | 同机 MRtrix 8 线程完整 tckgen（含 I/O）：100k seeds 为 16.032–16.286 s。 | 单 H100 固定输入追踪：默认热中位数 238.650 → 177.532 s；已有编译模式 116.345 → 107.344 s。ABBA Torch allocation/reserved 最大 1.076/1.216 GB；独立完整 100k 进程显存采样峰值 3.127 GB，监测覆盖 gate 通过。 | 新旧版在相同模式中全部流线与 TCK 逐字节一致，74 项 GPU 回归通过。每版两个热样本、共享负载波动，不承诺稳定倍率；不是原始 DWI 全流程或 MRtrix 随机重复等价验收。见 [追踪性能页](connectome/TRACKING_PERFORMANCE_OPTIMIZATION.md)。 |
| `MMORF` | 未发布官方速度/精度对照。 | 未发布官方速度/精度对照。 | 官方参照程序完整性失败并异常退出，故不纳入 benchmark；生产实现仍可按自身验证记录使用。 |

## sMRI、配准与重建

| 子功能 | 最新 CPU 结果 | 最新 GPU 结果 | 精度范围与公开记录 |
|---|---|---|---|
| `SynthStrip` | 两例官方→FNIT：45.56→14.90 s、49.20→14.03 s；脑图/掩膜相同。 | 真实 b0 掩膜 Dice 0.9999908（约 27.1 万脑内体素仅 5 个不同）。 | 两种结果均保留原页的输入和图示范围。见 [SynthStrip 页](synthstrip/README.md)。 |
| `SynthMorph` | 8 核 256 ABBA 中位数 163.094→155.086 s，缩短 4.91%；首次 JIT 成本单列。 | 本轮没有新的完整 GPU 端到端时钟；CUDA 回归保留在原报告。 | 六次 CPU 输出逐位一致；对象输入正/逆向 NRMSE 4.7×10⁻⁶/5.1×10⁻⁵，逆向边界门未通过。见 [SynthMorph 页](synthmorph/README.md)。 |
| `SynthSeg` / `SynthSegPlus` | C24、8 线程 API 85.499→80.621 s，缩短 5.71%；FNIT 新旧输出逐字节一致。 | GPU 路径优化前后保持；本轮没有新的显存峰值表。 | 与官方仍有 1 体素差异；SynthSegPlus 当前无完整等价验收。见 [SynthSeg 页](synthseg/README.md)。 |
| `TorchFAST` / `FastVBM` | FSL 兼容实测 388–395 s，FNIT 122–133 s；PVE 有少量差异。 | 本轮没有新的完整 GPU 端到端时钟。 | 速度为同节点观察，PVE 差异保留在原页。见 [FAST 页](fast/README.md)。 |
| `recon-all` | 完整 T1 链：官方 76.67 min，FNIT 80.50 min；68 个皮层区平均厚度绝对差 0.017 mm。 | GCSA cache 优化只有历史双侧 annotation 记录，本轮无新的端到端 H100 时钟。 | 顶点网格未建立有效对应，不能写成表面逐点等价；原始 T1 全亚区 FNIT 104.02 min，原网格 4/105、高分辨率 5/105 分区通过。见 [recon-all 页](recon_all/README.md)。 |
| `segment_subregions` / `segment_nuclei` | 脑干拟合占约 91% API；优化后后验/拟合状态相同，API 降低 8.76%。原始 T1 全亚区 104.02 min；官方同范围完整时钟缺失。 | 本轮没有新的完整 GPU 端到端时钟。 | 原网格仅 4/105、高分辨率 5/105 分区通过，不能外推完整亚区等价。见 [亚区页](subregions/README.md)。 |
| `robust_register` | 真实安装包 20/20 精度门通过；冷 API 2.254 s，导入另计 1.594 s；官方冷 CLI 更快。 | 未测完整 GPU 速度。 | RAS/几何/保存字段合同通过，保持“不宣称提速”。见 [robust register 页](robust_register/README.md)。 |
| `TorchFNIRT` | 2026-10-07 新增 392 项 saved-basis 有限对照：max abs 1.1926×10⁻¹⁸、relative L2 3.2073×10⁻¹⁵；参考 1.074 s、监督总计 1.468 s。 | 原 CUDA 路径未改，本轮没有新完整 GPU 计时。 | 仅为 392 项累加，不包含完整 1177/Hessian/PCG/采样/最终配准，不能写成完整 FNIRT 等价。见 [FNIRT 页](fnirt/README.md)。 |

## 可追溯报告

各页链接到 `validation/` 下的机器可读 JSON、README 和公开图。发布前应重新检查报告中的 `source_sha256`、输入范围、线程数、设备、I/O 是否计入以及共享负载说明；不以本索引代替原始报告。
