# 任务 05：完整 volume 与 helper 的 CPU 基线准备（2026-10-04）

本页记录可启动的基线和覆盖范围。**本阶段没有启动正式计时、CPU profile 或 GPU 作业，也没有修改生产函数。** baseline 为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。默认 STC 关闭；正式运行须由协调者从待启动清单派发。

## 1. 功能矩阵

| 模块/入口 | 实际支持功能与主要参数 | 基线覆盖和独立参照 |
|---|---|---|
| `fMRIVolume_pipeline` / `fnit-fmri volume` | 原始 BIDS T1w/BOLD；subject、session、task、run、acquisition、direction、reconstruction、echo 与显式 t1w_image；固定 MNI 模板/脑掩膜；FNIRT/SynthMorph 两种 registration_backend；device、batch_size、motion_iterations | 完整公开 180 帧 raw BIDS，两个配准后端各 1/8 CPU；原 fMRIPrep 25.2.4 独立 volume-only 1/8 CPU。两者算法和清洗范围逐项标明，不把总时间直接当作同功能比值。 |
| volume preproc | T1w 原生 BOLD 分辨率参考与 MNI；原始或可选 STC 的 BOLD，通过 HMC+BBR+非线性组合一次插值；`grid-constant`、`fmriprep` 坐标精度 | 默认 STC OFF；完整 FNIRT+STC 8 CPU 变体入口已准备；完整 490 帧 World 成熟实现的最新证据单独复用。 |
| volume clean | FEAT MCFLIRT spline、输入 dtype 回写、mask、grand mean scaling、Gaussian highpass、PICA、AROMA；可选 WM/CSF/motion/global 回归、motion_model、bandpass、aroma_mode、ICA components/max_iter/n_splits/random_state | 完整默认 ICA max_iter=500 / n_splits=1000；不降迭代、不裁帧。AROMA/confounds 的算法与官方对照由任务 02 提供，本任务计完整管线阶段与四份最终体积。 |
| 缓存/发布 | overwrite、reuse_anatomical、BIDS 派生路径、JSON/provenance、四份最终 4D 体积、每帧 RAS pull、掩膜/配准/QC、transaction publish | 同进程完整重复调用；新输出 cold 与完整 anatomy cache warm 分开记录。warm 仍重算运行级 MC/清洗。官方 warm 是新进程复用自己的工作流缓存，另列语义。 |
| `_anatomical.prepare_anatomical` | SynthStrip、分割、T1 配准、producer lock、manifest-last；T1/template/mask/weights/source/backend/execution/dtype/device/TF32/env 指纹与输出 SHA 校验 | 原 BIDS 完整调用观察真实 cache.reused、prepare_anatomical/register_t1_to_mni 阶段时间。覆盖缓存复用/无缓存/失配/overwrite 的源码合同；失配变体尚待后续真实运行，不能写成已完成。 |
| `run_feat_core` | BIDS selectors、显式 brain_mask；SynthStrip/Otsu；motion_iterations、batch_size、device；spatial_warp、postmat；highpass cutoff；overwrite 和旧 MAT 清理 | 完整公开 180 帧 1/8 CPU cold+warm；MCFLIRT 官方由任务 01；独立 temporal/helper 由本任务。发现 fieldmap 而无显式 spatial_warp 时应报错，不能静默忽略。 |
| `epi_brain_mask` | 完整 3D EPI 参考；正有限值 Otsu 512 bins、最大连通域、填洞、dilation（默认 2，允许 0） | 真实完整公开 EPI 参考，1/8 CPU；这是 FNIT 独立 Otsu 方法，不声称等同 BET。保存 uint8、原参考网格。非默认 dilation 尚待完整变体。 |
| `grand_mean_scale` / `scale_nifti` | 完整 4D × 3D mask，mask>0.5，FSL p50 的 floor(N/2) 项；target_median 默认 10000；array/API 与 NIfTI 仿射检查 | 完整真实 490 帧既有 MCFLIRT 输出；FSL `fslstats -k -p 50` 和 `fslmaths -mul`；1/8 CPU。掩膜用于 p50，不替换成时间中值。 |
| `gaussian_highpass` / `highpass_nifti` / matrix helper | sigma_volumes 或 cutoff_seconds/(2*TR)；voxel_chunk；preserve_mean 默认 True / False；局部 Gaussian 直线拟合、截断边界、残差时间均值去除、可加回原均值；TR 时间单位 | 完整真实 490 帧，FSL `-bptf sigma -1` 后按需 `-add mean`；1/8 CPU 默认与 remove-mean 变体。不能改变 T×T double 算式前先完成该基线；CUDA 原 double 计算、TF32 全局规则保持。 |
| `slice_timing_correct` | 真实 SliceTiming、RepetitionTime、i/j/k 与反向 SliceEncodingDirection；reference_fraction 0..1、ignore、voxel_batch、device；Fourier 插值、忽略帧原样保留 | 完整公开 180 帧、42 个真实切片时间；原 AFNI `3dTshift -Fourier -TR ...s -tzero ... -ignore ... -tpattern @...`；1/8 CPU。ignore=4/reference=0 完整 8 CPU 变体；原实际数据为 k 轴，其余轴源合同与后续测试单列。 |
| `native_bold_sampling_reference` | 完整 T1 与同网格 fov_mask、真实 BOLD 分辨率，RAS 网格，体素尺寸 3 位小数，脑包围盒+2；qform/sform=2 | 原 NiWorkflows `GenerateSamplingReference`，容器内 NiWorkflows/Nilearn；完整真实固定 T1/WM mask 和同例完整 BOLD 几何在两实现保持一致。该微基线声明 WM FOV，不把它当成生产脑掩膜。1/8 CPU。 |
| `apply_motion_warp` legacy | 完整 FSL scaled-mm T×4×4；warp None/relative/absolute/三次系数 embedded affine；postmat；linear/spline；batch_size、device；mcflirt 特殊 Constant/extraslice 协议 | 本队列完整 490 帧 motion-only 的 linear/spline、1/8 CPU；完整非线性/postmat World 链独立成熟证据与任务 01 MCFLIRT 链单列。legacy 分支还需要后续真实 warp/postmat 变体，未执行不称覆盖完成。 |
| World/helper `resample_world` 与最终 `_resample_final_volume` | 3D/4D，RAS pull、每帧运动、linear/spline/nearest，fmriprep/float64，grid-constant/periodic，batch/chunk，FNIRT 或 SynthMorph 生产包装 | 现有完整 490 帧正确官方 NiTransforms 对照在源码 SHA 相同后复用。现有 `benchmark_applywarp_world_cpu.py` 原参数入口+任务 05 新 CPU group 复跑命令已准备；未重复改成熟采样器。 |
| `register_t1_to_mni` | 原生 FNITFLIRT 12-DOF corratio 初始化；SynthMorph 或 FNIRT；模板 mask、FNIRT config、optimized/reference；FLIRT affine、RAS pull、QC/phase | 两种 raw BIDS 完整 pipeline 分别观察时间与输出；成熟注册模块官方局部对照继承其已有真实证据边界。优化先检查此函数调用各成熟模块的合同。 |
| BIDS、derivatives、timing | inherited JSON、唯一 T1/BOLD 选择、TR 一致性/固定 TR、fieldmap associations；保护已有输出；完整 metadata 时间单位、VolumeTiming/DelayTime/AcquisitionDuration、STC 标志 | 完整原 BIDS 路径运行触发真实继承/保存/覆盖；输入 SHA 在计时外。错误输入/多 echo/多 T1/稀疏时间等非本数据特征不当作新真实 benchmark。 |

没有独立 CLI 的 helper 以 Python API 和原软件命令说明；现有生产 CLI 由协调者维护。

## 2. 真实资源与许可核对

- 公开完整 BOLD `(64,64,42,180)`，T1 `(160,256,256)`，TR=2.1 s，真实 SliceTiming 共 42 项，原 BIDS 与完整参考可读；输出公开图只能来自许可允许公开的该公开数据。
- 补充真实完整 BOLD `(88,88,64,490)`，TR=0.735 s；正式默认 MCFLIRT 490 矩阵、参数、完整校正体积和匹配掩膜可读。该数据只发布聚合指标，不发布个体路径或脑图。
- 固定 MNI 模板、脑 mask、SynthStrip 和 SynthMorph 权重已按当前 manifest/文档核对大小和 SHA；SynthMorph deform.3 为 3,508,630,424 bytes，SHA `95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536`。复用既有明确来源资源，没有下载或再分发。
- 官方 FSL 6.0.7.22、fMRIPrep 25.2.4 的独立进程仅存在 validation；生产函数没有新增外部运行依赖。FSL 来源/许可参考项目原有 FSL reference notices；NiWorkflows Apache-2.0 和 Nilearn BSD-3-Clause 归属见 `THIRD_PARTY_NOTICES.md`。
- fMRIPrep SIF 大小 2,413,375,488 bytes。既有固定 SHA 为 `8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39`；adapter 在计时外重新校验。容器版本/CLI 已现场核对；容器内 `3dTshift` 位于 `/usr/local/bin/3dTshift`。
- 容器里没有可执行 `3dTproject`（PATH 和受限目录扫描均未发现）。已报告任务 02；不能将独立 NumPy 当成该官方混杂回归证据。
- 官方 TemplateFlow cache 已现场核对 MNI152NLin6Asym、MNI152NLin2009cAsym、OASIS30ANTs 目录与文件数；adapter 计时外复制到自身输出目录，运行不改共享旧缓存。

## 3. 可复用最新证据的范围

`src/fnit/_world_resampling.py` 当前 SHA `0b24714e94085feb4d40762e9144789d9f8b835aa42d00614a612be8c3046d83` 与最新完整 World CPU 报告匹配。完整 490 帧在 1 CPU 原版→FNIT API 为 291.460→239.193 s、8 CPU 为 46.610→45.040 s，各一次观察；进程时间分别 301.475→241.778 / 56.297→46.831 s。脑内 RMSE `1.487e-9`，全网格 `2.325e-8`，完整候选 1/8 CPU NIfTI SHA 相同。正确官方 8 CPU 需 OpenBLAS=1，旧 nested BLAS 的失败对照不能混入。

这些是原有最新实测的复用，**不是本任务的新重跑**。CUDA 完整输出已有 bit gate，单次共享 GPU 34.103→36.943 s 不能支持 GPU 性能不变结论；本轮新 GPU 验收由协调者按锁派发。公开十例与 threeway 的冻结记录用于完整输出/参考范围对照，不能把旧 frozen revision 写成当前源码重测。

参阅 `docs/applywarp/WORLD_CPU_BENCHMARK_20261004.md` §5.5 和 `docs/applywarp/world_cpu_latest_20261004.public.json`。

## 4. adapter、controller 和正式启动

提交 `benchmark_baseline.py`；私有输入、源目录、CPU dispatch、pending launch、run_manifest、controller 在外层 `outputs/fnit-fmri-cpu-20261004/task05_volume/`，不提交。

- 独占 physical CPU：`48,52,56,68,72,76,80,88`；1 线程固定 CPU48，8 线程使用全组；同一专属 lock；GPU 不可见。
- 33 项待执行作业，分 helpers / spatial / variants / feat / volume / volume_official / volume_variants。默认 full 180/490 帧；未降低默认迭代或 PICA 参数。
- controller 默认只准备；`--start` 才计算；`--groups` 支持分阶段派发，保存 PID、source commit、adapter SHA、退出状态与每个阶段 JSON。作业失败会保留日志并继续收集后续，最后返回失败状态，不把成功退出替代精度。
- FNIT首次/同进程 warm、官方新进程/官方 workflow warm、anatomy reuse 均单列。输入/SIF/source SHA 与准备复制在 API 计时外。正常 NIfTI 保存包含在 API 内；阶段观察写 JSON 的开销需要随最终计时范围说明。
- fMRIPrep volume-only 原始 BIDS 耗时预期小时级，1 CPU可能更长；FNIRT/SynthMorph完整 default CPU 也可能数小时。估计用于排队，正式结论必须来自产物。
- 官方 fMRIPrep预处理只输出 preproc，没有额外 FSL grand scale/highpass/AROMA clean；FNIT四份最终输出的完整API、可比preproc阶段和额外clean阶段各列，避免算法范围差异造成速度误读。
- 完整 World 490 复跑使用既有成熟 adapter，单独 pending command 与 manifest；当前源码一致，可由协调者决定复用或按组重跑。

## 5. 源码热点与下一阶段

1. `gaussian_highpass` 当前 T×T 双精度投影的每个空间 chunk matmul 是潜在 O(T²) 热点；先保留当前 matrix float32→double、边界/均值合同做 FSL 全490对照。随后仅 CPU 优化，CUDA 分支保持原路径。
2. grand p50 完整被掩膜数据复制/partition、gzip读取/保存需要分别观察；不能用降低 mask/帧数改善时间。
3. legacy spline 每帧 SciPy 系数和整网格拉取可能有重复开销；先按实际基线定位，成熟 World 已优化路径不重复改。
4. 完整管线的主要 CPU 费用预计来自 anatomy 配准、MCFLIRT/BBR、PICA与最终采样；StageObserver 包装原函数记录，不替换算法。跨模块成熟 bug 先报告协调者并归模块所有者修复。

下一阶段按真实报告决定 CPU-only 改动和功能变体顺序，再生成精度/性能汇总、公开脑图及最终中文模块说明。当前准备矩阵里的“待后续”均未宣称完成。

## 6. 原软件说明与参考

- FSL FEAT：<https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/index.html>；FSL source：<https://git.fmrib.ox.ac.uk/fsl>。
- AFNI Tshift：<https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html>；AFNI source：<https://github.com/afni/afni>。
- fMRIPrep：<https://fmriprep.org/en/25.2.4/>；source：<https://github.com/nipreps/fmriprep/tree/25.2.4>。
- NiWorkflows grid：<https://github.com/nipreps/niworkflows/blob/1.14.4/niworkflows/interfaces/nibabel.py>；Nilearn：<https://github.com/nilearn/nilearn/tree/0.11.1>。
- Esteban et al., *Nature Methods* 16, 111–116 (2019), fMRIPrep；Jenkinson et al., *NeuroImage* 17, 825–841 (2002), motion correction；Smith et al., *NeuroImage* 23, S208–S219 (2004), FSL。
