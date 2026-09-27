# recon-all Python/CUDA 实测索引

本目录记录同一真实 T1 的逐阶段对照，以及从原始 T1 连续运行后的文件级验收。主要影像是仓库中的去标识 `examples/data/sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）；各报告另列实际输入哈希、主机、命令和时间。冻结官方中间结果的同输入通过，只证明该函数在那些输入上匹配，不能代替 FNIT 自产上游输入的连续验收。

## 当前整例结果

最近一次从原始 T1 完成的 v3 调度运行了 39 步、用时 2903.79 秒。与 FreeSurfer 8.2 的同一被试逐文件比较，**19/138 项通过、47 项缺失、72 项不同**；13 张上游 MRI 图在体素、类型、仿射和 MGH 头前 284 字节上相同。[整例原始报告](native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)记录那一版的分步时间和脑区/顶点差异。此后拓扑、Talairach 精度和若干表面步骤已更新，**没有新的完整 138 项运行**，因此 v3 耗时和通过率不能当作当前代码的等价重建结果。

[发布门槛](RELEASE_GATES.md)规定从空被试目录起跑、逐体素与逐顶点对齐、脑区行比较和配对计时的验收方法。[严格比较器](compare_complete_subject.py)覆盖 39 张 MRI、18 个表面、46 张顶点图、12 个注释和 23 个统计文件：

```bash
python validation/recon_all/python_gpu_port/compare_complete_subject.py \
  /path/to/official-subject /path/to/fnit-subject \
  --report /path/to/compare.json
```

比较器的官方自比为 138/138；人工修改厚度或面积 0.02 单位时，仅对应图失败。该负对照核实了比较器灵敏度，不说明候选重建已通过。

## 当前首差与连通试验

| 位置 | 真实数据结果 | 记录 |
| --- | --- | --- |
| 旧 v3 整例的 MRI 前缀 | 当时的 13 张 MRI 图逐体素、仿射和头一致；尚未对更新后的代码重跑整例 | [v3 原始报告](native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md) |
| Talairach 精度修正与有限续跑 | 另一次从同一 T1 开始的定向测试中，eTIV 误差由 822.547 降到 0.895 mm³；后续 GCA、`norm`、`aseg.presurf` 匹配，但没有继续到 WM/皮层 | [Talairach](talairach_tf32_isolation_20260927/README.md)、[LTA 坐标转换](../../../docs/recon_all/TALAIRACH_LTA_NUMPY.md) |
| WM/filled 保存阶段 | 独立同输入试验核对了白质及填充体积图；这不是修正后从原始 T1 连续运行的结果 | [WM 链](native_cpp_conda_20260927/wm_chain_20260927/REPORT.md) |
| 候选左侧拓扑与白质前缀 | 自产 MRI 输入的保存阶段复跑得到逐点相同的 `orig.premesh` 和 `orig`；`white.preaparc` 平均顶点偏移 0.000421 mm，50 个顶点超过 0.1 mm；`smoothwm` 为 0.000311 mm，18 个超过 0.1 mm | [候选前缀](white_connected_prefix_20260927/README.md)、[双侧冻结输入拓扑](../../../docs/recon_all/TOPOLOGY_CONDA_GA.md) |
| 候选左侧标准球面 | 修正采样角度后，同候选输入的完整 Python/官方 `sphere` 有序顶点、面完全一致；Python 884.56 秒、官方 328.61 秒。对归档官方均差 2.959 mm，首个已保存上游几何差异在 `white.preaparc` | [完整单阶段验收](candidate_sphere_first_difference_20260927/full_stage/README.md)、[上游首差](candidate_sphere_first_difference_20260927/full_stage/UPSTREAM_FIRST_DIFFERENCE.md) |
| Python white 放置首轮 17 步 | 左半球冻结官方输入上，第 17 步与已安装官方 FreeSurfer 的首轮 RAM 曲面最大逐顶点距离为 0.00001641 mm，0 个顶点超过 0.0001 mm；尚未完成四轮或接入整例 | [首轮函数与报告](../../../docs/recon_all/WHITE_PYTHON_FIRST_PASS.md) |
| 最终 white 与 pial | Conda final white 在**左半球冻结官方输入**上的平均偏移 0.000358662 mm，57 个顶点超过 0.1 mm；Conda pial.T1 在同范围的平均偏移 0.0296608 mm，5,571 个超过 0.1 mm。独立 Python pial.T1 在双侧冻结正确输入上逐坐标相同，但尚未接入整例 | [final white](../../../docs/recon_all/FINAL_WHITE_CONDA.md)、[Conda pial](../../../docs/recon_all/PIAL_T1_CONDA.md)、[Python pial](../../../docs/recon_all/PYTHON_PIAL_PLACEMENT.md) |
| 脑区和顶点指标 | 对官方 white/pial、标签及注册球面等冻结输入，部分 Python 顶点图、注释和统计函数已单独验收；自产整例中的最终几何、注册球面和脑区输出仍未通过 | [厚度](thickness_stage_report.json)、[面积](area_cuda1_report.json)、[曲率](curvature_stage_report.json)、[注释](MRIS_CA_LABEL_STATUS.md)、[发布门槛](RELEASE_GATES.md) |

## 逐函数报告入口

函数参数、输入和输出结构、等价 FreeSurfer 命令及真实数据的精度/时间，见各功能说明或其原始对照记录：

- **载入、预处理、强度校正：** [NIfTI 导入](NIFTI_IMPORT.md)、[conform](CONFORM.md)、[N4 包装](../../../docs/recon_all/N4_WRAPPER_VALIDATION.md)、[去噪](ANTS_DENOISE_STATUS.md)、[T1/brain 归一化](../../../docs/recon_all/NORMALIZATION.md)、[CA 归一化](CA_NORMALIZE.md)。
- **体积分割和后处理：** [GCA 注册](MRI_EM_REGISTER_VALIDATION.md)、[初始白质分割](MRI_SEGMENT_VALIDATION.md)、[WM/aseg 编辑](WM_ASEGEDIT_FIXED.md)、[胼胝体](MRI_CC_PYTHON.md)、[体积掩膜](VOLMASK.md)、[MNI152/辅助标签](../../../docs/recon_all/MNI_AUX_CHAIN.md)、[brain.finalsurfs](../../../docs/recon_all/FINAL_SURFS_CHAIN.md)。
- **初始表面与球面：** [tessellate](TESSELLATE.md)、[pretess](PRETESS.md)、[初始平滑](SMOOTH_SURFACE.md)、[膨胀](INFLATE_STATUS.md)、[quick sphere](SPHERE_QUICK_STATUS.md)、[拓扑/重网格](../../../docs/recon_all/TOPOLOGY_CONDA_GA.md)、[标准球面](SPHERE_STANDARD_STATUS.md)、[球面配准](MRIS_REGISTER_STATUS.md)。标准球面现已通过同候选输入完整阶段核对；球面配准的旧逐点试验仍使用冻结官方输入，候选链整例尚未通过。
- **white、pial 与每顶点图：** [阈值](AUTODET_GWSTATS_STATUS.md)、[白质预放置](../../../docs/recon_all/WHITE_PREAPARC_CONDA_CHAIN.md)、[Python white 首轮 17 步](../../../docs/recon_all/WHITE_PYTHON_FIRST_PASS.md)、[smoothwm](../../../docs/recon_all/SMOOTHWM_FINAL_PARITY.md)、[Python pial](../../../docs/recon_all/PYTHON_PIAL_PLACEMENT.md)、[面积](area_cuda1_report.json)、[厚度](thickness_stage_report.json)、[顶点体积](vertex_volume_cuda0_report.json)、[曲率](curvature_stage_report.json)。
- **脑区、投影和统计：** [表面注释](MRIS_CA_LABEL_STATUS.md)、[label2annot](LABEL2ANNOT.md)、[表面标签转换](LABEL2LABEL_SURFACE_STATUS.md)、[surf2volseg](SURF2VOLSEG_CORTEX.md)、[ROI 曲率](ROI_CURVATURE.md)、[脑区统计行](ANATOMICAL_STATS_ROWS.md)、[全局统计](ANATOMICAL_STATS_GLOBAL.md)、[eTIV](ESTIMATED_TIV.md)。

`fnit-recon-all` 当前仍需三个必需、三个可选的 Conda 源码编译 C++ 程序，不需要系统安装的 FreeSurfer 运行包。[构建与调用说明](../../../docs/recon_all/CONDA_CPP_BUILD.md)列出固定源码、外置许可证和资产要求。阶段通过率只对报告中的同输入范围有效；当前完整发布状态以[发布门槛](RELEASE_GATES.md)为准。

[完整 Conda YAML 安装实测](conda_yaml_install_20260927/README.md)在独立新环境中编译了上述程序，并使 `mri_segment` 在真实 T1 冻结输入上获得 0/16,777,216 体素差。它验证安装与单阶段运行，不更新上文 v3 整例的 138 项结果。
