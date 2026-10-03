# CON11 显式实际来源路由（准备阶段）

当前没有新CON11 packing，没有启动新原软件或GPU任务。本目录只包含metadata核验工具，已运行进程的source/config未改。

旧CPU预算控制器1313等待旧heldout_aligned_v2的CON11成功CPU阶段；旧CPU阶段控制器等待原formal baseline CON11 packing。两者参数已加载，不能热改。旧比較器111428与完整交付器125935也绑定原baseline和统一官方root，直接继续等待不能覆盖新末例路径。

新root sidecar应明确提供以下键，均来自实际已出现的文件，不使用示例路径冒充实测值：

- `schema_version=1`、`case_id="sub-CON11"`、`version="baseline"`。
- `baseline_commit="da427c21a5a7619f4c3457c4406ab48a187ce390"`，`canonical_manifest_sha256`是task01原冻结canonical manifest实际SHA。
- `actual_fnit_case_directory`是新selected NVML namespace中真实baseline/sub-CON11绝对目录，不写到旧baseline、不建软链接。
- `packing`、`acqparams`、`config`、`driver_lineage_receipt`、`baseline_source_receipt`各为`{"path":绝对实际路径,"sha256":实际SHA}`。packing及acqparams必须属于上述真实病例topup目录。
- `baseline_source_receipt`是root冻结来源收据，含确切`code_commit`和`core_sha256`；core_sha256在已核实identity中指FLIRT core，对应现场baseline source flirt/core.py应为`981b61d822a4bacb2aba05637f1001cdd3114d5d70139d1f31f859bb3011aba9`。独立frozen_scientific_source_sha256.json还固定原实际baseline run的EDDY/TOPUP/FLIRT与connectome pipeline/bids科学文件SHA；`source_files`记录实际冻结源码文件path/SHA清单，不指向candidate。
- `scientific_parameters`含`seed=0`、`eddy_gp_seed=12345`、`tf32=true`、`pair_geometry="fslmerge-first"`、`official_EDDY_flags`完整数组（与原原生参考12项一致）。root提供的config/driver来源收据应明确该baseline、case及参数的对应关系，不将mutable进行中报告未来SHA写成已确认事实。

核验工具四个必填参数：`--origin`是root实际origin sidecar；`--manifest`是原canonical manifest；`--baseline-source`是该冻结da427 source；`--output`是全新核验收据。核验所有显式文件SHA、源码、参数、unique canonical b0逐体素与首AP affine、有效PE/readout；不借用FNIT field/mask/DWI，不初始化GPU、不执行原软件。

核验通过后再执行的路线：

1. 第一批九例继续原CPU参考和原baseline比较，报告/合同不重写。等九例全verified且原控制器仅等待末例、无native children，才安排末例，保持CPU并发最多2。若要退役旧待机控制器，只能重新核精确PID/start ticks/source、九例完成和无native children，保留原等待报告；准备阶段未发送信号。
2. CON11使用新官方namespace，复用科学source的`run_official_rawprep.py --subjects CON11 --workers 1 --phase all --solver cpu --fnit-selection-root <实际新baseline根目录>`。原工具CPU模式只允许fresh all-phase；不使用名称为gpu的两阶段入口。官方ROI/merge/TOPUP/SynthStrip/CPU8 EDDY全由该原始输入自行生成，同完整flags和实际AP ref。核验器得到的索引仅是共用原始输入元数据，不把FNIT场或mask复制给官方链。
3. 独立新completed合同核验工具绑定该CON11 namespace及origin receipt的SHA；比较工具绑定该自产官方末例与显式新formal CON11 preproc，待实际formal EDDY完整输出/QC存在才比较。不能拿旧baseline不存在的CON11路径、candidate或进行中输出填数字。
4. 另建明确per-case route sidecar给Task2：前九例official_case/actual_fnit_case仍指原完成证据，唯一CON11覆盖指向新自产官方结果和新实际formal结果，并绑定root origin、verified receipt、合同SHA。Task2据真实verified合同重核full102/gradients、自产field/mask来源和实际raw选帧；不是把新结果复制/软链接到旧目录。
5. 新最终collector/plotter逐例读取此route，前九例的时间仍是恢复旧CPU stage+新EDDY。CON11是全新连续CPU rawprep，必须另外标`execution_kind=fresh_official_CPU_rawprep`，记录实际fresh rawprep wall和新stage时间；不能伪造`source_CPU_stage_lineage`或`CPU_reference_activation_wall_seconds`，不能将末例新TOPUP标成恢复旧stage。现有collector直接索引恢复lineage，不能原样用于新末例。最终十例摘要、脑图和7节README须来自这个新route工具namespace。

目前只完成守卫/接口准备。没有root实测sidecar时不会推断CON11路径、索引、SHA，也不启动末例计算。

## v2 schema修复

真实canonical manifest使用case_id=sub-CON11、subject=CON11。v2严格按case_id选择唯一病例，再核subject.removeprefix("sub-")==CON11。原v1源码及SHA保留，不修改已部署v1。实际fixture来自原冻结manifest完整CON11记录，只有schema回归，不当MRI benchmark。
