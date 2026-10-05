# MSM 与 surface CPU 对照功能矩阵

起点 `cc9402734faeba93b3a13c29932fa1392eaccf62`。本页是测试覆盖清单；完成状态与数值由本轮实际报告填写，既有报告保留原源码范围。

| 功能 | 当前实现支持 | 本轮完整真实输入与原版对照 | CPU / GPU 边界 |
| --- | --- | --- | --- |
| `prepare_msmsulc_inputs` | recon-all 双侧 native sphere/sulc；FS→fsLR 仿射与 ROI；串行/双侧并行；覆盖保护；WB 路径；总 CPU 预算 | 真实双侧全顶点；固定 HCP 模板；原版 Workbench 仿射、重采样和 ROI 原步骤 | nibabel/NumPy、WB；该准备阶段不宣称 CUDA |
| `run_msmsulc` | 双侧；默认 HCP 四级；配置对象/原版配置；SSD/Pearson；optimized/reference；parallel；cpu_threads | 全 native 顶点与 164k 参考，完整四级/原停止条件；newMSM 实际 binary，CPU 1/8 预算 | PyTorch CPU/CUDA；有序 WLS/Rodrigues/HOCR/FastPD 是包内 CPU C++；CPU 改动还须回归 CUDA 全 API |
| `run_msmall` | 多列 Pearson；一级 coarse、三级 refine；initial sphere；双方/单方/无权重规则；源码浮点配置；optimized/reference；双侧执行 | 真实 WRN C 特征两侧各 32,492 顶点；完整一级和三级配置；newMSM 同输入 1/8 预算 | PyTorch CPU/CUDA；有序 graph 解算保留 |
| `compute_msmall_variance_normalization` | ICA mixing 数组/文本；显式 one-based noise list/文本/FIX 尾列表；sample SD 与下限；完整 CIFTI axis | 490 帧真实 clean CIFTI、106 ICA 列、84 noise/22 signal；固定 `ComputeVN.m` 已在 nodecw10 合法 MATLAB 实际完成 CPU1/8 基线对照 | PyTorch float64 CPU/CUDA；保存 float32 CIFTI |
| `run_msmall_regression` DR | matching CIFTI BOLD/reference axis；显式 VN 或 DR 的 `None`；one-based component selection；全部地图/weights/nodes/report | 全 490 帧；固定 d40 参考；原版 `MSMregression.m` 的 DR 与 DR+VN 已完成 CPU1/8；nodes 的原版 spectra 支路另列 | PyTorch float64 CPU/CUDA |
| `run_msmall_regression` WRN | 正均值 1 cortical area；d7–d21 全 15 组；14 mm WB smoothing；VN 必需；禁止常数时序；完整高维回归 | 全 490 帧、90,568 非常数有效 grayordinates；原版 MATLAB + WB 与候选 maps/weights CPU1/8 全部完成；nodes/spectra 原六项输出已写，nodes 全矩阵比较待导出 | 回归 CPU/CUDA，14 mm WB 为 CPU；不能用 shortened maps/frames 代替 |
| `prepare_msmall_inputs` | C、CA、CAT；RSN/weight/ROI；myelin、每轮 myelin bias、topography 显式输入；medial wall；HCP scale/逐列权重 | 已有真实 C 全顶点。CA/CAT 当前未发现可确认来源的个体 myelin 与 bias，记为资源未齐，不用 atlas 冒充个体 | NumPy CPU；独立准备不设置 CUDA |
| `prepare_t1w_surface_geometry` | white/pial/graymid 或 midthickness；tkRAS→scanner RAS；显式 fsnative→T1w affine；双侧/overwrite | 全 native 顶点；官方保存几何 + 原版 convert/apply affine 阶段；配准矩阵固定 | nibabel/NumPy CPU |
| `prepare_fmriprep_surface_inputs` | 几何、FS→fsLR 初始球面、厚度 ROI、填孔、现有 native mid；双侧/预算 | 同一已完成真实 recon-all 全网格，Workbench 准备阶段 | nibabel/NumPy/WB CPU |
| `run_fmriprep_surface_projection` | T1w cortex/MNI subcortex；ribbon、nearest dilation、ROI、ADAP_BARY_AREA；可选 goodvoxels；fsLR32k 双 GIFTI、91k CIFTI、TR、QC；双侧/overwrite | 已有完整 490 帧与公开 180 帧；固定 sphere 对官方 Workbench + NiWorkflows CIFTI；不把前序 registration 时间混入投影 | WB CPU；CIFTI nibabel/NumPy CPU，完整 saved numeric/axis 对照 |
| `create_fmriprep_cifti` | LAS MNI grid、21 structures、atlas ROI、series TR、L/R axis order、双侧读取 | 全帧匹配体积/两 GIFTI；官方 NiWorkflows pinned source | CPU；保存/读取和 axis 是功能的一部分 |
| `fMRISurface_pipeline` | signal preproc/clean；provided directory/ZIP、FNIT recon 路由；已有 registered spheres 或新 MSMSulc；可选 MSMAll native/fsLR32k composition；volume ready/missing/partial/mismatch；run entities；QC/事务保存；双侧总预算 | 空 surface 输出目录，以声明的完整 volume/recon 为起点；同时单列固定 sphere 与完整 MSMSulc。自动 raw→volume / recon 属于各自任务负责全 API，不能用 stage 缓存冒充原始 BIDS 整链 | 计算边界继承成熟子函数；默认 STC 关闭；既有第三方 reference backend 只在官方验证隔离环境 |

## 执行协议

最新候选注册六项和 features 八项的完整数值与实际源码/输入 SHA 已取回，见 [完整聚合回执](completion_status_20261005.public.json)。注册三种完整配置 CPU1/8 均保持严格 native1 双侧坐标、有序 faces 逐位一致。固定投影与完整 surface 尚无通过项：原版 command_0 在候选开始前失败，需先修复参照入口后重新验证。

1. 在 nodecw8 同一 8 个不同物理核中，预算 1 使用首核，预算 8 使用全组。候选/原版保持相同亲和性和所有 BLAS/OpenMP/Numba 上限，独占本任务锁。
   HCP MATLAB 特征在 nodecw10 同组八个独立物理核运行；nodecw8 的 MATLAB 实际启动返回许可错误，未进入原函数。两节点时间分别报告，不相除。
2. newMSM 原版左右依次运行，每侧用当前总预算。FNIT 默认预算 8 左右并行，本地查询各 4；完整进程均限制于同一 8 核。预算 1 双方串行。
3. 同线程速度和精度成对记录；原版 8 线程的非确定性另与严格 1 线程结果比较。优化前后 sphere 的 coords/ordered faces、CIFTI axes/全帧、有限性、dtype、输出取向及真实完成迭代不能只看平均相关。
4. 工具导入、完整输入读取、计算、传输、所有输出保存记入 fresh process；热 API 和分阶段另列，不相加为额外整链收益。
5. GPU 配对由协调者统一锁定后执行完整旧/新 API。当前共享 H100 繁忙，准备阶段未启动 GPU；不凭 CPU-only 分支名宣布 GPU 不退化。

## 资源和许可

newMSM 固定原代码与 FNIT 源码已有 MIT notice；独立 HOCR/FastPD 及原运算顺序的 C++ 来源见 `THIRD_PARTY_NOTICES.md` 和构建来源记录。实际原版 binary SHA `af5c04246cfeea19233232168acbc1f31266f28b1a7cb6779b8f1bfa32eb9615`，Workbench SHA `b0944851ba0da729cfd02a854c949207950e2a2f5c91e0bc222332085ae88458` 已现场校验。

HCP feature commit 为 `f8cac6892f88bdf889d644711ff038198eb81533`，BSD-style license 在 `licenses/HCPpipelines.txt`。原版 MATLAB 和 binary 不随运行时发布。HCP/TemplateFlow 资产逐文件大小和 SHA 绑定，未授权再分发资源继续仅从原站获取。UKB 个体影像和 per-vertex 数组保留私有；文档只含聚合报告及既有已公开的 HCP reference RSN 示例。
