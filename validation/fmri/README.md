# fMRI volume 与 surface 全流程 benchmark

[volume 用法](../../docs/fmri/README.md) · [surface 用法](../../docs/fmri/surface.md) · [单被试测量脚本](benchmark_bids.py) · [FNIT / DeepPrep 实测对照](deepprep/README.md)

2026-10-01 使用 `a7c5a64`，重新测量当时的 FLIRT 和修复后的完整 volume。surface 的完整测量来自 2026-09-30 的 `3f8b756`，输入是该次生成的 volume；两次结果分别报告，不合并为新的 volume→surface 总时间。

另一次 `8dbeea64` 更新的 BBR/FNIRT 修复、独立冷/热调用、CUDA profile 与解剖缓存复用见[当前配准报告](registration_gpu.current.public.json)、[BBR 功能页](../../docs/fmri/bbr.md#真实-ukb-数据对照)和[FNIRT 功能页](../../docs/fnirt/README.md#真实数据验证)。它们按各自源码哈希报告独立 registration/解剖步骤；491.36 s 仍是 `a7c5a64` 的完整 volume，不是合并后源码的整链速度。合并后的完整 volume 尚待重测，本次也未重跑 surface。

两次均使用一例真实 UKB：BOLD 88×88×64×490，TR 0.735 s，同次 SBRef；T1 取自匹配重建存档的 `orig/001.mgz`，经 nibabel 逐体素无误差转换，更早的结构预处理未核对。surface 使用既有皮层几何，不计 recon-all。权重和模板资源的许可与来源见[资源记录](assets_input_preflight.public.json)，本次权重再次校验的大小及 SHA-256 见[权重检查](volume_fixed_weights.public.json)。

## 完整 API 实测

| 项目 | volume：2026-10-01 | surface：2026-09-30 |
|---|---|---|
| 运行提交 | `a7c5a64` | `3f8b756` |
| 起点 | 原始 BOLD/SBRef、存档 T1 输入 | 9 月 30 日 volume 派生文件、匹配的既有 T1 表面 |
| 处理 | SynthStrip、FEAT、FAST、BBR、SynthMorph、PICA/AROMA、WM/CSF/24 项运动回归、MNI 重采样 | EPI→T1w、表面准备、MSMSulc、ribbon 投影、fsLR32k、皮层下组装 |
| 输出 | 原生 88×88×64×490；MNI 91×109×91×490 | 每侧 490×32,492；CIFTI 490×91,282 |
| API 墙钟，含最终写盘 | **491.36 s** | **905.71 s** |
| 验证进程墙钟 | 523.62 s | 912.06 s |
| CUDA allocated / reserved，十进制 GB | 13.34 / 16.96 | 0.73 / 1.26 |
| 检查 | float32、全部有限、TR/网格一致、MNI 掩膜外为零 | GIFTI/CIFTI 有限、JSON 齐全、TR 一致、90,572 个非常数灰质坐标 |
| 报告 | [当前 volume](fmri_volume.public.json) | [surface](fmri_surface.public.json) |

共享 H100、8 个 CPU 线程、float32/TF32，各为一次冷调用。API 计时排除导入、预先哈希、事后检查及 CUDA 上下文初始化，包含首次权重加载与最终保存。volume 验证为捕获中间文件增加 2.428 s 复制开销；公开 API 和阶段时间均已扣除，完整验证进程包含这些开销。原始 derivative 的阶段计时保留包装器开销，区别见[独立合同检查](volume_fixed_contract.public.json)。FNIT volume 运算不调用 FSL、FreeSurfer、fMRIPrep；surface 投影和组装使用 Workbench，球面配准使用 FNIT HOCR/FastPD。

| 当前 volume 阶段 | 秒 | 上一版同病例，秒 |
|---|---:|---:|
| SynthStrip | 8.85 | 13.05 |
| FEAT 核心 | 250.29 | 298.92 |
| TorchFAST | 1.48 | 3.30 |
| BBR 与 T1→MNI | 32.22 | 1186.44 |
| 混杂掩膜准备 | 0.20 | 0.21 |
| PICA、AROMA 与混杂回归 | 137.08 | 163.35 |
| MNI 重采样 | 55.82 | 48.44 |
| API 总时间 | 491.36 | 1719.19 |

当前配准显著缩短了本例的实测时间；共享 GPU 负载和冷启动边界影响耗时，不能由两次单例实验给出稳定加速比。当前源代码、输出及派生 metadata 的哈希已独立核对。新旧 pre-ICA 文件 SHA 完全相同；ICA 为 95 个成分、115 次迭代收敛、55 个 AROMA 噪声成分。

## 修复与最终图比较

修复了混杂项量纲导致的秩截断、浮点边界置零、可选组织掩膜构建和 surface 对 AROMA-only 输入的校验。单被试 Python/CLI 保持原接口，完整配置和去噪完成状态写入 JSON。173 项 CPU/CUDA 测试通过，SBRef benchmark 参考网格另有 2 项测试；真实 490 帧混杂回归与独立 float64 参照 RMSE `5.080e-6`。细节和复测命令见[修复验证](volume_fixed.md)。

| 控制，逐体素 490 帧时间 Pearson r | 均值 | 中位数 |
|---|---:|---:|
| 新 / 旧 FNIT 原生清理图 | 0.992658 | 0.993736 |
| 新 / 旧 FNIT MNI 清理图 | 0.990485 | 0.991469 |
| 新 FNIT / 官方 UKB 原生图 | 0.428608 | 0.502603 |
| 新 FNIT / 官方 UKB 完整 MNI 图 | 0.271106 | 0.228182 |
| 两侧清理图，固定同一 FNIT warp | 0.473946 | 0.564422 |
| 同一官方清理图，FNIT / 官方 warp | 0.486080 | 0.523288 |

原生表使用 96,011 个共同非常数体素；MNI 表使用新旧输出掩膜交集中的 220,977 个共同非常数体素。UKB 官方使用 FIX、GDC/B0 与官方配准，本流程选择 ICA-AROMA，候选没有 GDC/B0。表格报告处理差异；不以更接近 FIX 的 r 作为 AROMA 的质量标准。新旧 MNI→EPI 采样位置变化中位数 0.083 mm、p95 0.162 mm，整链相关性仍与旧版接近。所有输入/输出及 warp 哈希见[比较报告](volume_fixed_comparison.public.json)。

当前 warp 的首 8 个真实帧与原 FSL `applywarp --interp=spline` 同场比较，脑内 r=0.99999999993、RMSE=0.001956。FSL 退出 255，但输出完整 CRC、网格、有限值、TR、掩膜检查通过，原退出码保留于[报告](volume_fixed_resampling.public.json)。这验证插值，不验证 warp 估计。此前固定 490 帧同 warp 的插值和 temporal SD 格纹控制见[专门验证页](resampling.md)。

## 当前 FEAT 与 FSL 的同输入精度

候选取自上面这次完整 volume 的 pre-ICA 输出，而非旧缓存。参照是同一原始 BOLD/SBRef 的 FSL 6.0.7.22 逐条原版程序输出。两侧均未估计或应用 GDC/B0 校正。比较掩膜是两个最终 EPI 掩膜的交集；4D r 保留时间均值，时间 r 在每个体素的 490 帧内计算。

| 指标 | 结果 |
|---|---:|
| FNIT / FSL / 交集掩膜体素 | 97,347 / 113,881 / 96,776 |
| 掩膜 Dice | 0.916318 |
| filtered 4D Pearson r | 0.996417 |
| MAE / RMSE，归一化强度单位 | 423.501 / 556.959 |
| 去掉逐体素时间均值后的 pooled r | 0.944905 |
| 逐体素时间 r 中位数 / 均值 | 0.966798 / 0.951027 |

shape、affine 和全体素有限值检查通过。[当前标量](feat_current.public.json)和[独立比较脚本](compare_feat.py)记录完整定义。该参照保留 MCFLIRT `-spline_final`、BET/阈值掩膜、grand-mean 缩放及 100 s 高通；原 FSF 没有 slice timing、空间平滑或低通。FNIT 使用 SynthStrip 掩膜和自身运动求解器，两个步骤存在差异。

原 `feat` 启动器在该环境返回 255，正式参照按 `featlib.tcl` 逐条执行 MCFLIRT、BET、fslstats 和 fslmaths，并检查输出完整性。[参照脚本](official_feat_no_gdc.sh)保留命令；独立验证环境才需要 FSL。既有 FSL 计时中 MCFLIRT 为 397.54 s，其余已计时影像命令为 428.12 s，两次 fslstats 未单独计时；825.66 s 是分步耗时下界，不是完整 FSL/UKB fMRI 总时间。不能把它与 491.36 s 的完整 FNIT volume 时间相除。

最终去噪使用 ICA-AROMA 加可选混杂回归。与真实 UKB FIX 发布文件的最终图差异见上表；FEAT 同输入参照只验证前处理部分。

## 固定 volume 的 surface 球面对照

该实验测于 2026-09-30，用 `3f8b756` 生成的 clean BOLD、同一 T1 皮层几何和 HCP 资源，再运行一次 surface API；只把 FNIT 估计的球面替换为既有官方 newMSM 球面。参照球面来自 HCP v4.7.0 MSMSulc 配置，首层 `simval=1`，其余为 2，8 线程。这样可定位球面对应关系造成的差异；投影和 CIFTI 组装仍是相同 FNIT/Workbench 路径，未独立重跑完整 fMRIPrep。

| 结构 | 灰质坐标数 / 有效时间 r 数 | 时间 r 均值 | 中位数 | 第 5 百分位 | MAE | 最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| 左皮层 | 29,696 / 29,695 | 0.940496 | 0.971542 | 0.770144 | 18.1565 | 868.3127 |
| 右皮层 | 29,716 / 29,689 | 0.941404 | 0.968101 | 0.792942 | 19.7100 | 636.2633 |
| 皮层下 | 31,870 / 31,188 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |

两份 CIFTI 的时间轴和 BrainModel 轴完全相同；常数时序不计算 r。皮层仍有明显差异，皮层下逐值相同，符合本对照仅改变皮层球面的边界。固定官方球面的 API 为 697.45 s，验证进程为 703.76 s；它跳过球面估计，不能当作官方完整 surface 耗时。见[数值报告](fmri_surface_comparison.public.json)、[官方球面控制运行](fmri_surface_official_spheres.public.json)与[比较脚本](compare_cifti.py)。

## 示例图

图的三行依次是 MNI 解剖模板、FNIT 清理后 BOLD 时间标准差、一个清理后的时间点。回归去掉截距后时间均值接近零；模板用于定位，不是官方清理后 BOLD。每行共用色阶，切面为同一模板网格的中间位置。

![当前 490 帧 volume 示例](../../docs/fmri/figures/fmri_volume.png)

下面的 2026-09-30 surface 对照图把逐顶点时间 r 映射到公开标准 fsLR32k 球面，展示 +x 半球视图；右侧直方图包含双侧全部有效皮层坐标。灰色为内侧壁或常数信号，球面不是被试解剖表面。

![2026-09-30 surface 球面对照](../../docs/fmri/figures/fmri_surface_agreement.png)

## 单被试如何复测

先安装主页 Conda 环境、部署权重，并准备实际 BIDS、MNI 模板、HCP 表面资源和匹配的重建存档。以下命令先生成完整 volume，再让 surface 读取刚生成的派生数据；每次使用新输出目录。

```bash
python validation/fmri/benchmark_bids.py volume \
  --bids-root /data/bids --derivatives-root /results/fnit \
  --subject 0001 --mni-template /templates/MNI152_T1_2mm.nii.gz \
  --mni-brain-mask /templates/MNI152_T1_2mm_brain_mask.nii.gz \
  --synthstrip-weights /models/synthstrip.1.pt \
  --synthmorph-weights /models/synthmorph.deform.3.h5 \
  --device cuda:0 --threads 8 \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --source-revision YOUR_COMMIT \
  --report-out /results/volume.private.json

python validation/fmri/benchmark_bids.py surface \
  --bids-root /data/bids --derivatives-root /results/fnit --subject 0001 \
  --recon-all /data/matching-reconstruction \
  --hcp-assets-dir /templates/hcp --wb-command wb_command \
  --device cuda:0 --threads 8 \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --source-revision YOUR_COMMIT \
  --report-out /results/surface.private.json
```

`YOUR_COMMIT` 填写实际运行代码的 Git 提交号。脚本调用公开单被试 Python API，记录 API 时间、GPU 峰值和输出检查；不会调度其他被试。volume 示例启用三类混杂回归、默认 SynthMorph；`--registration-backend fnirt` 选择 T1 FNIRT 配置，本页新的完整链计时只覆盖 SynthMorph。surface 示例估计 FNIT 球面；`--registered-spheres LEFT RIGHT` 是固定球面控制，跳过 MSMSulc 估计，不应当作默认 surface 时间。

两份同轴 CIFTI 可用 `compare_cifti.py --candidate ... --reference ... --report-out ...` 比较；`render_surface.py` 接收这两份文件和公开标准双侧 fsLR32k 球面，生成上图。

新旧与官方时间标准差图见[volume 用法页](../../docs/fmri/README.md)，由 `render_volume_comparison.py` 在同一网格、同一色阶生成。volume 图示可用 `render_volume.py --bold ... --mask ... --template ... --figure-out ...` 生成。FEAT 中间结果只在公开 volume API 的临时目录存活；`compare_feat.py` 接收事先保存的候选与官方 FEAT 目录，不把最终清理图误当成 pre-ICA 图。官方和 FNIT 都应使用同一 raw BOLD/SBRef 和校正配置。

## 其他阶段参照

[BBR/T1 FNIRT 独立配对](registration_gpu.current.public.json)、[PICA](pica_summary.json)、[固定运动矩阵的插值](motion_spline_summary.json)与[MCFLIRT 求解差异](mcflirt_difference.public.json)采用各自注明的固定输入和参数，不能替代 `a7c5a64` 完整 volume 的最终图比较。surface 的 2026-09-30 测量仍对应该次输入；本次未重跑 surface 或 MS-HBM。

## 独立 BBR/FNIRT 与解剖缓存测量

以下为 `8dbeea64` 更新的真实单病例测量，与上面 `a7c5a64` 的完整 SynthMorph volume 分开。函数钟包含输入解压与 CPU 结果转换，排除最终写盘和事后精度计算。首次调用前已初始化 CUDA，未清空 Triton 磁盘缓存。完整定义与源码哈希见[独立配准报告](registration_gpu.current.public.json)。

| 同输入范围 | 首次 / 热调用 | 对官方参照的结果 |
|---|---:|---|
| BBR，固定 FSL WM/init | 3.581 / 1.409 s | 影像 r 0.99999970；逆变换位移 RMS 0.00260 mm。 |
| FNIT FAST＋FLIRT＋BBR | 6.932 / 5.072 s | 影像 r 0.9997382；逆变换位移 RMS 0.07203 mm。 |
| T1 FNIRT，固定 FSL affine/模板掩膜 | 32.595 / 30.422 s | warped T1 r 0.99771788；完整 pull 位移中位数 / p95 0.05176 / 0.23294 mm。 |
| 解剖准备，FNIT FLIRT＋FNIRT；第二次命中缓存 | 41.118 / 0.0785 s | 输出产物 SHA-256 相同；第二次仅核验并复用。 |

默认 `reuse_anatomical=True` 只复用同一 T1、模板、权重、配置、实现和环境的完整解剖产物，逐 run 的 BBR/EPI/BOLD 步骤仍独立计算。`reuse_anatomical=False` 或 CLI `--no-anatomical-cache` 可强制重算；细节见[volume 输出与缓存说明](../../docs/fmri/README.md#输出)。当前缓存测试使用 FNIRT，不代表 SynthMorph volume 整链重复运行。单函数计时不能替换完整 API 时间；当前 BBR/FNIRT 精度仍保留上表所示的官方差值。

## 配准冷/热调用与 profile 复测

[`tools/benchmark_registration_gpu.py`](../../tools/benchmark_registration_gpu.py)读取服务器本地的 JSON 输入清单，不启动 FSL。官方参照需先按 [BBR](../../docs/fmri/bbr.md#官方同输入命令)或 [FNIRT](../../docs/fnirt/README.md#python-与命令行t1w-专用预设)命令生成。影像、矩阵、warp 和含私有路径的清单留在本地；`report.safe.json` 只含汇总指标、环境和实现源码哈希。

BBR 的清单字段为 `epi`、`t1`、`wmseg`、`init`、`official_matrix`、`official_moved`，值均为相应文件的绝对路径；`init` 必须是 normmi 初始矩阵，不能填官方最终 BBR 矩阵。FNIRT 使用 `moving`、`reference`、`reference_mask`、`affine`、`official_warped`、`official_coeff`；`affine` 是 input→reference 的 FSL scaled-mm 初始矩阵。可再提供 `official_jacobian`（官方 `jout`）及 `official_pull_x/y/z`（官方 `applywarp` 重采样的 input RAS world 坐标图）。

```bash
# --source-root：待测试源码，独立 before/after 快照各执行一次。
# --case-json：含本地真实输入与同输入官方参照的清单。
# --output-dir：新的私有结果目录；计时结束后才保存图像和矩阵。
# --function：bbr 固定 WM/init；bbr_chain 自行运行 FAST 和 FLIRT；fnirt 使用 T1 六级预设。
# --warm-repeats：首次调用后重复次数；首次所需的 JIT 编译/加载计入该调用。
python tools/benchmark_registration_gpu.py \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --case-json /private/cases/bbr.json --output-dir /private/results/bbr-timing \
  --function bbr --bbr-execution batched --warm-repeats 1

python tools/benchmark_registration_gpu.py \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --case-json /private/cases/fnirt.json --output-dir /private/results/fnirt-timing \
  --function fnirt --affine-geometry header_pixdim \
  --fnirt-execution optimized --warm-repeats 1

# profile 与性能计时分开运行；profile 的插桩开销不进入速度表。
python tools/benchmark_registration_gpu.py \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --case-json /private/cases/fnirt.json --output-dir /private/results/fnirt-profile \
  --function fnirt --affine-geometry header_pixdim \
  --fnirt-execution optimized --profile-only --profile-full
```

`--bbr-execution reference`、`--fnirt-execution reference` 对照同一修正算法的原执行路径。完整链用 `--function bbr_chain --wm-header reference` 保留 WM 的 T1 header。`--affine-geometry affine_norm`、`--wm-header legacy` 用于复测明确注明的旧快照，不能与新 header 契约混用。`--fnirt-blur-reference`、`--fnirt-bending-reference` 仅用于定位单个执行改动，不是降低搜索或迭代的 fast mode。

“冷”指该进程首次函数调用，“热”指随后调用；未清空 Triton 磁盘缓存，CUDA 上下文初始化在函数计时前。墙钟包含函数内 ArrayProxy 读入/解压与 CPU 结果转换，排除输出写盘和事后精度计算。profile 汇总 kernel、H2D/D2H、CUDA 同步 API 与成本求值次数；嵌套阶段钟是无额外 GPU fence 的 CPU wall，不能把它们都相加。`nvidia-smi` 利用率属于整张共享 GPU，不能当作本进程利用率；kernel 累计时间也不等于独占 wall。公开前应再次核对本地报告载荷。

经数据持有者确认发布权限，仅公开匿名标量、代码及文件哈希和去标识化的 PNG。原始 NIfTI、MGZ、皮层几何、逐体素时序及含私有路径的日志不进入仓库。
