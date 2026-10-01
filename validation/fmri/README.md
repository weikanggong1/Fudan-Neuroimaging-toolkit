# fMRI volume 与 surface 全流程 benchmark

[volume 用法](../../docs/fmri/README.md) · [surface 用法](../../docs/fmri/surface.md) · [单被试测量脚本](benchmark_bids.py) · [FNIT / DeepPrep 实测对照](deepprep/README.md)

2026-10-01，用修复后 `ec57972` 从一例真实原始 BOLD/SBRef 和存档 T1，完整运行 FNIT FNIRT 分支与原 SynthStrip/FSL/ICA-AROMA 的相同步骤。两个流程均处理全部 490 帧，开启 WM、CSF 与 Friston-24 联合回归，不使用 GDC/B0、FIX或空间平滑。T1 为匹配重建存档的无误差 NIfTI 转换，未核对它的更早结构处理。

## 当前完整 volume 对照

| 项目 | FNIT | 原软件参照 |
|---|---:|---:|
| 封存运行提交 / 驱动 | `ec57972` | `380e23d` |
| 原生 / MNI 网格 | 88×88×64×490 / 91×109×91×490 | 相同 |
| API / 连续链墙钟 | **455.62 s** | **2570.47 s** |
| 独立验证进程墙钟 | 530.35 s | 2601.72 s |
| CUDA allocated / reserved，GB | 6.239 / 7.317 | 未单独记录 |
| 运动校正时间 r 均值 / 中位数 | 0.941467 / 0.964133 | 比较基准 |
| pre-ICA 时间 r 均值 / 中位数 | 0.951153 / 0.966887 | 比较基准 |
| 原生 clean 时间 r 均值 / 中位数 | **0.850471 / 0.871444** | 比较基准 |
| MNI clean 时间 r 均值 / 中位数 | **0.853589 / 0.872108** | 比较基准 |

共享 H100、8 线程，FNIT float32/TF32；原 SynthStrip 用 GPU，原 FSL 用 CPU。完整运行均使用新目录、不复用解剖缓存。FNIT API 扣除捕获中间影像的额外复制，原参照连续链包含阶段内验证，两边计时边界不同，不以单次值推导稳定加速比。输出检查涵盖全部体素有限值、网格、TR、CRC与掩膜外置零。

[完整报告与复测步骤](matched_native.md)列出每阶段参数、软件版本、源码/输入哈希、组织预滤波修复、ICA/回归设计和交叉 warp 控制；[机器可读比较](matched_pipeline.public.json)、[FNIT 输出合同](matched_fnirt_contract.public.json)及[原连续链时间](matched_native_pipeline.public.json)保存实际结果。固定同一图像、仅更换 warp 的时间 r 均值为 0.994511；固定原场后，FNIT sampler 对原 FSL applywarp 的均值为 0.999999999921，RMSE 0.002215。主要剩余差异在运动估计、ICA 与回归设计。

![相同步骤 FNIT 与原软件完整490帧结果](../../docs/fmri/figures/fmri_matched_native.png)

## 其他处理协议与 surface 范围

先前 `3b9b0f8` 的 SynthMorph/AROMA volume 为 551.07 s，尚未包含本轮组织预滤波修复。它与 UKB FIX/GDC/B0 发布图的 MNI 时间 r 均值为 0.271534；输入清理与配准协议不同，不能和上面的同步骤对照混用。旧协议的完整配置、测试及插值控制见[该次修复验证](volume_fixed.md)、[发布图比较](volume_fixed_comparison.public.json)和[时间 SD 格纹控制](resampling.md)。

surface 的完整测量来自 2026-09-30 `3f8b756`，起点是该次 volume 派生文件及已有皮层几何，包含 EPI→T1、MSMSulc、ribbon 投影、fsLR32k 与皮层下组装，不计 recon-all。API 905.71 s、验证进程 912.06 s、CUDA allocated/reserved 为 0.73/1.26 GB；每侧 490×32,492，CIFTI 490×91,282，全部有限、时间轴和 BrainModel 轴正确。本轮未重跑 surface 或 MS-HBM；该时间不与当前 volume 相加。见[surface 报告](fmri_surface.public.json)。

## 固定 volume 的 surface 球面对照

该实验测于 2026-09-30，用 `3f8b756` 生成的 clean BOLD、同一 T1 皮层几何和 HCP 资源，再运行一次 surface API；只把 FNIT 估计的球面替换为既有官方 newMSM 球面。参照球面来自 HCP v4.7.0 MSMSulc 配置，首层 `simval=1`，其余为 2，8 线程。这样可定位球面对应关系造成的差异；投影和 CIFTI 组装仍是相同 FNIT/Workbench 路径，未独立重跑完整 fMRIPrep。

| 结构 | 灰质坐标数 / 有效时间 r 数 | 时间 r 均值 | 中位数 | 第 5 百分位 | MAE | 最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| 左皮层 | 29,696 / 29,695 | 0.940496 | 0.971542 | 0.770144 | 18.1565 | 868.3127 |
| 右皮层 | 29,716 / 29,689 | 0.941404 | 0.968101 | 0.792942 | 19.7100 | 636.2633 |
| 皮层下 | 31,870 / 31,188 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |

两份 CIFTI 的时间轴和 BrainModel 轴完全相同；常数时序不计算 r。皮层仍有明显差异，皮层下逐值相同，符合本对照仅改变皮层球面的边界。固定官方球面的 API 为 697.45 s，验证进程为 703.76 s；它跳过球面估计，不能当作官方完整 surface 耗时。见[数值报告](fmri_surface_comparison.public.json)、[官方球面控制运行](fmri_surface_official_spheres.public.json)与[比较脚本](compare_cifti.py)。

## 示例图

下面的图来自先前 `3b9b0f8` SynthMorph 分支，未包含本轮组织预滤波修复。三行依次是 MNI 解剖模板、FNIT 清理后 BOLD 时间标准差、一个清理后的时间点。回归移除时间均值后信号基线接近零；模板用于定位，不是官方清理后 BOLD。每行共用色阶，切面为同一模板网格的中间位置。

![先前3b9b0f8 SynthMorph分支490帧volume示例](../../docs/fmri/figures/fmri_volume.png)

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
  --registration-backend fnirt \
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

`YOUR_COMMIT` 填写实际运行代码的 Git 提交号。脚本调用公开单被试 Python API，记录 API 时间、GPU 峰值和输出检查；不会调度其他被试。volume 示例启用三类混杂回归，并用 `--registration-backend fnirt` 选择与本次主对照一致的 T1 FNIRT 分支。测量脚本只运行 FNIT；原软件全流程复测使用 [matched_native.md](matched_native.md) 的独立驱动。surface 示例估计 FNIT 球面；`--registered-spheres LEFT RIGHT` 是固定球面控制，跳过 MSMSulc 估计，不应当作默认 surface 时间。

两份同轴 CIFTI 可用 `compare_cifti.py --candidate ... --reference ... --report-out ...` 比较；`render_surface.py` 接收这两份文件和公开标准双侧 fsLR32k 球面，生成上图。

本次 FNIT 与原软件的时间标准差图见[volume 用法页](../../docs/fmri/README.md)，由 `render_volume_comparison.py` 在同一网格、同一色阶生成。volume 图示可用 `render_volume.py --bold ... --mask ... --template ... --figure-out ...` 生成。FEAT 中间结果只在公开 volume API 的临时目录存活；`compare_feat.py` 接收事先保存的候选与官方 FEAT 目录，不把最终清理图误当成 pre-ICA 图。官方和 FNIT 都应使用同一 raw BOLD/SBRef 和校正配置。

## 其他阶段参照

[BBR/T1 FNIRT 独立配对](registration_gpu.current.public.json)、[PICA](pica_summary.json)、[固定运动矩阵的插值](motion_spline_summary.json)与[MCFLIRT 求解差异](mcflirt_difference.public.json)采用各自注明的固定输入和参数，不能替代上面的 `ec57972` 完整 FNIRT volume 最终图比较。surface 的 2026-09-30 测量仍对应该次输入；本次未重跑 surface 或 MS-HBM。

## 独立 BBR/FNIRT 与解剖缓存测量

以下为 `8dbeea64` 更新的真实单病例测量，与上面的完整 `ec57972` FNIRT volume 分开。函数钟包含输入解压与 CPU 结果转换，排除最终写盘和事后精度计算。首次调用前已初始化 CUDA，未清空 Triton 磁盘缓存。完整定义与源码哈希见[独立配准报告](registration_gpu.current.public.json)。

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
