# fMRI volume 修复与真实数据复测

2026-10-01，在 gpucw1 上用 `3b9b0f8128f10da77d3fbb6c395e9127353c3e46` 完整处理一例真实 UKB 的 490 帧 BOLD，使用当前 FLIRT、SynthMorph、ICA-AROMA、WM/CSF/24 项运动回归。运行时没有调用 FSL、FreeSurfer、AFNI 或封装软件；原 FSL 仅用于独立参照。

## 修改了什么

| 问题 | 修复 | 验证 |
|---|---|---|
| 原始设计矩阵直接做相对阈值伪逆，组织信号基线与运动单位影响数值秩 | 保留截距，去掉常数重复列，其余列中心化并按 L2 范数归一化；带通后再次处理有效列 | 真实 29 列设计旧算法 rank=23，新算法 rank=29；旋转单位、组织基线/缩放控制不再改变投影 |
| 斜位 affine 计算造成源坐标微小越界，边缘被错误置零 | 仅在源 voxel `1e-6` 容差内夹回边界；实际越界仍为零，linear/nearest/spline 共用 | CPU/CUDA identity 与正负边界测试；新 warp 的真实 8 帧 FSL 对照 |
| 不启用 WM/CSF 回归也强制构建相应原生掩膜 | 按选择构建；启用的掩膜为空时给出明确错误；保留 BBR 必需的 T1 白质分割 | 无额外组织回归、启用单项及空掩膜控制 |
| surface 接口要求额外 WM/CSF/运动回归，拒绝已经完成 AROMA 的结果 | 接受 AROMA-only；核对完成状态、BOLD/T1 来源、TR 和参考网格；保留兼容旧报告 | AROMA-only、旧 metadata、错误来源/TR、多 T1 选择测试 |
| JSON 没有足够配置供复现，TaskName 被文件标签替代 | 保留源 TaskName；记录完整配置、AROMA 完成状态、Python 源码清单及依赖版本 | 两份完整 derivative 配置与源码哈希独立核验 |
| benchmark 将原生输出与 raw BOLD 网格比较，误判不同网格的合法 SBRef | 使用实际选择的 SBRef；无 SBRef 才用 BOLD | 两项参考网格测试 |

测试记录见[机器可读结果](volume_fixed_tests.public.json)。修复保持公开单被试 Python 和 CLI 接口。`FNIT.Source` 是 Python 源码清单；权重、模板及 AROMA 掩膜的完整资源哈希由安装与 benchmark 清单记录。修复本身未增加低精度模式或依赖；合并的 GPU 配准依赖已列入主页 Conda 环境。

## 测试与整链

封存源码的 212 项目标测试通过。首次为 209 项通过、3 项因快照遗漏 `tools/benchmark_registration_gpu.py` 失败（349.02 s）；补入相同 Git 提交的文件后，3 项全部通过（0.33 s）。计算与接口未发生数值失败，原记录保留于测试 JSON。复测命令：

```bash
# 检查 fMRI、时间滤波、当前 FLIRT 和优化采样的回归测试。
pytest -q tests/test_fmri_*.py tests/test_feat_temporal.py tests/flirt \
  tests/test_registration_gpu_benchmark.py \
  tests/fast_vbm/test_flirt_fused_sampling.py \
  tests/fast_vbm/test_flirt_schedule_parity.py
```

SBRef 的两项测试包含在上述通过结果中；测试样本用于边界和接口控制，下面的 benchmark 使用真实影像。

| 整链检查 | 实测 |
|---|---|
| 原生 BOLD | 88×88×64×490，float32，242,851,840 个值全部有限 |
| MNI BOLD | 91×109×91×490，float32，442,288,210 个值全部有限 |
| 时间轴 | TR 0.735 s，NIfTI 单位 sec，与源 JSON 一致 |
| MNI 脑掩膜 | 221,058 个体素；掩膜外最大绝对值 0 |
| ICA-AROMA | 95 个成分，55 次迭代收敛，52 个噪声成分 |
| API / 验证进程墙钟 | 551.07 / 578.85 s |
| CUDA 分配 / 保留峰值 | 13.30 / 16.96 GB |
| 来源 | 公共报告 96 个源码/资源 SHA、derivative 各 77 个 Python SHA 均匹配提交及服务器封存源码 |

API 时间包含权重加载和最终写盘，排除测量脚本的预先导入、预先哈希及 CUDA 上下文初始化及事后检查。额外捕获中间文件的 1.798 s 已从 API/total 及相应外层阶段计时扣除；解剖复制发生在内部计时停止后，只从 API/total 扣除。共享 H100 上的一次冷运行不代表稳定加速比。完整阶段及配置见[整链报告](fmri_volume.public.json)，独立复读见[合同检查](volume_fixed_contract.public.json)。

## 混杂回归的独立参照

固定本次实际 AROMA 输出、组织掩膜及运动参数，以独立 NumPy float64 SVD 正交投影计算参考，比较脑掩膜内全部 97,345×490 个值。设计为三项二次趋势、WM/CSF 均值与 24 项运动，共 29 列。

| 控制 | 结果 |
|---|---:|
| 旧算法 / 修复算法，有效秩 | 23 / 29 |
| 实际输出 vs 独立参考，MAE / RMSE | 3.395e-6 / 5.237e-6 |
| 实际输出 vs float32 参考，最大绝对差 | 3.052e-5 |
| 同一真实输入旧 / 新回归，pooled r / RMSE | 0.991434 / 27.2643 |
| 修复后 rad / degree 投影，RMSE | 1.045e-11 |
| 修复后组织基线与缩放控制，RMSE | 3.309e-10 |

单位与基线控制固定同一 AROMA 数据，只改变设计矩阵的表达方式，没有另跑整套流程。真实 WM 掩膜 25,995 个体素，CSF 11,947 个体素，与原生参考同网格且位于脑掩膜内。定义与文件 SHA 见[标量报告](volume_fixed_confounds.public.json)。

```bash
# 对一个已经捕获中间数据的完整运行，核对源代码并计算独立回归参照。
python validation/fmri/check_real_confounds.py \
  --run-root /private/results/current_volume \
  --native-clean /private/results/current_volume/derivatives/sub-benchmark/func/sub-benchmark_task-rest_space-boldref_desc-clean_bold.nii.gz \
  --source-revision CURRENT_COMMIT \
  --report-out /private/results/confounds.public.json
```

脚本要求运行目录包含 `intermediates/feat`、`intermediates/aroma`、`intermediates/masks`、完整 `derivatives`、`fmri_volume.public.json` 及封存 `source/src`；默认不导出影像。可通过[测量脚本](benchmark_bids.py)的 `--capture-intermediates` 和 `--capture-resampling-inputs` 保存私有控制输入，两个选项只用于验证。

## 插值与配准

当前真实 native 清理图的首 8 帧、当前合成 EPI→MNI 场、同一输出掩膜，分别由 FNIT 和 FSL `applywarp --rel --interp=spline` 采样。脑内 r=0.99999999993，MAE=0.001461、RMSE=0.002063、最大绝对差 0.054962。原 FSL 返回 255，但写出的完整输出通过 gzip CRC、网格、TR、有限值、dtype 和脑外置零检查；原退出码保留。见[本次插值报告](volume_fixed_resampling.public.json)，用[独立参照脚本](compare_fsl_resampling.py)复测。

新旧 pre-ICA 时间 r 中位数为 0.999386，最终 MNI 为 0.943486。新旧 MNI→T1 采样位置变化中位数 0.011 mm，合成 MNI→EPI 为 0.115 mm，p95 为 0.202 mm。配准阶段合计从 1186.44 s 降到 16.94 s，API 从 1719.19 s 降到 551.07 s；共享负载不同，不能归因成单一优化的受控加速比。

官方发布数据的 FIX、GDC/B0 和配准与本流程不同。完整 MNI 时间 r 均值 0.271534 与上一版 0.273070 接近；当前使用 ICA-AROMA，不以 FIX 等价作为验收目标。固定同一个 warp 或同一个官方清理图的控制见[新旧与官方报告](volume_fixed_comparison.public.json)和[全流程表格](README.md)。同场插值的一致性不意味着配准估计等价。

复测新旧完整结果需传入私有运行目录、官方文件及模板：

```bash
# 固定一例数据，对比两套结果，并将同一官方原生清理图用新 warp 再采样。
python validation/fmri/compare_volume_revision.py \
  --current-root /private/results/current_volume \
  --previous-root /private/results/previous_volume \
  --current-revision CURRENT_COMMIT --previous-revision PREVIOUS_COMMIT \
  --official-native /private/official/clean_native.nii.gz \
  --official-mask /private/official/native_mask.nii.gz \
  --official-mni /private/official/clean_mni.nii.gz \
  --official-pre-ica /private/official/no_gdc_no_b0/filtered_func_data.nii.gz \
  --official-pre-mask /private/official/no_gdc_no_b0/mask.nii.gz \
  --template /templates/MNI152_T1_2mm.nii.gz \
  --private-output /private/results/spatial_control \
  --report-out /private/results/comparison.public.json --device cuda:0
```

脚本按本次匿名 `sub-benchmark` 单 run 布局读取结果，发现多个候选时报错；旧目录须提供 `feat_saved`，新目录须提供 `intermediates/feat`，两者均需 `resampling_inputs`。它是单例测量脚本，不是公开被试调度接口。逐体素时间相关在共同掩膜内计算，每项表格只使用各输入都有变化的体素。原始 MAE/RMSE保留强度与均值差，不能与去均值的时间 r 互换解释。

示例图、运行记录和匿名哈希已更新；原始影像、逐体素结果及日志留在服务器。surface 与 MS-HBM 的既有测量仍注明各自日期和输入，本次没有重跑它们。
