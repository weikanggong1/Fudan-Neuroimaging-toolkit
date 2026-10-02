# 同网格九张真实参数图的传播验收

本轮只复用固定 warp/affine 的坐标准备，每张图仍以原来的通道数独立插值。验收相对冻结的 FNIT `954ad19` sampler；官方精度按原功能页另行判读。真实 benchmark 没有用模拟图像代替。

`benchmark_map_propagation.py` 读取真实完整输出目录 `native/` 的 `dti_FA/MD/L1/L2/L3/MO.nii.gz` 和 `NODDI_ICVF/OD/ISOVF.nii.gz`。TBSS 的 FA 先运行原 `preprocess_fa`，其余图保持 native 输入；随后用冻结 coefficient 或 MMORF warp 比较原逐图调用与 prepared-plan 逐图调用。

```bash
# 这些变量由验证者填写为服务器上已完成的真实输出及原模板。
native_parameter_maps=/path/to/completed/outputs/native
standard_fa_template=/path/to/FMRIB58_FA_1mm.nii.gz
standard_fa_skeleton=/path/to/FMRIB58_FA-skeleton_1mm.nii.gz
fnirt_coefficients=/path/to/completed/outputs/registration/dti_FA_to_MNI_warp.nii.gz
frozen_applywarp_source=/path/to/frozen_954ad/src/fnit/applywarp/core.py
anonymous_tbss_report=/path/to/reports/tbss_nine_map_plan.real.json

PYTHONPATH=src python validation/dmri_pipeline/benchmark_map_propagation.py \
  --backend tbss --native-dir "$native_parameter_maps" \
  --reference "$standard_fa_template" --fa-skeleton "$standard_fa_skeleton" \
  --warp "$fnirt_coefficients" --legacy-module "$frozen_applywarp_source" \
  --device cuda:0 --repeats 3 --threads 8 --memory-limit-bytes 20000000000 \
  --output "$anonymous_tbss_report"

standard_t1_template=/path/to/MNI152_T1_1mm_brain.nii.gz
shared_mmorf_warp=/path/to/completed/outputs/registration/mmorf_warp.nii.gz
native_to_standard_affine=/path/to/completed/outputs/registration/dti_FA_to_MNI_affine.mat
frozen_mmorf_source=/path/to/frozen_954ad/src/fnit/mmorf/core.py
anonymous_mmorf_report=/path/to/reports/mmorf_nine_map_plan.real.json

PYTHONPATH=src python validation/dmri_pipeline/benchmark_map_propagation.py \
  --backend mmorf --native-dir "$native_parameter_maps" \
  --reference "$standard_t1_template" --warp "$shared_mmorf_warp" \
  --affine "$native_to_standard_affine" --legacy-module "$frozen_mmorf_source" \
  --device cuda:0 --repeats 3 --threads 8 --memory-limit-bytes 20000000000 \
  --output "$anonymous_mmorf_report"
```

| 参数 | 输入/输出及范围 |
|---|---|
| `--backend` | `tbss` 或 `mmorf`，选择冻结 sampler 与现有 warp 类型。 |
| `--native-dir` | 真实 native/ 九图目录；没有原图时不能运行此 benchmark。 |
| `--reference` | 原模板，定义相同输出坐标和 header；TBSS 还使用其非零支持区。 |
| `--warp` | 已估计的固定 coefficient 或 MMORF 三通道 reference-axis mm 场。 |
| `--affine` | MMORF 的 FSL tensor/FA→reference 4×4矩阵文件；TBSS coefficient 已含 affine。 |
| `--fa-skeleton` | TBSS 原 skeleton 模板；用于原阈值2000及九图 mask gate。 |
| `--legacy-module` | 冻结 `954ad19` 的 applywarp/core.py 或 mmorf/core.py；必须区别于当前 sampler 文件。 |
| `--device` | 默认 `cuda:0`；也可CPU。数据类型保持原精度。 |
| `--repeats` | 默认3，至少1；先后顺序交替，首轮与热调用分别记录。 |
| `--threads` | 默认8；设置 PyTorch CPU 线程数，与本次真实计时一致。 |
| `--memory-limit-bytes` | 默认20,000,000,000；CUDA 时按全卡容量设置 PyTorch 分配器限额，单位为 bytes。CPU 不设置 CUDA 限额。 |
| `--output` | 尚不存在的匿名JSON；保存hash、逐值gate、wall与CUDA内存，不保存图像/矩阵/输入路径。 |

计时前将原 NIfTI ArrayProxy 解码到内存，保留原 header/affine，TBSS FA预处理也在计时外。clock显式同步CUDA，覆盖坐标准备、九图独立采样、输出回传与有效掩膜；不包括图像读盘、gzip保存、重新估计配准。`prepared_prepare_seconds` 是同一个总wall中的子阶段，不能与总时间相加。allocator peak和全卡free bytes单独记录；全卡空闲显存包含其他作业的影响。

逐图 gate 比较原始解码voxel、完整header（含extensions）、affine、shape、dtype和最大绝对差。TBSS另比较valid_mask、FA有效mask、skeleton mask以及处理后的standard/skeleton九图。进程返回0须所有gate通过，任何差异返回1并保留报告。不要把输出相关性接近1当作无损通过。

原FA模板是TBSS最终掩膜验收的必要输入。同几何的已有standard/FA图可作采样坐标/header的临时参考，但其非零支持区与原模板不同，此时只可宣称固定采样检查通过，须注明surrogate-reference。MMORF传播不读取reference强度，可用固定同几何参考做采样对照；完整pipeline仍使用原模板。

## 2026-10-02 H100 真实九图结果

TBSS 和 MMORF 各自使用已完成真实病例的九张 native 参数图及固定形变。参考网格为 `182×218×182`；使用原 FA/T1 模板，TBSS 还使用原 FA skeleton 模板。PyTorch CPU 线程数为 8，CUDA 分配器限额为 20,000,000,000 bytes（十进制 20 GB）。三轮顺序为原逐图调用→采样计划、采样计划→原逐图调用、原逐图调用→采样计划；以下按方案列出同轮计时。

| Backend | 轮次 | 原逐图调用 / s | 复用采样计划 / s |
|---|---|---:|---:|
| TBSS | 首轮，含冷启动 | 0.702855815 | 0.248454907 |
| TBSS | 热调用，第2轮 | 0.359119744 | 0.214557394 |
| TBSS | 热调用，第3轮 | 0.372348846 | 0.147125627 |
| MMORF | 首轮，含冷启动 | 0.645693060 | 0.224990048 |
| MMORF | 热调用，第2轮 | 0.469982618 | 0.284881951 |
| MMORF | 热调用，第3轮 | 0.564293495 | 0.255300635 |

首轮单列，后两轮的中位数如下。

| Backend | 原逐图调用 / s | 复用采样计划 / s | 本例热调用速度比 |
|---|---:|---:|---:|
| TBSS | 0.365734295 | 0.180841511 | 2.02× |
| MMORF | 0.517138056 | 0.270091293 | 1.91× |

三轮全部通过逐值 gate：九张图的完整解码体素、完整 header、affine、shape 和 dtype 一致；TBSS 额外的 valid_mask、FA 有效 mask、skeleton mask 和后处理 standard/skeleton 九图也完全一致。TBSS 保留 float64 几何计算、float32 采样；MMORF 保留原 float64 变换与 float32 基础网格、最终采样坐标、图像及 reference axes 极分解的混合精度。

新方案时间包含创建采样计划，未将准备时间另行扣除。输入加载、FA 预处理、输出 NIfTI 写盘与配准估计不在上述计时中。这是本例固定 warp 的九图传播测量；完整 pipeline 的验收与其他优化结果见[主报告](lossless_20261002.md)。

## 函数正确性检查

2026-10-02 本地定向单测已检查空间几何漂移、完整 header/extensions 漂移、边界和模式固定、3D/4D、dense/coefficient 及 TBSS 同网格分组。另在 CPU/CUDA 对冻结原源码进行 56 组数值 gate，voxel/header/affine/valid/QC 逐值一致。上述函数测试用于覆盖输入与接口边界；耗时结论使用本页真实九图测量。
