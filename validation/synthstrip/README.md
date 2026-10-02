# SynthStrip 真实图像跨进程验证

`check_repeatability.py` 在新的 Python 进程中重复处理同一份真实三维影像，核查卷积选择策略、权重与模型状态、归一化网络输入、预测、原网格脑掩膜和毫米 SDT。它直接调用 FNIT `SynthStrip`，不启动原 FreeSurfer 程序。

成熟子函数原先同时设置 `cudnn.benchmark=True` 和 `cudnn.deterministic=True`。同一真实 T1 的两个新进程在模型状态和网络输入哈希相同的情况下，有 19 个掩膜边界体素不同，SDT RMSE 为 0.0002128 mm；仅在构造后关闭 benchmark，两个新进程的掩膜和 SDT 逐值相同。因此默认策略改为 `benchmark=False`，保留 deterministic 和 TF32。这个控制检查 FNIT 在相同环境下的重复性；与原程序的独立精度对照见[功能文档](../../docs/synthstrip/README.md)。

## 最新源码快照的原程序对照

[latest_native_comparison.public.json](latest_native_comparison.public.json)只读取 `cfb7beee` 完整真实流程、原 FreeSurfer 和冻结 `1eb9c417` 的已保存三维输出，不重新推理或运行原软件。运行时 `pipeline.py` SHA-256 严格匹配 `0ae5d3e35c2cd78f8a399b8c32d82164be657800db592c8b0e2edff023788fa1`；实际官方 checkpoint 匹配 `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33`，与原程序相同。当前和原程序输入文件哈希逐项一致。

| 完整三维指标 | EPI | T1 |
|---|---:|---:|
| mask Dice / 不同体素数 | 0.999994968 / 1 | 0.999998569 / 4 |
| FNIT / 原脑内体素数 | 99,371 / 99,372 | 1,397,746 / 1,397,746 |
| 脑图完整三维 RMSE / spatial r | 未单独捕获 | 0.389388 / 0.999999655 |
| 脑掩膜并集内脑图 RMSE / spatial r | — | 0.824671 / 0.999996445 |
| FNIT 子函数阶段 / 原独立进程（s） | 1.1654 / 8.5569 | 1.4442 / 9.3769 |

当前 T1 脑图和 mask 与冻结图示输入逐值相同，文件 SHA-256 相同，原 FNIRT 固定变换的示例图继续有效。FNIT 子函数使用已构造的模型，阶段包含影像处理、推理及结果保存；原子进程还包含启动和模型加载，分别记录计时边界。完整 EPI 脑图没有被当前流程单独捕获，不以其他阶段影像替代它。

### 只读保存结果比较

[compare_saved_outputs.py](compare_saved_outputs.py)直接读取完整三维 NIfTI，核查 binary mask、形状和 affine；检查当前 API 捕获的源码哈希、实际权重哈希及两份真实输入哈希，再计算 mask Dice 和完整图像/脑掩膜并集内的脑图 RMSE、空间 r。公开 JSON 不包含临床路径、affine、坐标或体素数组。

| 参数 | 输入与输出 |
|---|---|
| `--epi-mask`、`--native-epi-mask` | 必需：当前和原程序的同网格 EPI 二值掩膜。 |
| `--t1-mask`、`--native-t1-mask` | 必需：当前和原程序的同网格 T1 二值掩膜。 |
| `--t1-brain`、`--native-t1-brain` | 必需：同一 T1 网格的脑图。 |
| `--epi-brain`、`--native-epi-brain` | 可选，需同时提供：比较已保存的同网格 EPI 脑图。 |
| `--frozen-t1-mask`、`--frozen-t1-brain` | 可选，需同时提供：确认当前 T1 输出与冻结图示输入是否逐值相同。 |
| `--source-file` | 必需：当前运行的 SynthStrip `pipeline.py`；其 SHA-256 必须与 API 报告匹配。 |
| `--weights` | 必需：实际使用的官方 checkpoint 文件，计算完整 SHA-256。 |
| `--api-report`、`--native-report` | 必需：当前完整 API 和原 anatomy JSON，读取来源哈希及已测量的阶段/进程时间。 |
| `--source-revision` | 必需：当前运行源码 revision，须与 API 报告一致。 |
| `--expected-source-sha256`、`--expected-weights-sha256` | 可选：额外指定预期源码和权重哈希，任一不符即停止。 |
| `--report` | 必需：尚不存在的匿名输出 JSON；`-` 将 JSON 写到标准输出。 |

```bash
# 所有原图留在验证服务器；只读比较，不重新运行模型或原软件。
current_epi_mask=/path/to/current/feat/mask.nii.gz
original_epi_mask=/path/to/original/masks/epi_mask.nii.gz
current_t1_mask=/path/to/current/masks/T1_mask.nii.gz
original_t1_mask=/path/to/original/anat/T1_mask.nii.gz
current_t1_brain=/path/to/current/masks/T1_brain.nii.gz
original_t1_brain=/path/to/original/anat/T1_brain.nii.gz
frozen_t1_mask=/path/to/frozen/masks/T1_mask.nii.gz
frozen_t1_brain=/path/to/frozen/masks/T1_brain.nii.gz
runtime_synthstrip_source=/path/to/current/source/src/fnit/synthstrip/pipeline.py
synthstrip_checkpoint=/path/to/synthstrip.1.pt
current_api_report=/path/to/current/api.public.json
original_anatomy_report=/path/to/original/anatomy.public.json
anonymous_native_comparison=/path/to/reports/fresh_synthstrip_native.public.json

python validation/synthstrip/compare_saved_outputs.py \
  --epi-mask "$current_epi_mask" --native-epi-mask "$original_epi_mask" \
  --t1-mask "$current_t1_mask" --native-t1-mask "$original_t1_mask" \
  --t1-brain "$current_t1_brain" --native-t1-brain "$original_t1_brain" \
  --frozen-t1-mask "$frozen_t1_mask" --frozen-t1-brain "$frozen_t1_brain" \
  --source-file "$runtime_synthstrip_source" --weights "$synthstrip_checkpoint" \
  --api-report "$current_api_report" --native-report "$original_anatomy_report" \
  --source-revision cfb7beee202f7e89072faf1a8e69b78d143e451f \
  --report "$anonymous_native_comparison"
```

## 输入、输出与参数

| 参数 | 含义 |
|---|---|
| `--image` | 必需：真实三维 NIfTI，例如 T1w 或 SBRef。 |
| `--weights` | 必需：已下载的官方 PT 权重文件或目录；实际解析出的文件计算 SHA-256。 |
| `--preceding-image` | 可选：使用同一模型先提取此三维影像，再提取 `--image`；可复现 fMRI 中 EPI→T1 的顺序。 |
| `--reference-mask` | 可选：同网格、相同 affine 的已保存掩膜，报告与各次输出的 Dice、体素数及边界差异。 |
| `--private-output` | 必需：尚不存在的目录；保存各进程的掩膜、SDT 和日志。 |
| `--report` | 必需：尚不存在的 JSON 文件；仅保存匿名标量和哈希，不含影像路径、affine、坐标或体素数组。 |
| `--device` | 默认 `cuda:0`；可指定 `cuda:N` 或 `cpu`。 |
| `--threads` | 默认 8；当前进程的 PyTorch CPU 线程数。 |
| `--border` | 默认 1 mm；掩膜采用 `SDT < border`。 |
| `--no-csf` | 使用官方 no-CSF 权重；默认使用标准权重。 |
| `--repeats` | 每种策略的新进程数量，默认 2，最少 2。 |
| `--compare-autotune` | 额外测试旧 `benchmark=True` 策略；默认只测试当前构造策略。 |

默认运行两个新进程。私有输出按 `default_1`、`default_2` 等目录保存 `mask.private.nii.gz`、`distance.private.nii.gz` 和单次报告；开启旧策略对照时增加 `benchmark_on_*`。公开 JSON 中 `runs` 记录实际模型与输入哈希、策略、设备和诊断计时，`pair_comparisons` 记录逐值一致性、Dice、掩膜差异体素数与 SDT RMSE，`reference_mask_comparisons` 记录可选保存掩膜的对照。

## 完整变量名与示例

```bash
# 同一份真实 T1、流程实际保存的 EPI 参考和已验证官方权重。
anatomical_t1_image=/path/to/T1w.nii.gz
preceding_epi_reference=/path/to/reference_epi.nii.gz
synthstrip_checkpoint=/path/to/synthstrip.1.pt
saved_anatomical_mask=/path/to/frozen/T1_mask.nii.gz
private_repeatability_output=/path/to/private/fresh_synthstrip_repeatability
anonymous_repeatability_report=/path/to/reports/fresh_synthstrip_repeatability.public.json

# 两个新进程检查当前策略；影像和日志只写入私有目录。
PYTHONPATH=src python validation/synthstrip/check_repeatability.py \
  --image "$anatomical_t1_image" --weights "$synthstrip_checkpoint" \
  --preceding-image "$preceding_epi_reference" \
  --reference-mask "$saved_anatomical_mask" \
  --private-output "$private_repeatability_output" \
  --report "$anonymous_repeatability_report" \
  --device cuda:0 --threads 8 --repeats 2
```

省略 `--preceding-image` 检查单幅影像的独立调用；省略 `--reference-mask` 仅比较新进程间的结果。另建输出目录并加入 `--compare-autotune`，可在同一份输入上重现旧策略诊断。对应原程序脑提取命令为 `mri_synthstrip -i T1w.nii.gz -m T1_mask.nii.gz -d T1_sdt.nii.gz`；原程序的掩膜需提前生成，此驱动不会调用它。

驱动为记录实际张量哈希增加了 GPU→CPU 复制；调用时间包含这些观测开销，新进程时间还包括 Python 启动、模型加载和写盘，均不作为生产推理速度。`inputs_unchanged` 核对输入文件在运行前后没有改变。比较需同时确认实际 checkpoint、模型状态和网络输入哈希相同，再解读输出的重复性。

## 最近版本与报告结构

| 版本 | 测量记录与范围 |
|---|---|
| `cfb7beee` | 当前完整原程序对照见 [latest_native_comparison.public.json](latest_native_comparison.public.json)：EPI/T1 mask 差异为 1/4 体素；当前 T1 图示输入与冻结版逐值相同。FNIT 子函数为 1.1654/1.4442 秒，原独立进程为 8.5569/9.3769 秒，各自保存计时边界。 |
| `44364a8` | 默认固定卷积选择。当前验收使用本目录通用驱动；[cudnn_repeatability.public.json](cudnn_repeatability.public.json)顶层保存当前源码、输入/权重/模型状态/预测哈希和新进程比较，`original_diagnostic` 保存修复前的四进程因果控制。调用及进程计时读取对应字段，均包含前文所列验证开销；不包含完整 fMRI 处理。 |
| `1eb9c417` | 包含 `1db5917` 的几何修复。历史真实 SBRef/T1 几何控制为 7.61/4.83 秒，含模型构造、预测、两种回采样和比较，未含 Python 启动、输入 conform/归一化或保存；见[冻结报告](../fmri/synthstrip_geometry_control.public.json)。 |

实现来源及算法选择说明：[FNIT SynthStrip](../../src/fnit/synthstrip/pipeline.py)、[原 FreeSurfer 实现](https://github.com/freesurfer/freesurfer/blob/d932c45/mri_synthstrip/mri_synthstrip)、[PyTorch 2.5.1 随机性与 cuDNN 策略](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)。参考文献：Hoopes et al. *SynthStrip: Skull-Stripping for Any Brain Image*. NeuroImage, 2022，[DOI](https://doi.org/10.1016/j.neuroimage.2022.119474)。
