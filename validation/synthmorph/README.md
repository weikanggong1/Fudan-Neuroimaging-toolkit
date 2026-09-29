# SynthMorph 验证资产

本目录保存 12 例配准对照、公开示例，以及新增 FSL warp 转换的真实 T1w 对照。12 例 CPU/GPU 报告由以下源码生成，报告中的原始 hash 保持不变：

- `pipeline.py`：`e680d3…`
- `spatial.py`：`9c629a…`

当前 `pipeline.py` 和 `spatial.py` 分别为 `70e97c…` 和 `dab615…`；`SynthMorph.__call__` 和 registration 使用的 linear 路径未因 FSL warp 转换而改变。[源码等价证明](../runtime_dependencies/synthmorph_linear_source_equivalence.public.json)记录旧报告与此路径的关系。12 例报告没有重新测量新增的 `--fsl-warp` CLI 分支；它使用下面的独立真实数据报告验证。

| 文件 | 内容 |
|---|---|
| `report.real.current.gpu.json` | 12 例真实 T1w 的 FNIT/FreeSurfer GPU 对照；保留测量源码 hash、逐例指标、时间和显存。 |
| `report.real.current.cpu.json` | 同一 12 例的 CPU 对照；保留测量源码 hash、逐例指标和时间。 |
| `public_example.current.json` | OpenNeuro ds000114 去面部 T1w 示例的输入、输出、源码和图片 hash。 |
| `current_regression.py` | 重新生成真实数据报告的验证脚本；需要验证者自行提供数据和参考输出。 |
| `report.fsl_warp.real.json` | 一对公开真实 T1w：RAS→FSL 转换后与 TorchApplyWarp、FSL applywarp 的精度和时间。 |
| `validate_fsl_warp.py` | 上述转换验证的完整单被试脚本；只有 benchmark 分支调用 FSL。 |
| `SHA256SUMS` | 本目录报告、脚本、说明和公开示意图的 SHA-256。 |

## FSL warp 转换复核

仓库自带的 OpenNeuro ds000114 去面部 T1w 可直接复核。以下命令在 gpucw1 H100 上运行；`module load fsl` 只用于生成对照，FNIT 正常调用不需要 FSL。`--weights` 指向 `fnit-setup-weights` 下载的官方 SynthMorph 权重目录。

```bash
module load fsl
PYTHONPATH=src python validation/synthmorph/validate_fsl_warp.py \
  --moving examples/data/sub-02_T1w.nii.gz \
  --fixed examples/data/sub-01_T1w.nii.gz \
  --weights /path/to/fnit-weights \
  --output-dir /tmp/synthmorph-fsl-check \
  --device cuda:0 \
  --fsl-applywarp "$(command -v applywarp)"
```

输出目录包含 `synthmorph_moved.nii.gz`、`synthmorph_fsl_warp.nii.gz`、`torch_applywarp.nii.gz`、`fsl_applywarp.nii.gz`、`report.json` 和 `synthmorph_fsl_warp_comparison.png`。报告中的配准时间从模型加载完毕后开始，Torch/FSL applywarp 时间均包括各自读图和写盘。图像对照见[功能页面](../../docs/synthmorph/README.md)。

## linear registration 与 nearest apply 的证据边界

12 例报告和公开示例验证的是 registration 输出。`SynthMorph.__call__` 在网络空间预处理、变换组合和 moved image 输出中使用 `linear` 插值；这些结果可按源码等价证明继承到当前版本。

`apply_transform(method="nearest")` 面向离散标签。当前实现按 Surfa 0.6.3 使用 `floor(x + 0.5)`，有效坐标域为 `[0, size)`。这条路径由以下定向测试验证：

```bash
pytest -q tests/synthmorph/test_spatial.py \
  tests/synthmorph/test_pipeline_geometry.py
```

最终源码快照上，headcw CPU 为 `16 passed in 1.63 s`，gpucw1 H100 为 `26 passed in 4.00 s`。这些是半体素、边界、affine/dense warp 和 CPU/GPU 行为测试，不是新的真实标签数据 benchmark，也不能借用 12 例 registration 指标声明 nearest 的真实数据数值等价。
