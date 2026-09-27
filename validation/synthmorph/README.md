# SynthMorph 验证资产

本目录保存最近一次完整真实数据测量和公开示例。12 例 CPU/GPU 报告由以下源码生成，报告中的原始 hash 保持不变：

- `pipeline.py`：`e680d3…`
- `spatial.py`：`9c629a…`

当前发布源码分别为 `70e97c…` 和 `dab615…`。本轮改动只增加 `apply_transform(method="nearest")` 的 Surfa 0.6.3 最近邻规则；`SynthMorph.__call__` 和 registration 使用的 linear 路径不变。[源码等价证明](../runtime_dependencies/synthmorph_linear_source_equivalence.public.json)记录完整 hash、linear 路径 AST 指纹和测试边界。当前 hash 没有重新跑 12 例完整 registration，因此不能把源码继承写成 fresh current-source benchmark。
报告记录的根 CLI `2d5e3d…` 与当前 `c92a3f…` 之间，SynthMorph 参数定义和分发 AST
完全一致；完整入口证明见[包入口源码等价证明](../runtime_dependencies/package_entry_source_equivalence.public.json)。

| 文件 | 内容 |
|---|---|
| `report.real.current.gpu.json` | 12 例真实 T1w 的 FNIT/FreeSurfer GPU 对照；保留测量源码 hash、逐例指标、时间和显存。 |
| `report.real.current.cpu.json` | 同一 12 例的 CPU 对照；保留测量源码 hash、逐例指标和时间。 |
| `public_example.current.json` | OpenNeuro ds000114 去面部 T1w 示例的输入、输出、源码和图片 hash。 |
| `current_regression.py` | 重新生成真实数据报告的验证脚本；需要验证者自行提供数据和参考输出。 |
| `SHA256SUMS` | 本目录报告、脚本、说明和公开示意图的 SHA-256。 |

## linear registration 与 nearest apply 的证据边界

12 例报告和公开示例验证的是 registration 输出。`SynthMorph.__call__` 在网络空间预处理、变换组合和 moved image 输出中使用 `linear` 插值；这些结果可按源码等价证明继承到当前版本。

`apply_transform(method="nearest")` 面向离散标签。当前实现按 Surfa 0.6.3 使用 `floor(x + 0.5)`，有效坐标域为 `[0, size)`。这条路径由以下定向测试验证：

```bash
pytest -q tests/synthmorph/test_spatial.py \
  tests/synthmorph/test_pipeline_geometry.py
```

最终源码快照上，headcw CPU 为 `16 passed in 1.63 s`，gpucw1 H100 为 `26 passed in 4.00 s`。这些是半体素、边界、affine/dense warp 和 CPU/GPU 行为测试，不是新的真实标签数据 benchmark，也不能借用 12 例 registration 指标声明 nearest 的真实数据数值等价。
