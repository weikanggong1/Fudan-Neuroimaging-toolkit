# 模型 I/O 当前源码回归工具

[`measure_peak.py`](measure_peak.py) 对 WMH-SynthSeg、33 类 SynthSeg、SynthSR 或 TorchFAST 运行一个真实病例。脚本通过各功能的公开 Python API 构造一次模型、推理一次、保存全部主要影像，并写 JSON：

- 输入 SHA-256；
- 参与运行的 FNIT 源文件逐文件 SHA-256 和组合树哈希；
- PyTorch、CUDA、GPU、线程及 TF32 状态；
- 模型加载、推理、保存和 API 总时间；
- Torch allocated/reserved 峰值显存；
- 每幅输出的文件名、shape、dtype、affine、有限值状态和 SHA-256。qform/sform code 由各功能报告内的多病例输出清单补充记录。

示例：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python validation/model_io_current/measure_peak.py \
  --feature synthseg \
  --input examples/data/sub-01_T1w.nii.gz \
  --weights /path/to/official/weights \
  --device cuda:0 \
  --threads 8 \
  --case-id case01 \
  --source-root src \
  --output-dir work/peak/synthseg \
  --report work/peak/synthseg.json
```

`--feature` 可选 `wmh-synthseg`、`synthseg`、`synthsr` 或 `fast`。FAST 不使用权重，应省略 `--weights`。脚本不调用 FreeSurfer 或 FSL；参考输出比较由各功能验证页记录的独立脚本完成。
