# 合并后的相关 CPU 回归检查

## 1. 检查范围

检查 `7a9b369b` 合并后的 connectome，以及本轮直接影响的 EDDY、FLIRT 和 MGH 几何接口。它不是全仓库测试，也不提供 GPU 或真实影像耗时。

## 2. 输入与输出

输入为该提交的源码、测试、测试使用的 atlas 和归档夹具。输出为实际服务器的 `report.json`、`launch.json`、标准输出及源文件 SHA；[verification.json](verification.json) 绑定全部 15 个实际文件。

## 3. 运行命令

```bash
# 在包含完整测试夹具的 FNIT checkout 中运行；本检查关闭 GPU。
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/connectome \
  tests/eddy/test_ukb_reference_reuse.py \
  tests/eddy/test_ukb_synthstrip_mask.py \
  tests/recon_all/test_mgh_new_image_geometry.py \
  tests/flirt/test_output_semantics.py
```

## 4. 原软件调用

这些回归检查不调用原软件。真实 FSL、FreeSurfer 和 MRtrix 对照另见[十例评测协议](../../../../docs/connectome/raw_cohort_benchmark.md)。

## 5. 实际结果

**642 项测试、354 项子测试通过，28 项跳过。** pytest 耗时 41.70 s，含启动和记录的驱动耗时 42.422 s。跳过项不记为 GPU 测试通过。见[最终报告](attempt_3/report.json)。

本轮测量使用的 connectome、TOPUP、EDDY、FLIRT、SynthMorph、SynthStrip 和 CLI 计算源码，与 `641f16b2` 相比无修改。

## 6. 本次检查记录

第一次归档缺少下载工具和相关夹具，收集失败；第二次缺少 `assets/connectome`，641 项通过、1 项失败。第三次补齐归档后通过。三次记录各自保留，过程中没有修改产品源码或测试判断。

## 7. 来源

源码提交 `7a9b369bd8fc55996a80d1cb17222cc8e1ce395c`；实际执行在 headcw、FNIT Conda Python 3.11 环境。完整命令、源码归档和输出 SHA 见各次 `launch.json`、`report.json` 及汇总验证记录。
