# 全局 SFF、H/K 写回、曲率与法向先验准备

本目录记录独立阶段接口和已执行验证。默认生产调用不变，未运行新 head global observer，没有把后态参考送入计算。

## 已完成

- 全局 SFF 复用成熟 VNL/SVD/EISPACK 原语，保留 global 特有 tangent、rip、有序有效行及零矩阵行为。
- 独立完整 H/K 写回与原表面摘要；不覆盖 `curv/curvbak/valbak`，Ktotal 保留逐顶点写回 F32。
- 曲率先验复用现有 `curvature_histograms()`，normal-dot 先验使用另一几何阶段并保留源表达式。
- 43 项本地合同及 CLI 边界检查，10.53 秒，逐个模块 SHA 见 [contracts_v3.json](reports/contracts_v3.json)。此处是算子契约，不是 MRI benchmark。
- 真实472顶点校准生成的白/灰 rawcounts：旧/新 sigma2 输出、旧已保存产物及新重复全部 0 值差、0 字节差。报告绑定旧提交 `e87bcbd4790d140074d9f2edae9acb8adb28ed05` 和新模块 SHA，见 [real_smoothing_v1.json](reports/real_smoothing_v1.json)。该测试只确认共享函数 256-bin 兼容，不作为 full global stage 通过。

## 相位

源第一全局 SFF/PDF 在 raw TMP current 上，normal-dot 在 smoothed ORIGINAL/metric/vnum 上。`mrisComputeSurfaceStatistics()` 另做一次 smoothed-original SFF，位于 base copy 之后。旧初始统计 K 不能作为 SFF1 same-input。详见 [source_phase_audit.json](reports/source_phase_audit.json)。

既有3a DVS record 包含 corrected base 的 K1/K2，可作为自产同3a raw→SFF1→base映射后的 after 子集；没有 H/K、TMP/canonical，clone e1/e2 也非 SFF1 复制字段。必须自产完整上游和映射后再比较，不将 record 的坐标/法向/K 作为上游输入，也不混入79病例。

## 尚未完成

global SFF/HK 和 normal-dot 真实 before/after、完整参考重复、脑图、完整 GA、原始T1整例尚未执行。新 observer 上传许可待确认；共享平滑回归的时间不外推完整阶段速度。未测显存，未执行 CUDA，不改 TF32 或半精度。

## 复现

```bash
# 本地合同；显式限制Numba/OpenMP线程，非MRI benchmark
PYTHONPATH=src OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 python -m pytest -q \
  tests/recon_all/test_topology_global_sff.py \
  tests/recon_all/test_topology_global_dot_prior.py \
  tests/recon_all/test_topology_global_priors_benchmark.py
```

真实平滑工具 `scripts/compare_histogram_smoothing.py` 的全部参数为：`--input` 为自产校准 NPZ；`--old-source/--new-source` 为冻结 FNIT 函数文件；`--source-manifest` 为对应两个文件 SHA 和旧提交的清单；`--output` 为新报告路径。仅读 `calibration_white_raw/gray_raw/white_smooth/gray_smooth/selected`，不读 MRI、不修改旧产物。完整参数示例：

```bash
python scripts/compare_histogram_smoothing.py \
  --input frozen_calibration_candidate.npz \
  --old-source old_histogram.py \
  --new-source new_histogram.py \
  --source-manifest SOURCE_MANIFEST.json \
  --output new_smoothing_report.json
```

七节接口说明见 [全局 SFF](../../../../docs/recon_all/TOPOLOGY_GLOBAL_SFF.md) 和 [法向先验](../../../../docs/recon_all/TOPOLOGY_GLOBAL_DOT_PRIOR.md)。原源 SHA 与规则保留，原源主体不在本目录发布。
