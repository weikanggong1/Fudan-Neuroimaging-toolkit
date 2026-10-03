# Raw stage 完成判断的独立 CPU 回归

## 范围与版本

仅检查 staging、TOPUP 与 EDDY 完成 marker 的产物身份、完整性、原子发布、旧 marker 迁移和失败顺序。数值生产器以 stub 替代，MRI 是小型模拟输入，不是影像精度或性能 benchmark。

- v1：`ac0e7d9d12bd730d33135d36d3f218990444d436`，实际50 passed/2 failed，记录保留。
- v2：`22df4013b26a748b466dde48b80d1aee5851caa9`，实际52 passed，退出0。
- 独立目录：已登记 `FNIT/workspaces/connectome_paired_20261003_v1/task_03/raw_stage_cache_cpu_v1` 和 `raw_stage_cache_cpu_v2`；未写冻结科学目录、MRI 产物或既有 Conda prefix。

## 复现

在包含对应版本 `src/`、`tests/`、`pyproject.toml` 的独立 source 根目录中执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONPATH=src \
taskset -c 4-7 python -m pytest -q \
  tests/connectome/test_raw_stage_cache_integrity.py \
  tests/connectome/test_bids.py \
  tests/connectome/test_cli_numerical_revision.py \
  tests/connectome/test_paired_e2e.py
```

各版本附带的 `driver.py` 是实际执行脚本：须放到对应源码根目录，与 `src/`、`tests/` 同级，再以上述环境/affinity运行。不要直接在本证据目录执行；该目录只保存小报告，没有完整冻结源码。驱动记录实际 import、源码/测试 SHA、源内容聚合 SHA 前后、CUDA 初始化和 pytest 时间。

## 失败与修正

1. 首轮 stat 前后未能检出 shared FS 上保留 mtime 的同大小编辑。v2 增加第二次内容哈希确认，仍严格拒绝不稳定产物。
2. 首轮 staged AP.bval 的直接写入穿过成熟 staging 的 symlink 修改了原始 fixture。v2 原子替换 staged 文件，明确断言原始 bval不变，测试真正的 output-only damage。

二者的初始失败均未覆盖、删除或改标。现有非背景/内容匹配/失败门没有放宽。TOPUP/EDDY/recon/staging 生产调用 AST 对基线 `72f1802f` 全部相同，原核对保存在 `producer_call_ast_review.json`。

## 实际结果与边界

v2：52 passed in5.55s，pytest.main5.796754661947489秒，完整驱动9.019502708688378秒；线程4、affinity4–7，CUDA 初始化前后False，1215个源码内容聚合 SHA前后相同。新完整性测试20条＋旧BIDS/CLIrevision/配对E2E32条。时间属于 CPU 单元回归，不表示 raw pipeline 加速或数值等效。

新产物身份检查增加哈希 I/O；旧无输出 SHA 的 raw stage marker 需要重新建立一次。显式 supplied corrected DWI/bvec 分支保持原行为。旧命名 atlas CLI 的通用 fingerprint-only helper兼容没有改为新的 raw stage 完整性合同。

## 复制文件完整性

| 文件 | SHA-256 |
|---|---|
| `v1/focused_cpu_report.json` | `773a1870ddbac262658f7eba33b615949712bf2af8c4823394c23fd32929baa8` |
| `v1/focused_cpu.log` | `16a8827937d368d8591908f5f69e1b58610e079f6a9fc06ce66f8ed8bd81afb3` |
| `v2/focused_cpu_report.json` | `a0d362c8b6e16ea8e89eb4cb31be063da9da45957338ef49996177f9f7ff9d59` |
| `v2/focused_cpu.log` | `95c0dcd50c8cc212c92652a23cf7a2b857629d338ecab1749de62bbee330c11d` |
| `producer_call_ast_review.json` | `f7e73c3267104bfa1574dc0cbff5c1eea0d2ed340e7e7cca4963f97830741e3f` |

未复制真实 MRI 或真实输入影像哈希；公开证据仅含测试结果与代码 provenance。
