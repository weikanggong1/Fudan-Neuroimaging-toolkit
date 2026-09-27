# MS-HBM 真实数据验证

本目录保存 2026-09-27 的 MSC02 五分钟 fsLR32k 静息态对照。公开报告为 [`report.public.json`](report.public.json)，表面图位于 [`docs/mshbm/figures/mshbm_cbig_comparison.png`](../../docs/mshbm/figures/mshbm_cbig_comparison.png)。原始时序、CBIG 中间 profile 和 MATLAB 标签不进入仓库；报告记录原始时序 SHA-256。

## 输入和比较边界

- FNIT 当前源码读取 `100×59412` 的 `float32` 时序，并按前后各 50 frame 拆成两个 pseudo-session。
- CBIG 参考使用 Kong2019 MS-HBM commit `b69b822a15e2a94f1e439606552fc44b6858cf3c` 与 MATLAB R2018b。
- 精度比较包含两节 `59412×1483` 二值 profile 和最终 `64984` 个表面标签。
- 计时比较从同一对已保存 profile 开始，到标签写出结束；两边均使用 8 线程并包含解释器启动和文件 I/O。

## 复现 FNIT 输出

下列变量均需由复现者指向自己的数据；命令不会调用 MATLAB 或 CBIG。

```bash
FNIT_SOURCE=/path/to/Fudan-Neuroimaging-toolkit  # 当前 FNIT 仓库根目录
CASE_DIR=/path/to/msc02_validation               # 单被试验证目录

OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
PYTHONPATH="$FNIT_SOURCE/src" \
/usr/bin/time -v python -m fnit.mshbm.cli \
  --timeseries "$CASE_DIR/short_5min.npy" \
  --output-dir "$CASE_DIR/fnit_current" \
  --w 200 \
  --c 50
```

`--timeseries` 是 `time×59412` 或 `time×64984` 的单被试 `.npy`，也可传 fsLR32k `.dtseries.nii`；`--output-dir` 保存左右半球和完整标签；`--w` 与 `--c` 对应 CBIG 的 group prior 和 MRF 权重。

## 相同 profile 的 matched 计时

[`run_fnit_from_cbig_profiles.py`](run_fnit_from_cbig_profiles.py)只用于验证，读取 CBIG 保存的 `profile_sess1.mat` 和 `profile_sess2.mat`：

```bash
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
PYTHONPATH="$FNIT_SOURCE/src" \
/usr/bin/time -v python validation/mshbm/run_fnit_from_cbig_profiles.py \
  --profile "$CASE_DIR/profile_sess1.mat" \
  --profile "$CASE_DIR/profile_sess2.mat" \
  --output "$CASE_DIR/fnit_matched.npz" \
  --w 200 \
  --c 50
```

`--profile` 每节 session 重复一次；`--output` 保存 `labels` 和收敛 `history_json`。CBIG 侧以相同两份 profile 调用官方 subject-level inference，并用 `/usr/bin/time -v` 包住 MATLAB 进程。报告脚本的全部输入参数可通过以下命令查看：

```bash
python validation/mshbm/compare_real.py --help
```

报告绑定的 `core.py`、`cli.py`、资产和两个验证脚本 SHA-256 均写在 `source_sha256`。当前结果为两节 profile 各 88,107,996 个值全同、64,984 个标签全同；matched 计时 FNIT 141.29 秒、CBIG 145.71 秒。
